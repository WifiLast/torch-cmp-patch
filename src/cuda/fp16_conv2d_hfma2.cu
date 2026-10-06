// fp16 conv2d (implicit GEMM) for Turing CMP cards, HFMA2 only.
//
// Same reasoning as fp16_flash_attention_hfma2.cu: HMMA is throttled to
// ~3.3 TFLOPS on the CMP 50HX and FFMA to ~0.4, HFMA2 runs at ~26. The older
// fp16_conv.cu reaches ~2 TFLOPS at SD1.5's shapes and accumulates the whole
// K = Ci*KH*KW reduction in fp16, which autoselect already rejects on at
// least one shape (3.5% error).
//
// GEMM view: out[co][pixel] = sum_k W[co][k] * X[k][pixel],
//            k = (ky, kx, ci) with ci fastest, pixel = (n, oy, ox) flattened.
// Block tile BM=64 output channels x BN=128 output pixels, 256 threads, each
// thread 8 channels x 4 pixels. The K loop walks (ky, kx) and, inside, CK=64
// input channels at a time:
//   - weights come pre-permuted to [Co][KH][KW][Ci] (contiguous along ci) and
//     are staged with 128-bit loads into As[m][ci];
//   - the input tile is gathered from NCHW into Bs[pixel][ci] (zero padding
//     and stride handled in the gather);
//   - "dot-pair" HFMA2: a half2 accumulator holds two partial sums along ci,
//     operands are read along ci with 128-bit loads, no broadcasts needed;
//   - after each CK chunk the half2 partials are added into fp32
//     accumulators (FADD; --fmad=false keeps fp32 math off FFMA). fp16 thus
//     only ever sums 32 products, which keeps the error at stock's level.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int BM = 64, BN = 128, CK = 64, NT = 256;
constexpr int TM = 8, TN = 4;          // per-thread tile
constexpr int LDS = CK + 8;            // 72 halves = 144 B rows: odd # of 16 B chunks

__device__ __forceinline__ void hfma2_dot8(half2& acc, const int4& a, const int4& b) {
    const half2* ha = reinterpret_cast<const half2*>(&a);
    const half2* hb = reinterpret_cast<const half2*>(&b);
    acc = __hfma2(ha[0], hb[0], acc);
    acc = __hfma2(ha[1], hb[1], acc);
    acc = __hfma2(ha[2], hb[2], acc);
    acc = __hfma2(ha[3], hb[3], acc);
}

__global__ void __launch_bounds__(NT, 2)
conv2d_hfma2_kernel(const half* __restrict__ X,      // [N][Ci][H][W]
                    const half* __restrict__ Wt,     // [Co][KH][KW][Ci]
                    const half* __restrict__ bias,   // [Co] or nullptr
                    half* __restrict__ Y,            // [N][Co][Ho][Wo]
                    int N, int Ci, int H, int W, int Co, int KH, int KW,
                    int sh, int sw, int ph, int pw, int dh, int dw, int Ho, int Wo) {
    __shared__ __align__(16) half As[BM * LDS];
    __shared__ __align__(16) half Bs[BN * LDS];

    const int tid = threadIdx.x;
    const int co0 = blockIdx.y * BM;
    const int64_t P = (int64_t)N * Ho * Wo;
    const int64_t p0 = (int64_t)blockIdx.x * BN;
    const int64_t HW = (int64_t)H * W;

    // loader: each thread gathers one pixel's column, 32 of the CK channels
    const int lp = tid % BN, lc = tid / BN;   // lc in {0, 1}: even/odd channel
    const int64_t gp = p0 + lp;
    const bool lp_ok = gp < P;
    int ln = 0, loy = 0, lox = 0;
    if (lp_ok) {
        ln = (int)(gp / ((int64_t)Ho * Wo));
        int rem = (int)(gp % ((int64_t)Ho * Wo));
        loy = rem / Wo; lox = rem % Wo;
    }
    const half* Xn = X + (int64_t)ln * Ci * HW;

    // compute mapping: warp = 8-channel row group, lane = pixel column
    const int tm = tid / 32, tn = tid % 32;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    const int KK = KH * KW;
    for (int kk = 0; kk < KK; ++kk) {
        const int ky = kk / KW, kx = kk % KW;
        const int iy = loy * sh - ph + ky * dh, ix = lox * sw - pw + kx * dw;
        const bool in_ok = lp_ok && iy >= 0 && iy < H && ix >= 0 && ix < W;
        const half* Xp = Xn + (int64_t)iy * W + ix;

        for (int ci0 = 0; ci0 < Ci; ci0 += CK) {
            // ---- stage weights: As[m][c] = Wt[co0+m][ky][kx][ci0+c] -------
            for (int i = tid; i < BM * (CK / 8); i += NT) {
                int m = i / (CK / 8), c = (i % (CK / 8)) * 8;
                int4 v = make_int4(0, 0, 0, 0);
                const int co = co0 + m, ci = ci0 + c;
                if (co < Co) {
                    const half* src = Wt + ((int64_t)co * KK + kk) * Ci + ci;
                    if (ci + 8 <= Ci && ((reinterpret_cast<uintptr_t>(src) & 15) == 0)) {
                        v = *reinterpret_cast<const int4*>(src);
                    } else {
                        half* hv = reinterpret_cast<half*>(&v);
#pragma unroll
                        for (int e = 0; e < 8; ++e) hv[e] = (ci + e < Ci) ? src[e] : __float2half(0.f);
                    }
                }
                *reinterpret_cast<int4*>(As + m * LDS + c) = v;
            }
            // ---- gather input: Bs[p][c] = X[n][ci0+c][iy][ix] ------------
#pragma unroll 8
            for (int c = lc; c < CK; c += 2) {
                const int ci = ci0 + c;
                half v = __float2half(0.f);
                if (in_ok && ci < Ci) v = Xp[(int64_t)ci * HW];
                Bs[lp * LDS + c] = v;
            }
            __syncthreads();

            // ---- 64 x 128 x CK dot-pair HFMA2 ----------------------------
            half2 hacc[TM][TN];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) hacc[i][j] = __float2half2_rn(0.f);
#pragma unroll 2
            for (int c = 0; c < CK; c += 8) {
                int4 af[TM], bf[TN];
#pragma unroll
                for (int i = 0; i < TM; ++i) af[i] = *reinterpret_cast<const int4*>(As + (tm * TM + i) * LDS + c);
#pragma unroll
                for (int j = 0; j < TN; ++j) bf[j] = *reinterpret_cast<const int4*>(Bs + (tn + 32 * j) * LDS + c);
#pragma unroll
                for (int i = 0; i < TM; ++i)
#pragma unroll
                    for (int j = 0; j < TN; ++j) hfma2_dot8(hacc[i][j], af[i], bf[j]);
            }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    float2 f = __half22float2(hacc[i][j]);
                    acc[i][j] += f.x + f.y;
                }
            __syncthreads();
        }
    }

    // ---- epilogue: + bias, store NCHW ------------------------------------
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int64_t p = p0 + tn + 32 * j;
        if (p >= P) continue;
        const int n = (int)(p / ((int64_t)Ho * Wo));
        const int64_t rem = p % ((int64_t)Ho * Wo);
        half* Yp = Y + (int64_t)n * Co * Ho * Wo + rem;
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const int co = co0 + tm * TM + i;
            if (co >= Co) continue;
            float r = acc[i][j];
            if (bias) r += __half2float(bias[co]);
            Yp[(int64_t)co * Ho * Wo] = __float2half_rn(r);
        }
    }
}

}  // namespace

// x: [N][Ci][H][W] contiguous, w_khwc: [Co][KH][KW][Ci] contiguous, y: [N][Co][Ho][Wo].
// groups must be 1. Returns false if the launch failed.
bool launch_conv2d_hfma2(const void* x, const void* w_khwc, const void* bias, void* y,
                         int N, int Ci, int H, int W, int Co, int KH, int KW,
                         int sh, int sw, int ph, int pw, int dh, int dw, int Ho, int Wo,
                         cudaStream_t stream) {
    const int64_t P = (int64_t)N * Ho * Wo;
    dim3 grid((unsigned)((P + BN - 1) / BN), (unsigned)((Co + BM - 1) / BM));
    conv2d_hfma2_kernel<<<grid, NT, 0, stream>>>(
        static_cast<const half*>(x), static_cast<const half*>(w_khwc), static_cast<const half*>(bias),
        static_cast<half*>(y), N, Ci, H, W, Co, KH, KW, sh, sw, ph, pw, dh, dw, Ho, Wo);
    return cudaGetLastError() == cudaSuccess;
}
