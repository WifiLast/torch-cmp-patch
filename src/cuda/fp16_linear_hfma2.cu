// fp16 linear (y = x W^T + b) for Turing CMP cards, HFMA2 only.
//
// Same design as fp16_conv2d_hfma2.cu, minus the gather: x rows and W rows
// are both contiguous along K, which is exactly the layout the "dot-pair"
// HFMA2 scheme wants (a half2 accumulator holds two partial sums along K;
// operands are read along K with 128-bit loads). Block tile 64 output
// features x 128 rows, 256 threads, 8 features x 4 rows per thread; half2
// partials are flushed into fp32 every 64 K, so fp16 only ever sums 32
// products. Output rows are written 8 features at a time (one 16 B store).

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int BM = 64, BN = 128, CK = 64, NT = 256;
constexpr int TM = 8, TN = 4;
constexpr int LDS = CK + 8;   // 144 B rows: odd number of 16 B chunks

__device__ __forceinline__ void hfma2_dot8(half2& acc, const int4& a, const int4& b) {
    const half2* ha = reinterpret_cast<const half2*>(&a);
    const half2* hb = reinterpret_cast<const half2*>(&b);
    acc = __hfma2(ha[0], hb[0], acc);
    acc = __hfma2(ha[1], hb[1], acc);
    acc = __hfma2(ha[2], hb[2], acc);
    acc = __hfma2(ha[3], hb[3], acc);
}

__global__ void __launch_bounds__(NT, 2)
linear_hfma2_kernel(const half* __restrict__ X, int64_t ldx,   // [N][K], row stride ldx
                    const half* __restrict__ Wt,               // [M][K] contiguous
                    const half* __restrict__ bias,             // [M] or nullptr
                    half* __restrict__ Y, int64_t ldy,         // [N][M], row stride ldy
                    int N, int M, int K) {
    __shared__ __align__(16) half As[BM * LDS];
    __shared__ __align__(16) half Bs[BN * LDS];
    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * BM;
    const int64_t n0 = (int64_t)blockIdx.x * BN;
    const int tm = tid / 32, tn = tid % 32;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < K; k0 += CK) {
        for (int i = tid; i < BM * (CK / 8); i += NT) {
            int r = i / (CK / 8), c = (i % (CK / 8)) * 8;
            int4 v = make_int4(0, 0, 0, 0);
            if (m0 + r < M && k0 + c < K) v = *reinterpret_cast<const int4*>(Wt + (int64_t)(m0 + r) * K + k0 + c);
            *reinterpret_cast<int4*>(As + r * LDS + c) = v;
        }
        for (int i = tid; i < BN * (CK / 8); i += NT) {
            int r = i / (CK / 8), c = (i % (CK / 8)) * 8;
            int4 v = make_int4(0, 0, 0, 0);
            if (n0 + r < N && k0 + c < K) v = *reinterpret_cast<const int4*>(X + (n0 + r) * ldx + k0 + c);
            *reinterpret_cast<int4*>(Bs + r * LDS + c) = v;
        }
        __syncthreads();

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

    // epilogue: 8 consecutive features per row -> one 16 B store
    const int mb = m0 + tm * TM;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int64_t n = n0 + tn + 32 * j;
        if (n >= N) continue;
        half* yrow = Y + n * ldy;
        if (mb + TM <= M) {
            int4 out;
            half* ho = reinterpret_cast<half*>(&out);
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                float r = acc[i][j];
                if (bias) r += __half2float(bias[mb + i]);
                ho[i] = __float2half_rn(r);
            }
            *reinterpret_cast<int4*>(yrow + mb) = out;
        } else {
            for (int i = 0; i < TM && mb + i < M; ++i) {
                float r = acc[i][j];
                if (bias) r += __half2float(bias[mb + i]);
                yrow[mb + i] = __float2half_rn(r);
            }
        }
    }
}

}  // namespace

// x: [N][K] rows with stride ldx (elements), w: [M][K] contiguous, y: [N][M] rows with
// stride ldy. Needs K % 8 == 0, M % 8 == 0 and 16-byte aligned rows (caller checks).
bool launch_linear_hfma2(const void* x, int64_t ldx, const void* w, const void* bias,
                         void* y, int64_t ldy, int N, int M, int K, cudaStream_t stream) {
    dim3 grid((unsigned)((N + BN - 1) / BN), (unsigned)((M + BM - 1) / BM));
    linear_hfma2_kernel<<<grid, NT, 0, stream>>>(
        static_cast<const half*>(x), ldx, static_cast<const half*>(w), static_cast<const half*>(bias),
        static_cast<half*>(y), ldy, N, M, K);
    return cudaGetLastError() == cudaSuccess;
}
