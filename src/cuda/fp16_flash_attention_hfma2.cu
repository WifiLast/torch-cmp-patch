// fp16 fused attention for Turing CMP cards, HFMA2 only.
//
// Why a new kernel: on the CMP 50HX the tensor cores (HMMA) are throttled to
// ~3.3 TFLOPS and FFMA to ~0.4 TFLOPS, while HFMA2 on the CUDA cores runs at
// full speed (~26 TFLOPS; tools/peak_units_v2.cu). Stock SDPA and cmpext3's
// older fp16 attention kernel both end up at 2-3 TFLOPS. The older kernel
// also only took head_dim 128 and self-attention (Sq == Sk), so none of
// SD1.5's attention (head_dim 40/80/160, cross-attention with 77*n tokens)
// ever reached it.
//
// Design (FlashAttention-2 style, forward only, non-causal, no mask):
//   - One block = BQ query rows; loop over key/value tiles of BK rows.
//   - S = Q K^T and O += P V both use "dot-pair" HFMA2: a half2 accumulator
//     holds two partial sums along the reduction axis (d for S, k for PV),
//     which are added in fp32 at the end. Operands are then read along the
//     reduction axis with 128-bit shared-memory loads and need no broadcast
//     shuffles. V is stored transposed (Vt[d][k]) for the same reason.
//   - Q is pre-scaled by softmax_scale * log2(e) when it is staged, so the
//     softmax runs in the exp2 domain (one MUFU.EX2 per score).
//   - Each half2 accumulator only ever sums one tile's worth of products
//     (D/2 terms for S, BK/2 for PV), so fp16 range/rounding stays bounded.
//     Running max, row sum and the output accumulator are fp32; with
//     --fmad=false (setup.py) their updates compile to FMUL + FADD, which
//     are not throttled, instead of FFMA.
//
// Layout: q/k/v are [B, H, S, D] *views* with arbitrary B/H/S strides and a
// contiguous last dim (diffusers/webui pass transposed views; no copy needed).
// The output is written with arbitrary strides as well.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int NT = 128;   // threads per block
constexpr int BK = 64;    // keys per tile
constexpr int TK = 8;     // keys per thread in the S phase (strided by 8)
constexpr int PADK = 8;   // row padding of P / Vt (BK+8 halves = 144 B: odd # of 16 B chunks)

__device__ __forceinline__ float fast_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void hfma2_dot8(half2& acc, const int4& a, const int4& b) {
    const half2* ha = reinterpret_cast<const half2*>(&a);
    const half2* hb = reinterpret_cast<const half2*>(&b);
    acc = __hfma2(ha[0], hb[0], acc);
    acc = __hfma2(ha[1], hb[1], acc);
    acc = __hfma2(ha[2], hb[2], acc);
    acc = __hfma2(ha[3], hb[3], acc);
}

template <int D, int BQ, int OQ, int OD>
struct Cfg {
    static constexpr int PADD = ((D / 8) % 2 == 0) ? 8 : 0;   // odd # of 16 B chunks per row
    static constexpr int LDQ = D + PADD;     // Qs / Ks row stride (halves)
    static constexpr int LDP = BK + PADK;    // Ps / Vts row stride (halves)
    static constexpr int TQ = BQ / 16;       // queries per thread in the S phase
    static constexpr int DG = D / OD;        // thread groups along d in the O phase
    static_assert(D % 8 == 0, "head_dim must be a multiple of 8");
    static_assert(NT == 16 * 8, "S-phase mapping assumes 16 x 8 threads");
    static_assert(BQ * BK == NT * TQ * TK, "S tile must cover BQ x BK");
    static_assert(BQ * D == NT * OQ * OD, "O tile must cover BQ x D");
    static_assert(D % OD == 0 && (NT % DG) == 0, "O-phase mapping");
    static constexpr int SMEM_HALVES = BQ * LDQ + BK * LDQ + D * LDP + BQ * LDP;
    static constexpr int SMEM_BYTES = SMEM_HALVES * 2 + 3 * BQ * 4;
};

template <int D, int BQ, int OQ, int OD>
__global__ void __launch_bounds__(NT)
flash_attn_hfma2_kernel(const half* __restrict__ Q, const half* __restrict__ K,
                        const half* __restrict__ V, half* __restrict__ O,
                        int H, int Sq, int Sk,
                        int64_t qsb, int64_t qsh, int64_t qss,
                        int64_t ksb, int64_t ksh, int64_t kss,
                        int64_t vsb, int64_t vsh, int64_t vss,
                        int64_t osb, int64_t osh, int64_t oss,
                        float qscale) {
    using C = Cfg<D, BQ, OQ, OD>;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    half* Qs = reinterpret_cast<half*>(smem_raw);          // [BQ][LDQ]
    half* Ks = Qs + BQ * C::LDQ;                            // [BK][LDQ]
    half* Vts = Ks + BK * C::LDQ;                           // [D][LDP]
    half* Ps = Vts + D * C::LDP;                            // [BQ][LDP]
    float* alpha_s = reinterpret_cast<float*>(Ps + BQ * C::LDP);  // [BQ]
    float* l_s = alpha_s + BQ;                                     // [BQ]

    const int tid = threadIdx.x;
    const int bh = blockIdx.y, b = bh / H, h = bh % H;
    const int q0 = blockIdx.x * BQ;
    const half* Qg = Q + b * qsb + h * qsh;
    const half* Kg = K + b * ksb + h * ksh;
    const half* Vg = V + b * vsb + h * vsh;

    // ---- stage Q (pre-scaled into the exp2 domain) -------------------------
    constexpr int CH = D / 8;   // 16-byte chunks per row
    for (int i = tid; i < BQ * CH; i += NT) {
        int r = i / CH, c = (i % CH) * 8;
        int4 v = make_int4(0, 0, 0, 0);
        if (q0 + r < Sq) v = *reinterpret_cast<const int4*>(Qg + (int64_t)(q0 + r) * qss + c);
        half2* hv = reinterpret_cast<half2*>(&v);
        const half2 s2 = __float2half2_rn(qscale);
#pragma unroll
        for (int j = 0; j < 4; ++j) hv[j] = __hmul2(hv[j], s2);
        *reinterpret_cast<int4*>(Qs + r * C::LDQ + c) = v;
    }

    // S-phase thread mapping: 16 row-groups x 8 key lanes (one row = 8 lanes)
    const int sq = tid / 8, sk = tid % 8;
    float m_run[C::TQ], l_run[C::TQ];
#pragma unroll
    for (int i = 0; i < C::TQ; ++i) { m_run[i] = -INFINITY; l_run[i] = 0.f; }

    // O-phase thread mapping: OQ rows x OD dims (d strided by DG)
    const int od = tid % C::DG, oq = tid / C::DG;
    float o_acc[OQ][OD];
#pragma unroll
    for (int i = 0; i < OQ; ++i)
#pragma unroll
        for (int j = 0; j < OD; ++j) o_acc[i][j] = 0.f;

    for (int k0 = 0; k0 < Sk; k0 += BK) {
        // ---- stage K tile (row-major) and V tile (transposed) --------------
        for (int i = tid; i < BK * CH; i += NT) {
            int r = i / CH, c = (i % CH) * 8;
            int4 kv = make_int4(0, 0, 0, 0), vv = make_int4(0, 0, 0, 0);
            if (k0 + r < Sk) {
                kv = *reinterpret_cast<const int4*>(Kg + (int64_t)(k0 + r) * kss + c);
                vv = *reinterpret_cast<const int4*>(Vg + (int64_t)(k0 + r) * vss + c);
            }
            *reinterpret_cast<int4*>(Ks + r * C::LDQ + c) = kv;
            const half* hv = reinterpret_cast<const half*>(&vv);
#pragma unroll
            for (int j = 0; j < 8; ++j) Vts[(c + j) * C::LDP + r] = hv[j];
        }
        __syncthreads();

        // ---- S = Q K^T (dot-pair HFMA2) ------------------------------------
        half2 sacc[C::TQ][TK];
#pragma unroll
        for (int i = 0; i < C::TQ; ++i)
#pragma unroll
            for (int j = 0; j < TK; ++j) sacc[i][j] = __float2half2_rn(0.f);
#pragma unroll 2
        for (int c = 0; c < D; c += 8) {
            int4 qf[C::TQ], kf[TK];
#pragma unroll
            for (int i = 0; i < C::TQ; ++i)
                qf[i] = *reinterpret_cast<const int4*>(Qs + (sq * C::TQ + i) * C::LDQ + c);
#pragma unroll
            for (int j = 0; j < TK; ++j)
                kf[j] = *reinterpret_cast<const int4*>(Ks + (sk + 8 * j) * C::LDQ + c);
#pragma unroll
            for (int i = 0; i < C::TQ; ++i)
#pragma unroll
                for (int j = 0; j < TK; ++j) hfma2_dot8(sacc[i][j], qf[i], kf[j]);
        }

        // ---- online softmax (fp32, exp2 domain) ----------------------------
#pragma unroll
        for (int i = 0; i < C::TQ; ++i) {
            float s[TK];
            float mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < TK; ++j) {
                float2 f = __half22float2(sacc[i][j]);
                s[j] = (k0 + sk + 8 * j < Sk) ? f.x + f.y : -INFINITY;
                mx = fmaxf(mx, s[j]);
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 4));
            const float m_new = fmaxf(m_run[i], mx);
            const float alpha = fast_exp2(m_run[i] - m_new);   // exp2(-inf) = 0 on the first tile
            float rs = 0.f;
            const int row = sq * C::TQ + i;
#pragma unroll
            for (int j = 0; j < TK; ++j) {
                float p = fast_exp2(s[j] - m_new);
                rs += p;
                Ps[row * C::LDP + sk + 8 * j] = __float2half_rn(p);
            }
            rs += __shfl_xor_sync(0xffffffff, rs, 1);
            rs += __shfl_xor_sync(0xffffffff, rs, 2);
            rs += __shfl_xor_sync(0xffffffff, rs, 4);
            l_run[i] = l_run[i] * alpha + rs;
            m_run[i] = m_new;
            if (sk == 0) alpha_s[row] = alpha;
        }
        __syncthreads();

        // ---- O = O * alpha + P V (dot-pair HFMA2 along k) ------------------
        half2 pacc[OQ][OD];
#pragma unroll
        for (int i = 0; i < OQ; ++i)
#pragma unroll
            for (int j = 0; j < OD; ++j) pacc[i][j] = __float2half2_rn(0.f);
#pragma unroll 2
        for (int c = 0; c < BK; c += 8) {
            int4 pf[OQ], vf[OD];
#pragma unroll
            for (int i = 0; i < OQ; ++i)
                pf[i] = *reinterpret_cast<const int4*>(Ps + (oq * OQ + i) * C::LDP + c);
#pragma unroll
            for (int j = 0; j < OD; ++j)
                vf[j] = *reinterpret_cast<const int4*>(Vts + (od + C::DG * j) * C::LDP + c);
#pragma unroll
            for (int i = 0; i < OQ; ++i)
#pragma unroll
                for (int j = 0; j < OD; ++j) hfma2_dot8(pacc[i][j], pf[i], vf[j]);
        }
#pragma unroll
        for (int i = 0; i < OQ; ++i) {
            const float a = alpha_s[oq * OQ + i];
#pragma unroll
            for (int j = 0; j < OD; ++j) {
                float2 f = __half22float2(pacc[i][j]);
                o_acc[i][j] = o_acc[i][j] * a + (f.x + f.y);
            }
        }
        __syncthreads();   // Ks/Vts/Ps/alpha_s are overwritten next tile
    }

    // ---- normalize and store ---------------------------------------------
    if (sk == 0) {
#pragma unroll
        for (int i = 0; i < C::TQ; ++i) l_s[sq * C::TQ + i] = l_run[i];
    }
    __syncthreads();
    half* Og = O + b * osb + h * osh;
#pragma unroll
    for (int i = 0; i < OQ; ++i) {
        const int row = oq * OQ + i;
        if (q0 + row >= Sq) continue;
        const float inv = 1.f / l_s[row];
#pragma unroll
        for (int j = 0; j < OD; ++j)
            Og[(int64_t)(q0 + row) * oss + od + C::DG * j] = __float2half_rn(o_acc[i][j] * inv);
    }
}

template <int D, int BQ, int OQ, int OD>
bool launch_cfg(const half* q, const half* k, const half* v, half* o,
                int B, int H, int Sq, int Sk, const int64_t* qs, const int64_t* ks,
                const int64_t* vs, const int64_t* os, float qscale, cudaStream_t stream) {
    using C = Cfg<D, BQ, OQ, OD>;
    auto kern = flash_attn_hfma2_kernel<D, BQ, OQ, OD>;
    static bool attr_set = false;
    if (!attr_set) {
        if (C::SMEM_BYTES > 48 * 1024 &&
            cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM_BYTES) != cudaSuccess)
            return false;
        attr_set = true;
    }
    dim3 grid((Sq + BQ - 1) / BQ, B * H);
    kern<<<grid, NT, C::SMEM_BYTES, stream>>>(q, k, v, o, H, Sq, Sk,
        qs[0], qs[1], qs[2], ks[0], ks[1], ks[2], vs[0], vs[1], vs[2], os[0], os[1], os[2], qscale);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace

// Returns false (nothing launched) for an unsupported head_dim; the caller
// falls back. Strides are in elements: {batch, head, seq}; last dim must be
// contiguous and every row 16-byte aligned (checked by the caller).
bool launch_flash_attention_hfma2(const void* qv, const void* kv, const void* vv, void* ov,
                                  int B, int H, int Sq, int Sk, int D,
                                  const int64_t* qs, const int64_t* ks, const int64_t* vs,
                                  const int64_t* os, float scale, cudaStream_t stream) {
    const half* q = static_cast<const half*>(qv);
    const half* k = static_cast<const half*>(kv);
    const half* v = static_cast<const half*>(vv);
    half* o = static_cast<half*>(ov);
    const float qscale = scale * 1.4426950408889634f;   // log2(e): softmax in exp2 domain
    switch (D) {
        case 40:  return launch_cfg<40, 64, 4, 5>(q, k, v, o, B, H, Sq, Sk, qs, ks, vs, os, qscale, stream);
        case 64:  return launch_cfg<64, 64, 4, 8>(q, k, v, o, B, H, Sq, Sk, qs, ks, vs, os, qscale, stream);
        case 80:  return launch_cfg<80, 64, 4, 10>(q, k, v, o, B, H, Sq, Sk, qs, ks, vs, os, qscale, stream);
        case 128: return launch_cfg<128, 32, 4, 8>(q, k, v, o, B, H, Sq, Sk, qs, ks, vs, os, qscale, stream);
        case 160: return launch_cfg<160, 32, 4, 10>(q, k, v, o, B, H, Sq, Sk, qs, ks, vs, os, qscale, stream);
        default:  return false;
    }
}
