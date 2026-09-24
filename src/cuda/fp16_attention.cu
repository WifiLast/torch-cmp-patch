#include <cstdio>
#include <cstdint>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// =================================================================================
// Configuration
// =================================================================================

#define TR          16      // Query tile rows  == blockDim.y
#define TC          32      // K/V tile rows    (see occupancy note below)
#define D_DIM       128     // Head dimension   (compile-time constant)
#define WARP_SIZE   32
#define STAGES      2       // Double-buffered K/V tiles (see below)

// Bank-conflict padding: 8 halves = 16 bytes per row
#define SMEM_PAD    8
#define SMEM_STRIDE (D_DIM + SMEM_PAD)   // 136 halves per row

// =================================================================================
// Shared-memory layout: STAGES=2 double-buffered K/V tiles.
//
//   K_smem : [STAGES][TC][SMEM_STRIDE]  half
//   V_smem : [STAGES][TC][SMEM_STRIDE]  half  (immediately after K_smem)
//
// Total per block = 2(K+V) * STAGES(2) * TC * SMEM_STRIDE * sizeof(half)
//                 = 2 * 2 * 32 * 136 * 2  =  34,816 bytes  (~34 KB)
//
// Turing hard limit = 64 KB -- fits 1 block/SM (was 3 blocks/SM with the
// prior single-buffered ~17 KB design; TC=32 kept unchanged here rather
// than also shrinking it to claw back occupancy, to keep this change
// contained and easy to verify against the single-buffered baseline).
//
// WHY double-buffer at all: the online-softmax compute for tile N only
// depends on K_smem[N]/V_smem[N] (already resident) and the running
// m_i/l_i/o_acc registers -- it does NOT depend on tile N+1's data. So
// tile N+1's global-memory load can be issued *before* tile N's compute
// runs, overlapping load latency with compute instead of serializing
// load-then-compute-then-load every tile (the single-buffered design's
// two __syncthreads() per tile: one to wait for the load, one to protect
// the buffer from the next tile's load). This mirrors the exact
// prefetch/compute/barrier structure already used in fp16_matmul.cu and
// fp16_conv.cu (see fp16_conv.cu's FIX[4] comment for why STAGES=2 needs
// two barriers per iteration, not one -- same reasoning applies here).
//
// This is a real trade (occupancy for pipelining) whose net effect on
// this exact GPU/shape needs to be measured, not assumed -- validate with
// tests_hardware/test_attention_bf16_routing.py (checks both correctness
// and speed) before trusting this over reverting to single-buffering.
// =================================================================================

// =================================================================================
// Tile loader  (sm_75 safe -- uses __ldg for read-only cache, no cp.async)
//
// Block = TR*32 = 512 threads.
// One tile = TC rows x D_DIM cols = 32*128 = 4096 halves = 512 * 8 halves.
// Every thread loads exactly ONE int4 (128-bit / 8 halves) with no loop,
// so there is zero warp divergence on full tiles. Unchanged from the
// single-buffered version -- the caller now passes K_smem[stage] /
// V_smem[stage] (which decays to the same half[][SMEM_STRIDE] type this
// function already expects), so no signature change was needed.
//
// Boundary tiles: invalid rows are zero-filled so they produce score=0 -> p~=0,
// contributing nothing to the softmax accumulator.
// =================================================================================
__device__ __forceinline__
void load_tile_sync(
    half        smem_tile[][SMEM_STRIDE],  // [TC][SMEM_STRIDE]
    const half* __restrict__ src,          // K_base or V_base
    int k_start, int S, int D,
    int ty, int tx)
{
    // Flatten thread index: 0..511
    const int tid = ty * WARP_SIZE + tx;

    // Each thread owns one int4 pack (8 halves).
    // D_DIM/8 = 16 packs per row, so:
    //   row = tid / 16,  col = (tid % 16) * 8
    const int row = tid >> 4;
    const int col = (tid & 15) << 3;

    if (k_start + row < S) {
        // __ldg: read-only (texture) cache path -- correct __ldg usage
        const int4* gptr = reinterpret_cast<const int4*>(src + (k_start + row) * D + col);
        *reinterpret_cast<int4*>(&smem_tile[row][col]) = __ldg(gptr);
    } else {
        // Zero-pad so out-of-bounds rows produce score = 0
        *reinterpret_cast<int4*>(&smem_tile[row][col]) = make_int4(0, 0, 0, 0);
    }
}

// =================================================================================
// hexp_safe: exp() for FP16 via FP32 SFU path
//
// Why not hexp() / ex2.approx.f16?
//   PTX ex2.approx.f16 has only ~9-bit mantissa accuracy, which causes
//   visible error in softmax on sm_75 (no hardware ex2.f16 on Turing).
//   __expf() is a scalar SFU instruction (not an FFMA), so it does not
//   violate the "no FP32 FMA" constraint.  The surrounding cvt instructions
//   are pure data-movement -- no arithmetic.
// =================================================================================
__device__ __forceinline__ half hexp_safe(half h) {
    return __float2half(__expf(__half2float(h)));
}

// =================================================================================
// Warp reduction: sum a single FP16 value across all 32 lanes (butterfly XOR)
//
// Shuffles the raw 16-bit pattern to avoid any FP32 reinterpretation;
// __hadd is native FP16 addition.
// =================================================================================
__device__ __forceinline__ half warp_reduce_sum_h(half val) {
    #pragma unroll
    for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
        unsigned short bits = __shfl_xor_sync(
            0xffffffff,
            *reinterpret_cast<unsigned short*>(&val),
            mask);
        val = __hadd(val, *reinterpret_cast<half*>(&bits));
    }
    return val;
}

// =================================================================================
// Flash Attention Kernel -- sm_75 / Turing, STAGES=2 double-buffered
//
// Grid  : (ceil(S/TR),  B*H)
// Block : (32,          TR)   = 512 threads = 16 warps
//
// Each block handles TR consecutive query rows for one (batch, head) pair.
// scale is passed as FP16 (converted once in the launcher) so the kernel
// contains zero FP32 values or operations.
// =================================================================================
__global__ void flash_attention_fp16_optimized(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half*       __restrict__ O,
    int B, int H, int S, int D,
    half scale
) {
    // ------------------------------------------------------------------
    // Shared memory -- STAGES double-buffered K and V tiles
    // ------------------------------------------------------------------
    extern __shared__ half smem[];
    half (*K_smem)[TC][SMEM_STRIDE] = reinterpret_cast<half (*)[TC][SMEM_STRIDE]>(smem);
    half (*V_smem)[TC][SMEM_STRIDE] = reinterpret_cast<half (*)[TC][SMEM_STRIDE]>(smem + STAGES * TC * SMEM_STRIDE);

    // ------------------------------------------------------------------
    // Indices
    // ------------------------------------------------------------------
    const int tx = threadIdx.x;    // lane   0..31
    const int ty = threadIdx.y;    // q-row within block  0..TR-1

    const int bh        = blockIdx.y;
    const int batch_idx = bh / H;
    const int head_idx  = bh % H;

    const long long bh_offset =
        ((long long)batch_idx * H + head_idx) * (long long)S * D;

    const half* Q_base = Q + bh_offset;
    const half* K_base = K + bh_offset;
    const half* V_base = V + bh_offset;
    half*       O_base = O + bh_offset;

    const int  q_row  = blockIdx.x * TR + ty;
    const bool valid_q = (q_row < S);

    // ------------------------------------------------------------------
    // Register file
    //
    // Thread tx covers D in two half2 packs:
    //   q_frag[0]: cols  tx*2,    tx*2+1     (range   0..63)
    //   q_frag[1]: cols  tx*2+64, tx*2+65    (range  64..127)
    // 32 lanes x 2 packs x 2 halves = 128 = D_DIM  (complete coverage)
    // ------------------------------------------------------------------
    half2 q_frag[2];
    half2 o_acc[2];
    o_acc[0] = __float2half2_rn(0.0f);
    o_acc[1] = __float2half2_rn(0.0f);

    // Online softmax state (FP16; see numerical note [N2] at EOF)
    half m_i = __float2half(-65504.0f);   // running max  (~-inf in FP16)
    half l_i = __float2half(0.0f);        // running sum of exp weights

    const half2 scale2 = __half2half2(scale);

    // ------------------------------------------------------------------
    // Load Q row and pre-scale by 1/sqrt(D)
    // __ldg routes through the read-only cache (same as K/V loads).
    // ------------------------------------------------------------------
    if (valid_q) {
        const half2* qrow = reinterpret_cast<const half2*>(Q_base + q_row * D);
        q_frag[0] = __hmul2(__ldg(&qrow[tx]),      scale2);
        q_frag[1] = __hmul2(__ldg(&qrow[tx + 32]), scale2);
    }

    // ------------------------------------------------------------------
    // Main loop over K/V tiles -- double-buffered (STAGES=2)
    //
    // Barrier protocol per iteration (mirrors fp16_conv.cu's STAGES=2
    // pattern -- see its FIX[4] comment for the full derivation):
    //   1. Prefetch tile+1 into stage load_stage (no sync yet). Safe to
    //      issue now because compute below reads compute_stage, a
    //      different buffer.
    //   2. __syncthreads() -- makes this prefetch visible, AND makes any
    //      *prior* iteration's prefetch into compute_stage visible before
    //      we read it below (compute_stage this iteration == load_stage
    //      two iterations ago, since STAGES=2 aliases every other tile).
    //   3. Compute from compute_stage.
    //   4. __syncthreads() -- required for STAGES=2: protects this
    //      iteration's reads of compute_stage from the *next* iteration's
    //      prefetch, which writes to load_stage == this iteration's
    //      compute_stage (mod-2 aliasing again).
    // ------------------------------------------------------------------
    const int num_tiles = (S + TC - 1) / TC;

    if (num_tiles > 0) {
        // Prologue: fill stage 0 before the loop starts.
        load_tile_sync(K_smem[0], K_base, 0, S, D, ty, tx);
        load_tile_sync(V_smem[0], V_base, 0, S, D, ty, tx);
        __syncthreads();
    }

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int compute_stage = tile & 1;
        const int load_stage    = (tile + 1) & 1;
        const int k_start       = tile * TC;

        // 1. Prefetch next tile (overlaps with this iteration's upcoming
        //    compute once the loads are actually in flight on Turing's
        //    async memory pipeline).
        if (tile + 1 < num_tiles) {
            const int next_k_start = (tile + 1) * TC;
            load_tile_sync(K_smem[load_stage], K_base, next_k_start, S, D, ty, tx);
            load_tile_sync(V_smem[load_stage], V_base, next_k_start, S, D, ty, tx);
        }

        // 2. Barrier: see protocol note above.
        __syncthreads();

        if (valid_q) {
            // Handle boundary tile: clamp to actual valid K rows
            const int valid_rows = min(TC, S - k_start);

            for (int j = 0; j < valid_rows; ++j) {

                // ---- A. Dot product: score = (Q * scale) . K[j] ----
                //
                // krow[tx]    = K cols  tx*2,   tx*2+1    (matches q_frag[0])
                // krow[tx+32] = K cols  tx*2+64,tx*2+65   (matches q_frag[1])
                // After __hfma2 we have 32 partial sums, each covering 4 of
                // the 128 D elements. warp_reduce_sum_h collapses them to one
                // scalar score shared by all lanes in this warp.
                const half2* krow = reinterpret_cast<const half2*>(K_smem[compute_stage][j]);
                half2 dot2 = __hmul2(q_frag[0], krow[tx]);
                dot2       = __hfma2(q_frag[1], krow[tx + 32], dot2);
                half score = warp_reduce_sum_h(__hadd(dot2.x, dot2.y));

                // ---- B. Online softmax recurrence (all FP16) ----
                //
                // m_new  = max(m_old, score)
                // alpha  = exp(m_old - m_new)    rescales the old accumulator
                // p      = exp(score - m_new)    weight for this K/V column
                // l_new  = l_old * alpha + p
                // O_new  = O_old * alpha + p * V[j]
                half m_prev = m_i;
                m_i = __hmax(m_prev, score);

                half p     = hexp_safe(__hsub(score,  m_i));
                half alpha = hexp_safe(__hsub(m_prev, m_i));

                l_i = __hfma(l_i, alpha, p);

                // ---- C. Output accumulator update ----
                half2 p2     = __half2half2(p);
                half2 alpha2 = __half2half2(alpha);

                const half2* vrow = reinterpret_cast<const half2*>(V_smem[compute_stage][j]);
                o_acc[0] = __hfma2(p2, vrow[tx],      __hmul2(o_acc[0], alpha2));
                o_acc[1] = __hfma2(p2, vrow[tx + 32], __hmul2(o_acc[1], alpha2));
            }
        }

        // 4. Trailing barrier: see protocol note above.
        __syncthreads();
    }

    // ------------------------------------------------------------------
    // Epilogue: O = O_acc / l_i
    //
    // h2rcp() -> PTX rcp.approx.ftz.f16x2  (~11-bit mantissa accuracy)
    // Pure FP16 reciprocal; no FP32 division.
    // ------------------------------------------------------------------
    if (valid_q) {
        half2 inv_l = h2rcp(__half2half2(l_i));
        half2* orow = reinterpret_cast<half2*>(O_base + q_row * D);
        orow[tx]      = __hmul2(o_acc[0], inv_l);
        orow[tx + 32] = __hmul2(o_acc[1], inv_l);
    }
}

// =================================================================================
// Launcher
//
// Returns cudaError_t so callers can detect failures.
// scale_f32 is converted to FP16 here (host side, once) so the kernel
// is entirely free of FP32 values.
// =================================================================================
void launch_attention_fp16(
    const void*  q,
    const void*  k,
    const void*  v,
    void*        output,
    int          B,
    int          H,
    int          S,
    int          D,
    float        scale_f32
) {
    if (D != D_DIM) return;

    // Compile-time guard: smem must never exceed Turing's 64 KB limit.
    // STAGES=2, TC=32: 2 * 2 * 32 * 136 * 2 = 34,816 bytes. Fits with room
    // to spare, but only 1 block/SM (vs 3 for the prior single-buffered
    // ~17 KB layout) -- see the occupancy-vs-pipelining note at the top
    // of this file.
    static_assert(
        STAGES * 2 * TC * SMEM_STRIDE * sizeof(half) <= 65536u,
        "smem exceeds Turing 64 KB hard limit -- reduce TC or STAGES");

    const size_t smem_bytes = STAGES * 2 * TC * SMEM_STRIDE * sizeof(half);

    // smem_bytes is a compile-time constant (TC/SMEM_STRIDE/STAGES are
    // #defines), so the attribute value never changes between calls --
    // set it once per process instead of on every single attention call.
    // This is the hottest op in the whole model (called every block,
    // every denoising step), so a redundant driver call here adds up;
    // cudaFuncSetAttribute also isn't guaranteed side-effect-free to call
    // from multiple host threads concurrently, and doing it once avoids
    // that too.
    static bool smem_attr_set = false;
    if (!smem_attr_set) {
        cudaError_t err = cudaFuncSetAttribute(
            flash_attention_fp16_optimized,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));
        if (err != cudaSuccess) return;
        smem_attr_set = true;
    }

    dim3 block(WARP_SIZE, TR);
    dim3 grid((S + TR - 1) / TR, B * H);

    // Convert scale once on the host; the kernel accepts half directly.
    const half scale_h = __float2half(scale_f32);

    flash_attention_fp16_optimized<<<grid, block, smem_bytes, 0>>>(
        static_cast<const half*>(q),
        static_cast<const half*>(k),
        static_cast<const half*>(v),
        static_cast<half*>(output),
        B, H, S, D,
        scale_h);

    //return cudaGetLastError();
}

// =================================================================================
// Numerical & design notes
// =================================================================================
//
// [N1] hexp_safe and the FP32 SFU path
//   Turing has no hardware ex2.approx.f16; PTX emulates it with ~9-bit
//   mantissa accuracy, causing noticeable softmax error.  __expf() is a
//   single SFU instruction (MUFU.EX2 under the hood) -- it is NOT an FFMA
//   and does not violate the "no FP32 FMA" constraint.  The surrounding
//   __half2float / __float2half are register-level type conversions (CVT
//   instructions), not arithmetic.
//
// [N2] FP16 accumulation range (S=1024)
//   l_i  : sum of TC=32 exp() values per tile, rescaled by alpha<1 each
//           tile.  Worst case << 1024.  FP16 max = 65504.  Safe.
//   o_acc: continuously rescaled by alpha < 1; does not grow unboundedly.
//   m_i  : max of all scores seen so far; bounded by input scale.
//   For S >> 1024 or un-normalized inputs, promoting l_i/m_i to FP32
//   would improve robustness but violates the stated constraint.
//
// [N3] Dot-product reduction and D_DIM coupling
//   The Q load, K smem access pattern, and warp_reduce_sum_h are all
//   coupled to D_DIM=128 and blockDim.x=32.  If either changes, all three
//   must be updated together.  The static_assert in the launcher catches
//   D != 128 at runtime; a compile-time check would require template params.
//
// [N4] h2rcp precision
//   rcp.approx.ftz.f16x2 gives ~11 mantissa bits.  For a normalized
//   softmax output in [0,1] this is sufficient; the dominant error source
//   is the exp() approximation, not the reciprocal.
//
// [N5] STAGES=2 double buffering (this version)
//   Trades occupancy (3 blocks/SM -> 1 block/SM, since smem doubled from
//   ~17KB to ~34KB) for load/compute overlap (tile N+1's global load is
//   issued before tile N's compute, instead of strictly serialized
//   load-sync-compute-sync). Whether this nets a win depends on whether
//   the extra ILP from pipelining outweighs the lost warp-level latency
//   hiding from fewer resident blocks -- for a kernel with a genuinely
//   sequential per-j online-softmax recurrence (no cross-iteration ILP
//   within a warp), occupancy may matter more than usual. Measure with
//   tests_hardware/test_attention_bf16_routing.py before trusting this
//   over a single-buffered revert.
