// =================================================================================
// conv3d_fp16_turing.cu
//
// Extends fp16_conv.cu's conv2d_fp16_turing (shared-memory-tiled, GEMM-style,
// double-buffered) to a depth axis, replacing the naive one-thread-per-voxel
// kernel this file used to contain (see README.md / cmpext3/__init__.py's
// enable() comment for the ~460ms-vs-cuDNN's-83ms numbers on real VAE-decode
// shapes that motivated this rewrite).
//
// CHANGES FROM THE 2D VERSION:
//  [1] GEMM dimensions gain a depth term:
//        M (pixels -> "voxels") = B * D_out * H_out * W_out
//        K (flattened kernel window) = C_in * K_D * K_H * K_W
//        N (output channels) = C_out, unchanged.
//
//  [2] Voxel decomposition gains a D_out level. Where the 2D kernel unpacks
//      a flat pixel index into (b, h_out, w_out) via one division by
//      stride_hw = H_out*W_out, the 3D kernel unpacks (b, d_out, h_out,
//      w_out) via two divisions: first by voxels_per_batch = D_out*H_out*W_out,
//      then the remainder by stride_hw. This only matters for LOAD_A (needs
//      d_out/h_out/w_out separately, each combined with its own kernel
//      offset and bounds check) -- the store path still only needs
//      (b_idx, rem) exactly like 2D, since rem = m_idx % voxels_per_batch is
//      already the correct flat (d_out,h_out,w_out) offset within a channel.
//
//  [3] LOAD_A's flat-K decomposition gains one nesting level: (kv, ku, kd,
//      kc) instead of (kv, ku, kc), with d_in = d_out*s_d - p_d + kd*d_d
//      added to the existing 3-way (now) bounds check alongside h_in/w_in.
//
//  [4] LOAD_B is structurally unchanged -- weight is still contiguous
//      [C_out, K] once flattened, only K itself is larger (C_in*K_D*K_H*K_W
//      instead of C_in*K_H*K_W).
//
//  [5] max_k_tiles grows proportionally with K (e.g. 432 vs. conv2d's ~144
//      for the bench.py VAE-decode shape) -- more main-loop iterations, not
//      a structural change.
//
// TILE PARAMETERS ARE NOW INDEPENDENTLY TUNABLE (this revision):
//  conv2d_fp16_turing's LOAD_A/LOAD_B (and the first version of this file)
//  did a *single-shot* tile load: the thread-to-data mapping (load_a_row =
//  tid/4, load_b_col = (tid%16)*8, load_b_active = tid<512, etc.) was
//  algebra hardcoded for exactly BM=256/BN=128/BK=32/BLOCK_SIZE=1024 -- every
//  thread loaded exactly one int4 and the numbers happened to divide evenly.
//  Changing BM/BN/BK without also redoing that algebra would silently load
//  the wrong data. LOAD_A/LOAD_B below are now grid-stride loops over the
//  tile's int4 count (`for (i = tid; i < TILE_INT4; i += BLOCK_SIZE)`), so
//  they're correct for any BM/BN/BK/BLOCK_SIZE combination (BM, BN multiples
//  of 32; BK, BN multiples of 8 for the int4/8-half vectorized loads) at the
//  cost of recomputing each visited row's voxel decomposition inside the
//  loop instead of once per thread up front -- only matters when a thread
//  must make more than one pass (BM*(BK/8) or BK*(BN/8) > BLOCK_SIZE).
//
//  BLOCK_SIZE remains DERIVED, not free: every warp covers a fixed 32(M) x
//  32(N) output sub-tile (8 lanes_m * TM=4, 4 lanes_n * TN=8 -- see the
//  lane_m/lane_n split below), so the block must contain exactly
//  (BM/32) x (BN/32) warps to tile the whole block output, giving
//  BLOCK_SIZE = (BM/32)*(BN/32)*32. TM/TN stay fixed at 4/8 (the half2
//  accumulator packing and the store loop are written specifically for that
//  shape); decoupling them from BM/BN/threads-per-block would need a second,
//  separate lane-tiling redesign, not just a #define change.
//
// Defaults below (BM=256/BN=128/BK=32/STAGES=2) are unchanged from the
// previous revision -- this is a structural generalization, not a retune.
// =================================================================================

#include <cstdio>
#include <cstdint>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// =================================================================================
// Part 1: Configuration -- BM/BN/BK/STAGES are independently tunable (see
// header comment); BLOCK_SIZE is derived from BM/BN, not a free parameter.
//
// Overridable from the command line via -DCONV3D_BM=.. -DCONV3D_BN=..
// -DCONV3D_BK=.. -DCONV3D_STAGES=.. (see ../../setup.py's autotune_conv3d_tile()).
// The CONV3D_ prefix is deliberate: fp16_conv.cu (the 2D kernel this file
// was extended from) defines its own, INDEPENDENT BM/BN/BK/STAGES with the
// same bare names. Extension-wide compile flags apply to every .cu file in
// setup.py's sources list, so an unprefixed -DBM=.. would silently retune
// conv2d too; fp16_conv.cu never references CONV3D_BM etc., so it can't.
// =================================================================================

#ifndef CONV3D_BM
#define CONV3D_BM 256   // Block M (Voxels)      -- must be a multiple of 32
#endif
#ifndef CONV3D_BN
#define CONV3D_BN 128   // Block N (Channels)    -- must be a multiple of 32
#endif
#ifndef CONV3D_BK
#define CONV3D_BK 32    // Block K (Accumulation axis) -- must be a multiple of 8
#endif
#ifndef CONV3D_STAGES
#define CONV3D_STAGES 2 // Double/triple-buffer depth, >= 2 (main loop's sync
                        // pattern is over-synchronized but still correct for
                        // STAGES > 2 -- see the main-loop comment below)
#endif

#define BM CONV3D_BM
#define BN CONV3D_BN
#define BK CONV3D_BK
#define STAGES CONV3D_STAGES

// Register tile per thread: 4 rows x 8 cols -- FIXED (see header comment)
#define TM 4
#define TN 8

// Shared memory padding to reduce bank conflicts
#define PAD 8

// Derived: one warp per 32x32 output sub-tile, (BM/32)x(BN/32) warps/block.
#define BLOCK_SIZE ((BM / 32) * (BN / 32) * 32)

// Total int4 (8-half) loads needed to fill the BMxBK / BKxBN smem tiles.
#define A_TILE_INT4 (BM * (BK / 8))
#define B_TILE_INT4 (BK * (BN / 8))

// =================================================================================
// Part 2: Kernel
// =================================================================================

__global__ void __launch_bounds__(BLOCK_SIZE) conv3d_fp16_turing(
    const half* __restrict__ input,     // [B, C_in, D_in, H_in, W_in]
    const half* __restrict__ weight,    // [C_out, C_in*KD*KH*KW]  (GEMM-layout)
    const half* __restrict__ bias,
    half* __restrict__ output,          // [B, C_out, D_out, H_out, W_out]
    int B, int C_in, int D_in, int H_in, int W_in,
    int C_out, int K_D, int K_H, int K_W,
    int D_out, int H_out, int W_out,
    int s_d, int s_h, int s_w, int p_d, int p_h, int p_w, int d_d, int d_h, int d_w)
{
    extern __shared__ char smem_raw[];

    // As: [STAGES][BM][BK + PAD]
    half (*As)[BM][BK + PAD] = reinterpret_cast<half (*)[BM][BK + PAD]>(smem_raw);

    // Bs: [STAGES][BK][BN + PAD]
    half (*Bs)[BK][BN + PAD] = reinterpret_cast<half (*)[BK][BN + PAD]>(
        smem_raw + STAGES * BM * (BK + PAD) * sizeof(half));

    // Accumulators: [TM][TN/2] half2  (each half2 holds 2 output columns)
    half2 accum[TM][TN / 2];
    half2 frag_a[TM / 2];   // TM=4 -> 2 half2s
    half2 frag_b[TN / 2];   // TN=8 -> 4 half2s

    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN / 2; j++)
            accum[i][j] = __float2half2_rn(0.0f);

    int tid = threadIdx.x;
    int bx  = blockIdx.x;   // Output channel block
    int by  = blockIdx.y;   // Voxel block

    // ------------------------------------------------------------------
    // Thread -> tile mapping. lane_m/lane_n (the 8x4 split of one warp's 32
    // lanes) is fixed by TM=4/TN=8 and independent of BM/BN. warp_m/warp_n
    // (the block's warp grid) depends on how many warps span N, which does
    // depend on BN -- warp_n_count generalizes the old hardcoded "4".
    // ------------------------------------------------------------------
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int warp_n_count = BN / 32;
    int warp_m = warp_id / warp_n_count;   // 0 .. (BM/32 - 1)
    int warp_n = warp_id % warp_n_count;   // 0 .. (BN/32 - 1)

    int lane_m = lane_id / 4;   // 0..7
    int lane_n = lane_id % 4;   // 0..3

    int thread_m_start = warp_m * 32 + lane_m * TM;   // max BM-4 < BM
    int thread_n_start = warp_n * 32 + lane_n * TN;   // max BN-8 < BN

    int block_pixel_start   = by * BM;
    int block_channel_start = bx * BN;

    int voxels_per_batch = D_out * H_out * W_out;
    int total_voxels     = B * voxels_per_batch;
    int total_k           = C_in * K_D * K_H * K_W;
    int stride_hw          = H_out * W_out;
    int stride_bvox        = C_out * voxels_per_batch;

    int max_k_tiles = (total_k + BK - 1) / BK;

    // ==================================================================
    // Helper macros: grid-stride load of one A/B tile into smem stage s at
    // GEMM-K offset k_base. Correct for any BM/BN/BK/BLOCK_SIZE (a thread
    // may visit 0, 1, or several int4 slots depending on how the tile size
    // compares to BLOCK_SIZE) -- see header comment for why this replaced
    // the earlier single-shot version. Flat K index still decomposes to
    // (c_in, kd, kh, kw), same as before.
    // ==================================================================
#define LOAD_A(s, k_base)                                                   \
    do {                                                                     \
        for (int _i = tid; _i < A_TILE_INT4; _i += BLOCK_SIZE) {            \
            int a_row = _i / (BK / 8);                                      \
            int a_col = (_i % (BK / 8)) * 8;                                \
            int4 lv = make_int4(0,0,0,0);                                   \
            half* vd = reinterpret_cast<half*>(&lv);                        \
            int pixel_idx = block_pixel_start + a_row;                      \
            bool valid_pixel = (pixel_idx < total_voxels);                  \
            int kb = (k_base) + a_col;                                      \
            if (valid_pixel && kb < total_k) {                              \
                int rem      = pixel_idx % voxels_per_batch;                \
                int p_b_idx  = pixel_idx / voxels_per_batch;                \
                int p_d_out  = rem / stride_hw;                             \
                int rem2     = rem % stride_hw;                             \
                int p_h_out  = rem2 / W_out;                                \
                int p_w_out  = rem2 % W_out;                                \
                _Pragma("unroll")                                            \
                for (int v = 0; v < 8; ++v) {                               \
                    int ck = kb + v;                                         \
                    if (ck < total_k) {                                      \
                        int tmp = ck;                                        \
                        int kv  = tmp % K_W; tmp /= K_W;                    \
                        int ku  = tmp % K_H; tmp /= K_H;                    \
                        int kd  = tmp % K_D;                                 \
                        int kc  = tmp / K_D;                                 \
                        int d_in = p_d_out * s_d - p_d + kd * d_d;         \
                        int h_in = p_h_out * s_h - p_h + ku * d_h;         \
                        int w_in = p_w_out * s_w - p_w + kv * d_w;         \
                        if (d_in >= 0 && d_in < D_in && h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) \
                            vd[v] = input[(((p_b_idx * C_in + kc) * D_in + d_in) * H_in + h_in) * W_in + w_in]; \
                    }                                                        \
                }                                                            \
            }                                                                \
            *reinterpret_cast<int4*>(&As[(s)][a_row][a_col]) = lv;          \
        }                                                                    \
    } while(0)

#define LOAD_B(s, k_base)                                                   \
    do {                                                                     \
        for (int _i = tid; _i < B_TILE_INT4; _i += BLOCK_SIZE) {            \
            int b_row = _i / (BN / 8);                                      \
            int b_col = (_i % (BN / 8)) * 8;                                \
            int gn = block_channel_start + b_col;                          \
            int gk = (k_base) + b_row;                                     \
            int4 lv = make_int4(0,0,0,0);                                   \
            half* vd = reinterpret_cast<half*>(&lv);                        \
            if (gn < C_out && gk < total_k) {                               \
                _Pragma("unroll")                                            \
                for (int v = 0; v < 8; ++v) {                               \
                    int cn = gn + v;                                         \
                    if (cn < C_out)                                          \
                        vd[v] = weight[cn * total_k + gk];                  \
                }                                                            \
            }                                                                \
            *reinterpret_cast<int4*>(&Bs[(s)][b_row][b_col]) = lv;          \
        }                                                                    \
    } while(0)

    // ==================================================================
    // Prologue: fill stage 0
    // ==================================================================
    if (max_k_tiles > 0) {
        LOAD_A(0, 0);
        LOAD_B(0, 0);
        __syncthreads();
    }

    // ==================================================================
    // Main loop -- double-buffered by default. The leading sync (after
    // prefetch, before compute) and trailing sync (after compute, before
    // the next prefetch) are both required for STAGES=2 (see fp16_conv.cu's
    // FIX [4]); for STAGES>2 the same pattern is over-synchronized but
    // still correct -- the slot being (re)written was last read at least
    // STAGES-1 iterations ago, well before the immediately preceding sync.
    // ==================================================================
    for (int k_step = 0; k_step < max_k_tiles; k_step++) {
        int compute_idx = k_step       % STAGES;   // stage to READ
        int load_idx    = (k_step + 1) % STAGES;   // stage to WRITE next tile

        if (k_step + 1 < max_k_tiles) {
            int next_k = (k_step + 1) * BK;
            LOAD_A(load_idx, next_k);
            LOAD_B(load_idx, next_k);
        }

        __syncthreads();

        // Compute outer product over BK inner steps using HFMA2
        #pragma unroll
        for (int k_inner = 0; k_inner < BK; k_inner++) {

            half a0 = As[compute_idx][thread_m_start + 0][k_inner];
            half a1 = As[compute_idx][thread_m_start + 1][k_inner];
            half a2 = As[compute_idx][thread_m_start + 2][k_inner];
            half a3 = As[compute_idx][thread_m_start + 3][k_inner];
            frag_a[0] = __halves2half2(a0, a1);
            frag_a[1] = __halves2half2(a2, a3);

            int4 b_vec = *reinterpret_cast<int4*>(
                &Bs[compute_idx][k_inner][thread_n_start]);
            const half2* b_h2 = reinterpret_cast<const half2*>(&b_vec);
            frag_b[0] = b_h2[0];
            frag_b[1] = b_h2[1];
            frag_b[2] = b_h2[2];
            frag_b[3] = b_h2[3];

            half2 va0 = __half2half2(frag_a[0].x);
            half2 va1 = __half2half2(frag_a[0].y);
            half2 va2 = __half2half2(frag_a[1].x);
            half2 va3 = __half2half2(frag_a[1].y);

            #pragma unroll
            for (int j = 0; j < 4; j++) {
                accum[0][j] = __hfma2(va0, frag_b[j], accum[0][j]);
                accum[1][j] = __hfma2(va1, frag_b[j], accum[1][j]);
                accum[2][j] = __hfma2(va2, frag_b[j], accum[2][j]);
                accum[3][j] = __hfma2(va3, frag_b[j], accum[3][j]);
            }
        }

        __syncthreads();
    }

#undef LOAD_A
#undef LOAD_B

    // ==================================================================
    // Store accumulators to global memory. rem = m_idx % voxels_per_batch
    // is already the correct flat (d_out,h_out,w_out) offset within a
    // channel -- no need to decompose it further here, same as 2D only
    // needing (b_idx, rem) via stride_hw.
    // ==================================================================
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int m_idx = block_pixel_start + thread_m_start + i;
        if (m_idx < total_voxels) {
            int b_idx = m_idx / voxels_per_batch;
            int rem   = m_idx % voxels_per_batch;

            #pragma unroll
            for (int j = 0; j < TN / 2; j++) {
                int n_base = block_channel_start + thread_n_start + j * 2;

                half v0 = accum[i][j].x;
                half v1 = accum[i][j].y;

                if (n_base < C_out) {
                    if (bias) v0 = __hadd(v0, bias[n_base]);
                    output[b_idx * stride_bvox + n_base * voxels_per_batch + rem] = v0;
                }
                if (n_base + 1 < C_out) {
                    if (bias) v1 = __hadd(v1, bias[n_base + 1]);
                    output[b_idx * stride_bvox + (n_base + 1) * voxels_per_batch + rem] = v1;
                }
            }
        }
    }
}

// =================================================================================
// Part 3: Launcher
// =================================================================================

void launch_conv3d_fp16(
    const void* input, const void* weight, const void* bias, void* output,
    int B, int C_in, int D_in, int H_in, int W_in,
    int C_out, int K_D, int K_H, int K_W,
    int D_out, int H_out, int W_out,
    int s_d, int s_h, int s_w,
    int p_d, int p_h, int p_w,
    int d_d, int d_h, int d_w)
{
    const half* input_ptr  = reinterpret_cast<const half*>(input);
    const half* weight_ptr = reinterpret_cast<const half*>(weight);
    const half* bias_ptr   = reinterpret_cast<const half*>(bias);
    half*       output_ptr = reinterpret_cast<half*>(output);

    long long total = (long long)B * C_out * D_out * H_out * W_out;
    if (total <= 0) return;

    int m = B * D_out * H_out * W_out;
    int n = C_out;

    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BN - 1) / BN,
              (m + BM - 1) / BM);

    // Same smem footprint as fp16_conv.cu -- depends only on
    // STAGES/BM/BK/PAD/BN, unaffected by the extra K decomposition level.
    //   As: 2 * 256 * 40 * 2 = 40960 bytes
    //   Bs: 2 *  32 * 136 * 2 = 17408 bytes
    //   Total: 58368 bytes (~57 KB) -- within Turing's 64 KB hard limit
    int as_bytes    = STAGES * BM * (BK + PAD) * sizeof(half);
    int bs_bytes    = STAGES * BK * (BN + PAD) * sizeof(half);
    int total_smem  = as_bytes + bs_bytes;

    static bool smem_attrs_set = false;
    if (!smem_attrs_set) {
        cudaFuncSetAttribute(conv3d_fp16_turing,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared);

        cudaFuncSetAttribute(conv3d_fp16_turing,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            total_smem);
        smem_attrs_set = true;
    }

    conv3d_fp16_turing<<<grid, block, total_smem, 0>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        B, C_in, D_in, H_in, W_in,
        C_out, K_D, K_H, K_W, D_out, H_out, W_out,
        s_d, s_h, s_w, p_d, p_h, p_w, d_d, d_h, d_w);
}

// =================================================================================
// Part 4: Autotune benchmark harness (opt-in, NOT part of the real build)
//
// Only compiled when -DCMPEXT3_AUTOTUNE_HARNESS is passed -- see
// autotune_conv3d_tile() in ../../setup.py, which compiles this file to a
// standalone .exe once per (CONV3D_BM, CONV3D_BN, CONV3D_BK, CONV3D_STAGES)
// candidate and times it directly, rather than reimplementing the kernel
// call anywhere else. Exercises the exact same launch_conv3d_fp16 the real
// extension calls (including its cudaFuncSetAttribute smem carveout), on
// synthetic data at tools/bench.py's representative VAE-decode shape --
// correctness at this shape/dtype is already established separately (see
// the index-arithmetic simulation from the tile generalization); this harness
// only answers "how many ms/iter", nothing else. Never linked into cmpext3.
// =================================================================================
#ifdef CMPEXT3_AUTOTUNE_HARNESS
#include <cstdio>

int main() {
    // tools/bench.py's Conv3d section: HunyuanVideo-style VAE-decode shape.
    const int B = 1, C_in = 512, D_in = 8, H_in = 32, W_in = 32;
    const int C_out = 512, K_D = 3, K_H = 3, K_W = 3;
    const int s = 1, p = 1, dil = 1;
    const int D_out = (D_in + 2 * p - dil * (K_D - 1) - 1) / s + 1;
    const int H_out = (H_in + 2 * p - dil * (K_H - 1) - 1) / s + 1;
    const int W_out = (W_in + 2 * p - dil * (K_W - 1) - 1) / s + 1;

    size_t in_elems  = (size_t)B * C_in * D_in * H_in * W_in;
    size_t w_elems   = (size_t)C_out * C_in * K_D * K_H * K_W;
    size_t out_elems = (size_t)B * C_out * D_out * H_out * W_out;

    half *d_in, *d_w, *d_bias, *d_out;
    if (cudaMalloc(&d_in, in_elems * sizeof(half)) != cudaSuccess ||
        cudaMalloc(&d_w, w_elems * sizeof(half)) != cudaSuccess ||
        cudaMalloc(&d_bias, (size_t)C_out * sizeof(half)) != cudaSuccess ||
        cudaMalloc(&d_out, out_elems * sizeof(half)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }

    // 0x3C3C is a repeated fp16 bit pattern equal to 1.0h -- finite,
    // deterministic, and irrelevant to timing (the kernel's control flow
    // depends on shape/indices, not data values).
    cudaMemset(d_in, 0x3C, in_elems * sizeof(half));
    cudaMemset(d_w, 0x3C, w_elems * sizeof(half));
    cudaMemset(d_bias, 0, (size_t)C_out * sizeof(half));

    const int warmup = 5, iters = 30;
    for (int i = 0; i < warmup; ++i) {
        launch_conv3d_fp16(d_in, d_w, d_bias, d_out,
            B, C_in, D_in, H_in, W_in, C_out, K_D, K_H, K_W, D_out, H_out, W_out,
            s, s, s, p, p, p, dil, dil, dil);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        launch_conv3d_fp16(d_in, d_w, d_bias, d_out,
            B, C_in, D_in, H_in, W_in, C_out, K_D, K_H, K_W, D_out, H_out, W_out,
            s, s, s, p, p, p, dil, dil, dil);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Machine-parseable line -- setup.py's autotuner greps for this prefix.
    printf("RESULT_MS=%f\n", ms / iters);

    cudaFree(d_in); cudaFree(d_w); cudaFree(d_bias); cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
#endif // CMPEXT3_AUTOTUNE_HARNESS
