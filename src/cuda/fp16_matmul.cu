// =================================================================================
// gemm_fp16_sm75_final.cu
//
// FP16 GEMM for NVIDIA Turing (sm_75, e.g. TU102 RTX 2080 Ti / Quadro RTX 6000).
// Derived from the Ampere (sm_80) triple-buffered reference, adapted to Turing's
// hardware constraints.
//
// KEY DIFFERENCES vs the Ampere reference:
// ┌─────────────────────┬──────────────────────────┬──────────────────────────────┐
// │                     │ Ampere (reference)        │ Turing (this file)           │
// ├─────────────────────┼──────────────────────────┼──────────────────────────────┤
// │ Load instruction    │ cp.async (hardware DMA)   │ __ldg scalar/vector (cached) │
// │ Pipeline stages     │ 3 (triple-buffer)         │ 2 (double-buffer)            │
// │ BK                  │ 64                        │ 32                           │
// │ Smem (As+Bs)        │ ~107 KB  (A100: 164 KB)   │ ~33 KB  (Turing: 64 KB)      │
// │ Pipeline control    │ cp.async.commit/wait_group│ __syncthreads()              │
// └─────────────────────┴──────────────────────────┴──────────────────────────────┘
//
// WHY cp.async IS REMOVED ENTIRELY:
//   cp.async requires sm_80+. On sm_75 the PTX assembles but the instruction is
//   not in the ISA — the driver silently downgrades or the hardware raises an
//   illegal-instruction fault. There is no Turing equivalent. The correct
//   replacement is a plain register load (__ldg) followed by a smem store, which
//   is blocking by nature. A double-buffer with __syncthreads() gives the same
//   latency-hiding structure as cp.async with wait_group(1), just without the
//   DMA overlap — which was already illusory with blocking loads.
//
// WHY THE ALIGNMENT CRASH HAPPENED:
//   Both the original Turing port and the first fix attempt used
//   __ldg(reinterpret_cast<const int4*>(glob_ptr)), which requires the source
//   to be 16-byte aligned. PyTorch tensor.storage_offset() can place the data
//   pointer at any 2-byte boundary. A slice or view shifts the base by
//   offset * sizeof(half) = offset * 2 bytes — easily breaking 16-byte alignment.
//   The fix is: load 8 halves individually via __ldg(const half*). Each load is
//   2 bytes with 2-byte alignment — always safe. Turing's L1 cache issues one
//   32-byte transaction per cache-line anyway, so 8 scalar __ldg from the same
//   address range cost the same number of L1 transactions as one int4 load.
//
// WHY BK=64 IS IMPOSSIBLE ON TURING:
//   As: [2][128][64]  = 32,768 bytes
//   Bs: [2][ 64][128] = 32,768 bytes
//   Total = 65,536 bytes = exactly 64 KB — the Turing hard limit.
//   No room for the driver's own smem bookkeeping. In practice the launch fails.
//   BK=32 gives 33,024 bytes (~32 KB), leaving 32 KB for L1 partitioning.
// =================================================================================

#include <cstdio>
#include <cstdint>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// =================================================================================
// Part 1: Configuration
// =================================================================================

#define BLOCK_SIZE 256   // threads per block
#define BM 128           // tile rows (M dimension)
#define BN 128           // tile cols (N dimension)
#define BK 32            // tile depth (K dimension) — halved from Ampere's 64 to fit smem
#define STAGES 2         // double-buffer — Turing has no cp.async, so triple-buffer adds zero benefit

#define TM 8             // rows per thread
#define TN 8             // cols per thread

// Shared memory padding.
// As[BM][BK]:  stride in 4B words = BK*sizeof(half)/4 = 32*2/4 = 16. 16%32=16 ✓ no conflict.
// Bs[BK][BN]:  stride in 4B words = BN*sizeof(half)/4 = 128*2/4 = 64. 64%32= 0 → conflict.
//   With PAD_B=8: stride = (128+8)*2/4 = 68 words. 68%32 = 4 ✓ no conflict.
//   PAD_B=8 matches the Ampere reference PAD and eliminates all Bs bank conflicts.
#define PAD_A 0
#define PAD_B 8

// Smem sizes:
//   As: STAGES * BM * (BK+PAD_A) * sizeof(half) = 2*128*32*2  = 16,384 bytes
//   Bs: STAGES * BK * (BN+PAD_B) * sizeof(half) = 2*32*136*2  = 17,408 bytes
//   Total: 33,792 bytes (~33 KB) — well within Turing's 64 KB limit.
static_assert(
    STAGES * BM * (BK + PAD_A) * 2 +
    STAGES * BK * (BN + PAD_B) * 2 <= 65536u,
    "smem exceeds Turing 64 KB hard limit");

// =================================================================================
// Part 2: Global → Register → Shared load helper
//
// On Ampere this was cp.async (hardware DMA, non-blocking).
// On Turing we have no hardware async copy. The ONLY correct replacement is:
//   1. Load from global to registers (blocking — completes before next instruction).
//   2. Store from registers to smem.
//
// We use __ldg() (load-via-read-only/texture cache) for each individual half.
// Loading 8 halves individually looks more expensive than one int4 load, but:
//   - Turing's L1 line is 32 bytes. All 8 halves (16 bytes) sit in ≤1 cache line.
//   - The 8 scalar __ldg calls coalesce to 1–2 L1 transactions at the warp level.
//   - More importantly: scalar half* loads require only 2-byte alignment — always
//     satisfied. int4* loads require 16-byte alignment — frequently violated by
//     PyTorch sliced/viewed tensors, causing the "misaligned address" crash.
//
// We also provide a fast path (ldg128) used exclusively for smem→smem copies
// (the store phase), which is always 16-byte aligned by construction.
// =================================================================================

// Load 8 halves from global memory into a local array, handling out-of-bounds.
// No alignment requirement on src beyond the natural 2-byte alignment of half*.
__device__ __forceinline__
void ldg_half8(half dst[8], const half* src, bool valid)
{
    if (valid) {
        // 8 individual __ldg calls. The compiler fuses these into 128-bit LDG
        // when the pointer is statically known to be 16B aligned; otherwise it
        // emits 4×32-bit or 8×16-bit LDG — all correct regardless of alignment.
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            dst[i] = __ldg(src + i);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            dst[i] = __float2half(0.0f);
        }
    }
}

// Store 8 halves from registers to smem. smem pointers are always correctly
// aligned (base from extern __shared__, offsets are multiples of 16 bytes).
__device__ __forceinline__
void sts_half8(half* dst, const half src[8])
{
    // smem is always 128-bit aligned here — safe to use int4 store.
    int4* dst4 = reinterpret_cast<int4*>(dst);
    int4 val;
    // Pack 8 halves into 128 bits.
    half* val_h = reinterpret_cast<half*>(&val);
    #pragma unroll
    for (int i = 0; i < 8; i++) val_h[i] = src[i];
    *dst4 = val;
}

// =================================================================================
// Part 3: Kernel
// =================================================================================

__global__ void __launch_bounds__(BLOCK_SIZE)
gemm_fp16_sm75(
    const half* __restrict__ A,   // [M × K], row-major
    const half* __restrict__ B,   // [K × N], row-major
    half*       __restrict__ C,   // [M × N], row-major
    int M, int N, int K,
    int lda, int ldb, int ldc)
{
    extern __shared__ char smem_raw[];

    // ── Smem layout ─────────────────────────────────────────────────────────
    // As: [STAGES][BM][BK + PAD_A]
    // Bs: [STAGES][BK][BN + PAD_B]   (placed after As)
    half (*As)[BM][BK + PAD_A] =
        reinterpret_cast<half (*)[BM][BK + PAD_A]>(smem_raw);

    half (*Bs)[BK][BN + PAD_B] =
        reinterpret_cast<half (*)[BK][BN + PAD_B]>(
            smem_raw + STAGES * BM * (BK + PAD_A) * sizeof(half));

    // ── Accumulators & fragments ─────────────────────────────────────────────
    half2 accum[TM][TN / 2];
    half2 frag_a[TM / 2];
    half2 frag_b[TN / 2];

    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN / 2; j++)
            accum[i][j] = __float2half2_rn(0.0f);

    // ── Thread / block indices ───────────────────────────────────────────────
    const int tid = threadIdx.x;
    const int bx  = blockIdx.x;   // N-tile index
    const int by  = blockIdx.y;   // M-tile index

    const int block_row = by * BM;   // first M-row this block owns
    const int block_col = bx * BN;   // first N-col this block owns

    // ── Load-thread mapping ──────────────────────────────────────────────────
    // Each thread loads 8 halves per tile (= one 128-bit word).
    // BK=32: A tile = BM*BK = 4096 halves. 256 threads × 8 halves × 2 loads = 4096 ✓
    // BK=32: B tile = BK*BN = 4096 halves. 256 threads × 8 halves × 2 loads = 4096 ✓
    //
    // A mapping — 2 loads, stride 64 rows:
    //   row = load_a_row + i*64   (i=0,1)  → rows [0..63], [64..127]
    //   col = load_a_col          fixed     → each thread covers 8 consecutive K cols
    //   load_a_row = tid/4    → 0..63   (unique per 4 threads)
    //   load_a_col = (tid%4)*8 → 0,8,16,24 (covers the 32 BK cols in 4 groups of 8)
    //
    // B mapping — 2 loads, stride 16 rows:
    //   row = load_b_row + i*16   (i=0,1)  → rows [0..15], [16..31]
    //   col = load_b_col          fixed     → each thread covers 8 consecutive N cols
    //   load_b_row = tid/16   → 0..15
    //   load_b_col = (tid%16)*8 → 0,8,..,120

    const int load_a_row = tid / 4;
    const int load_a_col = (tid % 4) * 8;

    const int load_b_row = tid / 16;
    const int load_b_col = (tid % 16) * 8;

    // ── Compute-thread mapping ───────────────────────────────────────────────
    // 256 threads → 16 "tx" (N direction) × 16 "ty" (M direction)
    // Each thread owns TM=8 rows × TN=8 cols of output.
    const int ty = tid / 16;
    const int tx = tid % 16;
    const int thread_row = ty * TM;
    const int thread_col = tx * TN;

    const int k_tiles = (K + BK - 1) / BK;

    // =========================================================================
    // LOAD-TILE MACRO
    // Loads one (A, B) tile pair from global memory into smem stage `s`
    // at K-offset `kb`.  Boundary threads zero-fill.
    //
    // The loads go:  global → registers (ldg_half8) → smem (sts_half8).
    // This is a fully synchronous, blocking sequence — exactly what Turing
    // supports. The double-buffer structure around it provides the same
    // producer/consumer separation that cp.async + wait_group gave on Ampere.
    // =========================================================================
#define LOAD_TILE(s, kb)                                                          \
    do {                                                                           \
        half _tmp[8];                                                              \
        /* A: 2 loads per thread, rows offset by 64 */                            \
        _Pragma("unroll")                                                          \
        for (int _i = 0; _i < 2; _i++) {                                          \
            int  _r  = load_a_row + _i * 64;                                      \
            int  _c  = load_a_col;                                                \
            bool _ok = (block_row + _r < M) && ((kb) + _c < K);                  \
            ldg_half8(_tmp, A + (block_row + _r) * lda + (kb) + _c, _ok);        \
            sts_half8(&As[(s)][_r][_c], _tmp);                                    \
        }                                                                          \
        /* B: 2 loads per thread, rows offset by 16 */                            \
        _Pragma("unroll")                                                          \
        for (int _i = 0; _i < 2; _i++) {                                          \
            int  _r  = load_b_row + _i * 16;                                      \
            int  _c  = load_b_col;                                                \
            bool _ok = ((kb) + _r < K) && (block_col + _c < N);                  \
            ldg_half8(_tmp, B + ((kb) + _r) * ldb + block_col + _c, _ok);        \
            sts_half8(&Bs[(s)][_r][_c], _tmp);                                    \
        }                                                                          \
    } while (0)

    // =========================================================================
    // PROLOGUE — load tile 0 into stage 0, sync.
    //
    // With double-buffering we only need ONE tile preloaded before the loop.
    // The loop body loads tile k+1 into stage (k+1)%2 at the start of
    // iteration k, so the prologue seeds just stage 0 with tile 0.
    // =========================================================================
    if (k_tiles > 0) {
        LOAD_TILE(0, 0);
    }
    __syncthreads();   // stage 0 fully visible to all threads before loop

    // =========================================================================
    // MAIN LOOP — double-buffered
    //
    // Iteration k_step:
    //   compute_stage = k_step % 2        ← read this iteration's tile
    //   load_stage    = (k_step + 1) % 2  ← write next iteration's tile
    //
    // Barrier protocol (four steps per iteration):
    //
    //   [A] PREFETCH next tile into load_stage (if it exists).
    //       Safe to write: the previous iteration's [D] confirmed all threads
    //       finished reading this slot.  On k_step==0 the prologue sync plays
    //       that role.
    //
    //   [B] __syncthreads()  (UNCONDITIONAL)
    //       (b1) Makes the [A] prefetch above visible for the *next* iteration.
    //       (b2) Ensures all threads finished reading the slot that [A] just
    //            overwrote (from two iterations ago), preventing a write-after-read
    //            race.  With STAGES=2 both concerns collapse to this one barrier.
    //
    //   [C] COMPUTE outer product from compute_stage.
    //       All reads are safe: tile was written in the previous iteration's [A]
    //       (or by the prologue for k_step==0), and [B] (or the prologue sync)
    //       confirmed it is visible.
    //
    //   [D] __syncthreads()  (UNCONDITIONAL, including last iteration)
    //       All threads have finished reading compute_stage.  The next
    //       iteration's [A] may now overwrite this slot safely.
    //       On the last iteration no overwrite follows, but the barrier is
    //       emitted anyway to leave smem in a committed, race-free state.
    // =========================================================================
    for (int k_step = 0; k_step < k_tiles; k_step++) {

        const int compute_stage = k_step       % STAGES;
        const int load_stage    = (k_step + 1) % STAGES;

        // [A] Prefetch next tile
        if (k_step + 1 < k_tiles) {
            LOAD_TILE(load_stage, (k_step + 1) * BK);
        }

        // [B] Sync 1: makes prefetch visible; guards slot from RAW race
        __syncthreads();

        // [C] Compute outer product
        #pragma unroll
        for (int ki = 0; ki < BK; ki++) {

            // Load TM=8 A elements (column of the A tile) into 4 half2 fragments.
            // Scalar smem loads — no alignment concern, always safe.
            #pragma unroll
            for (int i = 0; i < TM; i += 2) {
                half a0 = As[compute_stage][thread_row + i    ][ki];
                half a1 = As[compute_stage][thread_row + i + 1][ki];
                frag_a[i / 2] = __halves2half2(a0, a1);
            }

            // Load TN=8 B elements (row of the B tile) via 128-bit smem read.
            // thread_col = tx*TN = (tid%16)*8. In bytes: (tid%16)*16.
            // As analyzed above, Turing smem is 4B-banked; tx stride of 4 half2
            // (= 4 banks) causes 2-way conflicts at tx=0 vs tx=8 within a warp.
            // With PAD_B=8 the row stride is 136 halves = 68 words; 68%32=4 ≠ 0,
            // which breaks the periodic column alignment and eliminates conflicts.
            int4 b_vec = *reinterpret_cast<int4*>(
                &Bs[compute_stage][ki][thread_col]);
            const half2* bh = reinterpret_cast<const half2*>(&b_vec);
            frag_b[0] = bh[0];
            frag_b[1] = bh[1];
            frag_b[2] = bh[2];
            frag_b[3] = bh[3];

            // FP16 outer product — CUDA cores only, no tensor cores, no FP32.
            #pragma unroll
            for (int i = 0; i < TM / 2; i++) {
                half2 a_top = __half2half2(frag_a[i].x);   // broadcast frag_a row 2i
                half2 a_bot = __half2half2(frag_a[i].y);   // broadcast frag_a row 2i+1
                #pragma unroll
                for (int j = 0; j < TN / 2; j++) {
                    accum[i * 2    ][j] = __hfma2(a_top, frag_b[j], accum[i * 2    ][j]);
                    accum[i * 2 + 1][j] = __hfma2(a_bot, frag_b[j], accum[i * 2 + 1][j]);
                }
            }
        }

        // [D] Sync 2: all reads from compute_stage complete; slot may be reused
        __syncthreads();
    }

#undef LOAD_TILE

    // =========================================================================
    // STORE — write accumulators to C, with boundary guards
    // =========================================================================
    const int c_row = block_row + thread_row;
    const int c_col = block_col + thread_col;

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        if (c_row + i >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN / 2; j++) {
            int col0 = c_col + j * 2;
            int col1 = col0 + 1;
            if (col0 < N) C[(c_row + i) * ldc + col0] = accum[i][j].x;
            if (col1 < N) C[(c_row + i) * ldc + col1] = accum[i][j].y;
        }
    }
}

// =================================================================================
// Part 4: Launcher
// =================================================================================

void launch_matmul_fp16(
    const void* input_ptr,
    const void* weight_ptr,
    void*       output_ptr,
    int M, int N, int K)
{
    // ── Sanity checks ────────────────────────────────────────────────────────
    assert(K > 0 && "K must be > 0");

    // M, N, K need NOT be multiples of BM/BN/BK/TN.
    //
    // Partial K tiles: LOAD_TILE's boundary guard
    //   _ok = ... && ((kb) + _c < K)
    // zero-fills any smem slot whose global source is out of range.
    // __hfma2(a, 0, c) == c, so zero-filled lanes contribute nothing to the
    // accumulator. The BK compute loop running past the real K edge is safe.
    //
    // Partial M/N tiles: the store phase guards every write with
    //   if (col0 < N) / if (col1 < N) / if (c_row + i < M)
    // so out-of-bounds threads write nothing.
    //
    // The smem B read (int4) is always 16B-aligned by construction:
    //   thread_col = (tid%16)*8 → byte offset = (tid%16)*16, divisible by 16.
    //   Bs row stride = (BN+PAD_B)*sizeof(half) = 272 bytes, divisible by 16.
    //   Bs base follows As (16384 bytes from a 16B-aligned extern __shared__).
    // So N%TN==0 is NOT required for the smem read alignment.

    const half* A = reinterpret_cast<const half*>(input_ptr);
    const half* B = reinterpret_cast<const half*>(weight_ptr);
    half*       C = reinterpret_cast<half*>(output_ptr);

    const int lda = K, ldb = N, ldc = N;

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    const int smem_bytes =
        STAGES * BM * (BK + PAD_A) * (int)sizeof(half) +
        STAGES * BK * (BN + PAD_B) * (int)sizeof(half);
    // smem_bytes = 33,792 — below the default 48 KB limit, but we set the
    // attribute anyway for clarity and to allow future tuning. It's a
    // compile-time constant, so set it once per process rather than on
    // every matmul/linear call -- this is one of the hottest ops in the
    // model (every Linear layer, every block, every step).
    static bool smem_attr_set = false;
    if (!smem_attr_set) {
        cudaFuncSetAttribute(gemm_fp16_sm75,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_bytes);
        smem_attr_set = true;
    }

    gemm_fp16_sm75<<<grid, block, smem_bytes, /*stream=*/0>>>(
        A, B, C, M, N, K, lda, ldb, ldc);
}

// =================================================================================
// Part 5: Bias addition (unchanged, sm_75 compatible)
// =================================================================================

__global__ void add_bias_fp16_vectorized(
    half2* __restrict__       output,
    const half2* __restrict__ bias,
    int total_h2,
    int width_h2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_h2)
        output[idx] = __hadd2(output[idx], bias[idx % width_h2]);
}

void launch_add_bias_fp16(void* output_ptr, const void* bias_ptr, int rows, int cols)
{
    if (cols % 2 != 0) return;
    half2*       out   = reinterpret_cast<half2*>(output_ptr);
    const half2* bias  = reinterpret_cast<const half2*>(bias_ptr);
    const int    total = (rows * cols) / 2;
    const int    wh2   = cols / 2;
    add_bias_fp16_vectorized<<<(total + 255) / 256, 256>>>(out, bias, total, wh2);
}
