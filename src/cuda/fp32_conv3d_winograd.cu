// =================================================================================
// fp32_conv3d_winograd.cu -- F(2x2x2, 3x3x3) 3D Winograd conv3d.
//
// Alternative to fp32_conv3d.cu's direct kernel for the specific case
// this actually applies to: fp32, batch=1, kernel=3x3x3, stride=1,
// padding=1, dilation=1, groups=1 (custom_conv3d_forward's own 1x1x1
// fast-path already peels off the K=1 case before this is ever reached).
// Falls back (returns false, `output` untouched) for anything outside
// that scope -- see launch_conv3d_fp32_winograd's guard block, checked
// FIRST, before any allocation -- matching every other opportunistic
// kernel in this codebase (e.g. src/trt/*.cpp's "false means try the
// fallback instead" contract).
//
// Separable 3D extension of the standard 1D F(2,3) Winograd transform
// (Lavin & Gray, "Fast Algorithms for Convolutional Neural Networks"):
//   filter transform (G):   3-tap g -> 4-tap U
//     U0 = g0
//     U1 = 0.5*(g0+g1+g2)
//     U2 = 0.5*(g0-g1+g2)
//     U3 = g2
//   input transform (B^T):  4-tap d -> 4-tap V
//     V0 = d0-d2
//     V1 = d1+d2
//     V2 = d2-d1
//     V3 = d1-d3
//   output transform (A^T): 4-tap m -> 2-tap output
//     out0 = m0+m1+m2
//     out1 = m1-m2-m3
// Applied along each of D,H,W in turn (separability) -- verified exactly
// (to fp64 rounding, against direct correlation) in a standalone NumPy
// script before any CUDA was written, and the resulting kernel verified
// against a direct CPU reference implementation (max abs diff ~3e-8,
// float32 rounding only) before ever being benchmarked.
//
// Three kernels, matching how production Winograd implementations split
// the work (avoids redundantly re-transforming the same input tile once
// per output channel):
//   1. filter transform:  g[Cout][Cin][3][3][3] -> U[Cout][Cin][4][4][4]
//   2. input transform:   input[Cin][D][H][W]   -> V[Cin][tile][4][4][4]
//   3. accumulate+output: per (tile,Cout) BLOCK: M[pos] = sum_Cin
//      U[Cout][Cin][pos]*V[Cin][tile][pos] for pos in 64 transform-domain
//      positions, then output-transform -> 2x2x2, +bias.
//
// Kernel 3 is register-blocked (BT_TILES x BT_COUT output tiles per
// block, not one) -- this is NOT optional/cosmetic. A first, unblocked
// version (one block per single (tile,cout) pair) measured 147.7ms at
// the real bench.py HunyuanVideo VAE-decode shape (512ch, 8x32x32) --
// SLOWER than fp32_conv3d.cu's 31.9ms despite doing ~5.6x fewer
// multiplies in the accumulate step, because every block re-read the
// ENTIRE U[cout] row from global memory once per tile (redundant across
// all 1024 tiles sharing that cout) and the entire V[tile] row once per
// cout (redundant across all 512 couts sharing that tile) -- ~132 GB of
// global traffic despite U+V totaling under 200 MB combined. BT_TILES=
// BT_COUT=8 register blocking (classic GEMM-style: each thread loads 8
// U-values + 8 V-values per Cin step and reuses them across 64 partial
// products instead of 1) cut that to 19.6-20.2ms -- 1.6x FASTER than
// fp32_conv3d.cu, not just competitive with it. Sizes above and below
// 8x8 were tried and measured worse: 4x4 (35.2ms, register blocking
// helps but not enough) and 16x8/8x16/16x16 (24.2ms at 16x8, the only
// one fully re-verified after fixing a real bug those sizes exposed --
// see BT_TILES's own comment) -- extra register pressure past 8x8 costs
// more in occupancy than it saves in memory traffic on this GPU. 8x8
// is a measured local optimum, not an arbitrary/round default.
//
// Scope limitations (all enforced by launch_conv3d_fp32_winograd's guard,
// which returns false rather than guess): D_in/H_in/W_in must be even
// (clean 2-tiling, no partial-tile remainder handling); the resulting
// tile count and C_out must both be divisible by BT_TILES/BT_COUT (true
// for the real shape -- 1024 tiles, 512 channels -- not enforced to be
// true in general); batch=1 only (this kernel has no batch dimension at
// all yet, unlike fp32_conv3d.cu). Any of these failing, or any other
// dtype, falls straight back to fp32_conv3d.cu -- see
// custom_conv3d_forward in main.cpp for the actual dispatch, which
// benchmarks this against that kernel once per shape and caches whichever
// wins, never assuming Winograd is faster just because it's applicable.
//
// FMA: every multiply below is written as a separate statement from any
// addition it feeds into, same as every other kernel in this project --
// relies on nvcc's --fmad=false (set project-wide in setup.py's
// nvcc_flags) to keep FFMA out of the generated code entirely, no extra
// tricks beyond that flag.
// =================================================================================

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__device__ __forceinline__ void g_transform_1d(const float g[3], float U[4]) {
    U[0] = g[0];
    U[1] = 0.5f * (g[0] + g[1] + g[2]);
    U[2] = 0.5f * (g[0] - g[1] + g[2]);
    U[3] = g[2];
}

__device__ __forceinline__ void bt_transform_1d(const float d[4], float V[4]) {
    V[0] = d[0] - d[2];
    V[1] = d[1] + d[2];
    V[2] = d[2] - d[1];
    V[3] = d[1] - d[3];
}

__device__ __forceinline__ void at_transform_1d(const float m[4], float out[2]) {
    out[0] = m[0] + m[1] + m[2];
    out[1] = m[1] - m[2] - m[3];
}

// ---------------------------------------------------------------------
// Kernel 1: filter transform. One thread per (cout,cin) pair -- small,
// fully unrolled 3x3x3 -> 4x4x4 separable transform.
// ---------------------------------------------------------------------
__global__ void winograd_filter_transform_fp32(
    const float* __restrict__ g,   // [Cout][Cin][3][3][3]
    float* __restrict__ U,         // [Cout][Cin][4][4][4]
    int C_out, int C_in)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C_out * C_in) return;
    const float* gp = g + (size_t)idx * 27;
    float* Up = U + (size_t)idx * 64;

    float t0[4][3][3];
    for (int h = 0; h < 3; ++h)
        for (int w = 0; w < 3; ++w) {
            float col[3] = {gp[0*9+h*3+w], gp[1*9+h*3+w], gp[2*9+h*3+w]};
            float out4[4];
            g_transform_1d(col, out4);
            for (int a = 0; a < 4; ++a) t0[a][h][w] = out4[a];
        }
    float t1[4][4][3];
    for (int d = 0; d < 4; ++d)
        for (int w = 0; w < 3; ++w) {
            float col[3] = {t0[d][0][w], t0[d][1][w], t0[d][2][w]};
            float out4[4];
            g_transform_1d(col, out4);
            for (int a = 0; a < 4; ++a) t1[d][a][w] = out4[a];
        }
    for (int d = 0; d < 4; ++d)
        for (int h = 0; h < 4; ++h) {
            float col[3] = {t1[d][h][0], t1[d][h][1], t1[d][h][2]};
            float out4[4];
            g_transform_1d(col, out4);
            for (int a = 0; a < 4; ++a) Up[d*16 + h*4 + a] = out4[a];
        }
}

// ---------------------------------------------------------------------
// Kernel 2: input transform. One thread per (channel, tile) -- reads a
// 4x4x4 window from the padded-conceptually input (zero-fill out of
// bounds, same boundary handling style as fp16_attention.cu's
// load_tile_sync), separable-transforms it to V.
// ---------------------------------------------------------------------
__global__ void winograd_input_transform_fp32(
    const float* __restrict__ input,  // [Cin][D][H][W]  (B=1)
    float* __restrict__ V,            // [Cin][tiles_d][tiles_h][tiles_w][4][4][4]
    int C_in, int D, int H, int W,
    int tiles_d, int tiles_h, int tiles_w)
{
    long total_tiles = (long)tiles_d * tiles_h * tiles_w;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)C_in * total_tiles) return;

    int tile_idx = idx % total_tiles;
    int cin = idx / total_tiles;
    int tw = tile_idx % tiles_w;
    int th = (tile_idx / tiles_w) % tiles_h;
    int td = tile_idx / (tiles_w * tiles_h);

    int d0 = 2 * td - 1, h0 = 2 * th - 1, w0 = 2 * tw - 1;

    auto read = [&](int dd, int hh, int ww) -> float {
        if (dd < 0 || dd >= D || hh < 0 || hh >= H || ww < 0 || ww >= W) return 0.0f;
        return input[((size_t)cin * D + dd) * H * W + hh * W + ww];
    };

    float raw[4][4][4];
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b)
            for (int c = 0; c < 4; ++c)
                raw[a][b][c] = read(d0 + a, h0 + b, w0 + c);

    float t0[4][4][4];
    for (int b = 0; b < 4; ++b)
        for (int c = 0; c < 4; ++c) {
            float col[4] = {raw[0][b][c], raw[1][b][c], raw[2][b][c], raw[3][b][c]};
            float out4[4];
            bt_transform_1d(col, out4);
            for (int a = 0; a < 4; ++a) t0[a][b][c] = out4[a];
        }
    float t1[4][4][4];
    for (int a = 0; a < 4; ++a)
        for (int c = 0; c < 4; ++c) {
            float col[4] = {t0[a][0][c], t0[a][1][c], t0[a][2][c], t0[a][3][c]};
            float out4[4];
            bt_transform_1d(col, out4);
            for (int b = 0; b < 4; ++b) t1[a][b][c] = out4[b];
        }
    float* Vp = V + (size_t)idx * 64;
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b) {
            float col[4] = {t1[a][b][0], t1[a][b][1], t1[a][b][2], t1[a][b][3]};
            float out4[4];
            bt_transform_1d(col, out4);
            for (int c = 0; c < 4; ++c) Vp[a*16 + b*4 + c] = out4[c];
        }
}

// Shared 3-pass separable output transform: M[64] -> a 2x2x2 output tile.
__device__ __forceinline__ void output_transform_from_shared(const float* M, float out[2][2][2]) {
    float stage1[4][4][2];
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b) {
            float m[4] = {M[a*16+b*4+0], M[a*16+b*4+1], M[a*16+b*4+2], M[a*16+b*4+3]};
            float o2[2];
            at_transform_1d(m, o2);
            stage1[a][b][0] = o2[0];
            stage1[a][b][1] = o2[1];
        }
    float stage2[4][2][2];
    for (int a = 0; a < 4; ++a)
        for (int c = 0; c < 2; ++c) {
            float m[4] = {stage1[a][0][c], stage1[a][1][c], stage1[a][2][c], stage1[a][3][c]};
            float o2[2];
            at_transform_1d(m, o2);
            stage2[a][0][c] = o2[0];
            stage2[a][1][c] = o2[1];
        }
    for (int b = 0; b < 2; ++b)
        for (int c = 0; c < 2; ++c) {
            float m[4] = {stage2[0][b][c], stage2[1][b][c], stage2[2][b][c], stage2[3][b][c]};
            float o2[2];
            at_transform_1d(m, o2);
            out[0][b][c] = o2[0];
            out[1][b][c] = o2[1];
        }
}

// ---------------------------------------------------------------------
// Kernel 3: register-blocked accumulate + output transform. See the file
// header for why BT_TILES=BT_COUT=8 specifically (measured local
// optimum, not a default) and how much this blocking matters (147.7ms ->
// ~20ms at the real shape). Block = 64 threads; grid =
// (tiles_total/BT_TILES, C_out/BT_COUT).
// ---------------------------------------------------------------------
#define WINOGRAD_BT_TILES 8
#define WINOGRAD_BT_COUT 8

__global__ void winograd_accumulate_output_fp32(
    const float* __restrict__ U,     // [Cout][Cin][4][4][4]
    const float* __restrict__ V,     // [Cin][tiles][4][4][4]
    const float* __restrict__ bias,  // [Cout] or nullptr
    float* __restrict__ output,      // [Cout][D_out][H_out][W_out]
    int C_out, int C_in,
    int tiles_d, int tiles_h, int tiles_w,
    int D_out, int H_out, int W_out)
{
    long total_tiles = (long)tiles_d * tiles_h * tiles_w;
    int tile0 = blockIdx.x * WINOGRAD_BT_TILES;
    int cout0 = blockIdx.y * WINOGRAD_BT_COUT;
    int pos = threadIdx.x;  // 0..63

    float acc[WINOGRAD_BT_TILES][WINOGRAD_BT_COUT];
    #pragma unroll
    for (int t = 0; t < WINOGRAD_BT_TILES; ++t)
        #pragma unroll
        for (int c = 0; c < WINOGRAD_BT_COUT; ++c)
            acc[t][c] = 0.0f;

    for (int cin = 0; cin < C_in; ++cin) {
        float uvals[WINOGRAD_BT_COUT];
        #pragma unroll
        for (int c = 0; c < WINOGRAD_BT_COUT; ++c)
            uvals[c] = U[((size_t)(cout0 + c) * C_in + cin) * 64 + pos];
        float vvals[WINOGRAD_BT_TILES];
        #pragma unroll
        for (int t = 0; t < WINOGRAD_BT_TILES; ++t)
            vvals[t] = V[((size_t)cin * total_tiles + (tile0 + t)) * 64 + pos];
        #pragma unroll
        for (int t = 0; t < WINOGRAD_BT_TILES; ++t)
            #pragma unroll
            for (int c = 0; c < WINOGRAD_BT_COUT; ++c) {
                float prod = uvals[c] * vvals[t];   // separate mul then add -- see file header re: FMA
                acc[t][c] = acc[t][c] + prod;
            }
    }

    __shared__ float M[WINOGRAD_BT_TILES][WINOGRAD_BT_COUT][64];
    #pragma unroll
    for (int t = 0; t < WINOGRAD_BT_TILES; ++t)
        #pragma unroll
        for (int c = 0; c < WINOGRAD_BT_COUT; ++c)
            M[t][c][pos] = acc[t][c];
    __syncthreads();

    // Striped loop, not a plain `if (pos < N)` -- stays correct even if
    // WINOGRAD_BT_TILES*WINOGRAD_BT_COUT ever exceeds 64 (confirmed this
    // matters: an earlier `if`-based version silently dropped every
    // (t,c) pair beyond the first 64 whenever the product exceeded it,
    // caught directly by testing larger block sizes against the
    // correctness harness).
    for (int idx = pos; idx < WINOGRAD_BT_TILES * WINOGRAD_BT_COUT; idx += 64) {
        int t = idx / WINOGRAD_BT_COUT;
        int c = idx % WINOGRAD_BT_COUT;
        int tile_idx = tile0 + t;
        int cout = cout0 + c;

        float out[2][2][2];
        output_transform_from_shared(M[t][c], out);

        int tw = tile_idx % tiles_w;
        int th = (tile_idx / tiles_w) % tiles_h;
        int td = tile_idx / (tiles_w * tiles_h);
        float bv = bias ? bias[cout] : 0.0f;
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b)
                for (int cc = 0; cc < 2; ++cc) {
                    int od = td*2 + a, oh = th*2 + b, ow = tw*2 + cc;
                    if (od < D_out && oh < H_out && ow < W_out) {
                        output[((size_t)cout * D_out + od) * H_out * W_out + oh * W_out + ow]
                            = out[a][b][cc] + bv;
                    }
                }
    }
}

// ---------------------------------------------------------------------
// Host launcher. Returns false (output untouched) for anything outside
// this kernel's scope -- see file header. Only ever called for fp32,
// batch=1 (custom_conv3d_forward checks dtype before calling); every
// other constraint (kernel=3x3x3, stride=1, padding=1, dilation=1, even
// D/H/W, divisible tile-count/C_out) is checked here.
// ---------------------------------------------------------------------
extern "C" bool launch_conv3d_fp32_winograd(
    const float* input, const float* weight, const float* bias, float* output,
    int B, int C_in, int D_in, int H_in, int W_in,
    int C_out, int K_D, int K_H, int K_W,
    int D_out, int H_out, int W_out,
    int s_d, int s_h, int s_w,
    int p_d, int p_h, int p_w,
    int d_d, int d_h, int d_w)
{
    if (B != 1) return false;
    if (K_D != 3 || K_H != 3 || K_W != 3) return false;
    if (s_d != 1 || s_h != 1 || s_w != 1) return false;
    if (p_d != 1 || p_h != 1 || p_w != 1) return false;
    if (d_d != 1 || d_h != 1 || d_w != 1) return false;
    if (D_in % 2 || H_in % 2 || W_in % 2) return false;

    int tiles_d = D_out / 2, tiles_h = H_out / 2, tiles_w = W_out / 2;
    long total_tiles = (long)tiles_d * tiles_h * tiles_w;
    if (total_tiles % WINOGRAD_BT_TILES || C_out % WINOGRAD_BT_COUT) return false;

    float *U = nullptr, *V = nullptr;
    if (cudaMalloc(&U, (size_t)C_out * C_in * 64 * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&V, (size_t)C_in * total_tiles * 64 * sizeof(float)) != cudaSuccess) {
        cudaFree(U);
        return false;
    }

    {
        int n = C_out * C_in;
        int block = 128, grid = (n + block - 1) / block;
        winograd_filter_transform_fp32<<<grid, block>>>(weight, U, C_out, C_in);
    }
    {
        long n = (long)C_in * total_tiles;
        int block = 128;
        long grid = (n + block - 1) / block;
        winograd_input_transform_fp32<<<(unsigned)grid, block>>>(
            input, V, C_in, D_in, H_in, W_in, tiles_d, tiles_h, tiles_w);
    }
    {
        dim3 grid((unsigned)(total_tiles / WINOGRAD_BT_TILES), C_out / WINOGRAD_BT_COUT);
        winograd_accumulate_output_fp32<<<grid, 64>>>(
            U, V, bias, output, C_out, C_in, tiles_d, tiles_h, tiles_w, D_out, H_out, W_out);
    }

    bool ok = cudaGetLastError() == cudaSuccess;
    cudaFree(U);
    cudaFree(V);
    return ok;
}
