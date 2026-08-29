#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>


#define DIV_CEIL(a, b) (((a) + (b) - 1) / (b))

// CTILE = output channels processed per thread. Independently tunable via
// -DCONV3D_FP32_CTILE=.. (see ../../setup.py's autotune_kernel_tile()) --
// prefixed per-file so it can't collide with fp32_conv.cu's/
// fp32_ConvTranspose2d.cu's own CTILE overrides.
#ifndef CONV3D_FP32_CTILE
#define CONV3D_FP32_CTILE 8
#endif
#define CTILE CONV3D_FP32_CTILE

// =================================================================================
// Register-blocked Conv3d -- FP32
//
// Direct extension of fp32_conv.cu's conv2d_fp32_kernel_ampere_opt_v2 to a
// depth axis, replacing the naive one-thread-per-voxel kernel this file used
// to contain (see README.md / cmpext3/__init__.py's enable() comment for the
// ~460ms-vs-cuDNN's-83ms numbers that motivated this rewrite). No
// shared-memory tiling, same tradeoff as the 2D fp32 kernel: this design was
// already proven on this hardware, unlike the heavier smem-tiled GEMM
// approach used for fp16 (see fp16_conv3d.cu) which carries more
// occupancy/correctness risk for a first rewrite.
//
// Thread coarsening: each thread computes 2 W-adjacent output pixels x CTILE
// output channels (2*CTILE accumulator registers total, sum0/sum1). Pointer
// hoisting: CTILE weight pointers advanced via ++ once per (kd,kh,kw) step
// instead of recomputed by multiplication, so each loaded weight value is
// reused across both pixels and the address arithmetic leaves the hot loop.
//
// Batch and output-depth are folded into a single grid.z = B*D_out and
// unpacked inside the kernel (b_idx = blockIdx.z / D_out, d_out = blockIdx.z
// % D_out) -- D_out rides along the axis grid.z already owned exclusively by
// B in the 2D kernel.
//
// Weight tensor is contiguous [C_out, C_in, K_D, K_H, K_W]; the pointer walk
// through kd -> kh -> kw -> (next c_in) must visit every element in that
// exact order regardless of whether a given (kd,kh,kw) step is in-bounds, or
// the weight pointers desync from the input reads. Out-of-range depth/row
// values are handled the same way the 2D kernel handles out-of-range rows:
// skip the *read* (leave that step's input contribution at 0), never skip
// the pointer advance.
//
// Plain scalar multiply-accumulate throughout, so --fmad=false (see
// ../../setup.py) keeps nvcc from fusing these into FFMA, same as every
// other kernel in this project.
// =================================================================================

__global__ void __launch_bounds__(256) conv3d_fp32_kernel_opt(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C_in, int D_in, int H_in, int W_in,
    int C_out, int K_D, int K_H, int K_W,
    int D_out, int H_out, int W_out,
    int s_d, int s_h, int s_w, int p_d, int p_h, int p_w, int d_d, int d_h, int d_w
) {
    // ----------------------------------------------------------------
    // 1. Thread -> output coordinate mapping
    // ----------------------------------------------------------------
    int tid_x = threadIdx.x; // 0..15
    int tid_y = threadIdx.y; // 0..15

    int blocks_in_row = DIV_CEIL(W_out, 32);

    int blk_w = blockIdx.x % blocks_in_row;
    int blk_h = blockIdx.x / blocks_in_row;

    int w_out_0 = blk_w * 32 + tid_x * 2;
    int w_out_1 = w_out_0 + 1;

    int h_out = blk_h * 16 + tid_y;
    int c_out_base = blockIdx.y * CTILE;

    // grid.z packs (b, d_out) together: D_out depth-slices per batch entry.
    int b_idx = blockIdx.z / D_out;
    int d_out = blockIdx.z % D_out;

    if (h_out >= H_out) return;

    bool valid_w0 = (w_out_0 < W_out);
    bool valid_w1 = (w_out_1 < W_out);
    if (!valid_w0 && !valid_w1) return;

    // ----------------------------------------------------------------
    // 2. Accumulators and base pointers
    // ----------------------------------------------------------------
    float sum0[CTILE] = {0.0f};
    float sum1[CTILE] = {0.0f};

    int d_in_base = d_out * s_d - p_d;
    int h_in_base = h_out * s_h - p_h;
    int w_in_base_0 = w_out_0 * s_w - p_w;
    int w_in_base_1 = w_out_1 * s_w - p_w;

    long long input_batch_offset = (long long)b_idx * C_in * D_in * H_in * W_in;
    const float* input_base_ptr = input + input_batch_offset;

    // Pointer hoisting: precompute CTILE output-channel weight start pointers,
    // advanced by ++ in the hot loop instead of recomputed via multiplication.
    const float* w_ptrs[CTILE];
    int weight_stride_oc = C_in * K_D * K_H * K_W;

    #pragma unroll
    for (int k = 0; k < CTILE; ++k) {
        int c = c_out_base + k;
        // Out-of-range channel points at weight[0] to avoid an illegal
        // address; the write-back stage filters it out via current_c_out.
        w_ptrs[k] = (c < C_out) ? (weight + (long long)c * weight_stride_oc) : weight;
    }

    // ----------------------------------------------------------------
    // 3. Core accumulation loop
    // ----------------------------------------------------------------
    for (int c = 0; c < C_in; ++c) {
        const float* current_in_channel = input_base_ptr + (long long)c * D_in * H_in * W_in;

        for (int kd = 0; kd < K_D; ++kd) {
            int d_in = d_in_base + kd * d_d;
            bool d_valid = (d_in >= 0 && d_in < D_in);
            long long d_offset = d_valid ? (long long)d_in * H_in * W_in : 0;

            for (int i = 0; i < K_H; ++i) {
                int in_row = h_in_base + i * d_h;
                bool row_valid = d_valid && (in_row >= 0 && in_row < H_in);
                long long row_offset = row_valid ? (d_offset + (long long)in_row * W_in) : 0;

                for (int j = 0; j < K_W; ++j) {
                    float in_val0 = 0.0f;
                    float in_val1 = 0.0f;

                    if (row_valid) {
                        int in_col0 = w_in_base_0 + j * d_w;
                        int in_col1 = w_in_base_1 + j * d_w;

                        if (valid_w0 && in_col0 >= 0 && in_col0 < W_in) {
                            in_val0 = current_in_channel[row_offset + in_col0];
                        }
                        if (valid_w1 && in_col1 >= 0 && in_col1 < W_in) {
                            in_val1 = current_in_channel[row_offset + in_col1];
                        }
                    }

                    // Weight pointers must advance exactly once per
                    // (kd,kh,kw) step regardless of d_valid/row_valid, to
                    // stay in sync with the flattened weight layout.
                    #pragma unroll
                    for (int k = 0; k < CTILE; ++k) {
                        float w_val = *w_ptrs[k];
                        w_ptrs[k]++;

                        sum0[k] += in_val0 * w_val;
                        sum1[k] += in_val1 * w_val;
                    }
                }
            }
        }
    }

    // ----------------------------------------------------------------
    // 4. Write back
    // ----------------------------------------------------------------
    long long out_batch_offset = (long long)b_idx * C_out * D_out * H_out * W_out;
    long long out_channel_stride = (long long)D_out * H_out * W_out;
    long long out_depth_offset = (long long)d_out * H_out * W_out;

    auto write_back = [&](int w_curr, float* acc, bool w_valid) {
        if (!w_valid) return;
        long long out_spatial = out_depth_offset + (long long)h_out * W_out + w_curr;

        #pragma unroll
        for (int k = 0; k < CTILE; ++k) {
            int current_c_out = c_out_base + k;
            if (current_c_out < C_out) {
                float val = acc[k];
                if (bias) val += bias[current_c_out];

                long long out_addr = out_batch_offset + (long long)current_c_out * out_channel_stride + out_spatial;
                output[out_addr] = val;
            }
        }
    };

    write_back(w_out_0, sum0, valid_w0);
    write_back(w_out_1, sum1, valid_w1);
}

// -------------------------------------------------------------------------
// Launcher (name/signature unchanged -- see src/main.cpp)
// -------------------------------------------------------------------------
void launch_conv3d_fp32(
    const float* input, const float* weight, const float* bias, float* output,
    int B, int C_in, int D_in, int H_in, int W_in,
    int C_out, int K_D, int K_H, int K_W,
    int D_out, int H_out, int W_out,
    int s_d, int s_h, int s_w,
    int p_d, int p_h, int p_w,
    int d_d, int d_h, int d_w)
{
    long long total = (long long)B * C_out * D_out * H_out * W_out;
    if (total <= 0) return;

    // Block: 16x16 = 256 threads, matching __launch_bounds__(256).
    dim3 threads_per_block(16, 16);

    // Grid: x packs W/H output tiles (32x16 pixels/block), y is C_out/CTILE,
    // z folds (batch, output-depth) together.
    int blocks_w = DIV_CEIL(W_out, 32);
    int blocks_h = DIV_CEIL(H_out, 16);
    int grid_x = blocks_w * blocks_h;
    int grid_y = DIV_CEIL(C_out, CTILE);
    int grid_z = B * D_out;

    dim3 blocks(grid_x, grid_y, grid_z);

    conv3d_fp32_kernel_opt<<<blocks, threads_per_block>>>(
        input, weight, bias, output,
        B, C_in, D_in, H_in, W_in, C_out, K_D, K_H, K_W, D_out, H_out, W_out,
        s_d, s_h, s_w, p_d, p_h, p_w, d_d, d_h, d_w
    );
}

// =================================================================================
// Autotune benchmark harness (opt-in, NOT part of the real build)
//
// Only compiled when -DCMPEXT3_AUTOTUNE_HARNESS is passed -- see
// autotune_kernel_tile() in ../../setup.py. Exercises the real
// launch_conv3d_fp32 on synthetic data at tools/bench.py's Conv3d shape,
// timed with cudaEvents. Never linked into cmpext3.
// =================================================================================
#ifdef CMPEXT3_AUTOTUNE_HARNESS
#include <cstdio>

int main() {
    const int B = 1, C_in = 512, D_in = 8, H_in = 32, W_in = 32;
    const int C_out = 512, K_D = 3, K_H = 3, K_W = 3;
    const int s = 1, p = 1, dil = 1;
    const int D_out = (D_in + 2 * p - dil * (K_D - 1) - 1) / s + 1;
    const int H_out = (H_in + 2 * p - dil * (K_H - 1) - 1) / s + 1;
    const int W_out = (W_in + 2 * p - dil * (K_W - 1) - 1) / s + 1;

    size_t in_elems  = (size_t)B * C_in * D_in * H_in * W_in;
    size_t w_elems   = (size_t)C_out * C_in * K_D * K_H * K_W;
    size_t out_elems = (size_t)B * C_out * D_out * H_out * W_out;

    float *d_in, *d_w, *d_bias, *d_out;
    if (cudaMalloc(&d_in, in_elems * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_w, w_elems * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_bias, (size_t)C_out * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_out, out_elems * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }

    cudaMemset(d_in, 0x3F, in_elems * sizeof(float));
    cudaMemset(d_w, 0x3F, w_elems * sizeof(float));
    cudaMemset(d_bias, 0, (size_t)C_out * sizeof(float));

    const int warmup = 5, iters = 20;
    for (int i = 0; i < warmup; ++i) {
        launch_conv3d_fp32(d_in, d_w, d_bias, d_out,
            B, C_in, D_in, H_in, W_in, C_out, K_D, K_H, K_W, D_out, H_out, W_out,
            s, s, s, p, p, p, dil, dil, dil);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        launch_conv3d_fp32(d_in, d_w, d_bias, d_out,
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

    printf("RESULT_MS=%f\n", ms / iters);

    cudaFree(d_in); cudaFree(d_w); cudaFree(d_bias); cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
#endif // CMPEXT3_AUTOTUNE_HARNESS
