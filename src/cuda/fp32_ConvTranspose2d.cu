#include <cuda.h>
#include <cuda_runtime.h>

#define DIV_CEIL(a, b) (((a) + (b) - 1) / (b))

// CTILE = output channels processed per thread. Independently tunable via
// -DCONVT2D_FP32_CTILE=.. (see ../../setup.py's autotune_kernel_tile()) --
// prefixed per-file so it can't collide with fp32_conv.cu's/
// fp32_conv3d.cu's own CTILE overrides.
#ifndef CONVT2D_FP32_CTILE
#define CONVT2D_FP32_CTILE 8
#endif
#define CTILE CONVT2D_FP32_CTILE

// =================================================================================
// Register-blocked ConvTranspose2d -- FP32
//
// Replaces the previous conv_transpose2d_fp32_opt_kernel (one thread per
// single output channel, single output pixel -- no channel or pixel reuse
// at all) with the same thread-coarsening + pointer-hoisting design used in
// fp32_conv.cu / fp32_conv3d.cu: 2 W-adjacent output pixels x CTILE output
// channels per thread (2*CTILE accumulator registers), 16x16-thread blocks.
//
// Weight layout differs from regular conv: PyTorch's ConvTranspose2d weight
// is [C_in, C_out, K_H, K_W] (C_in outermost), not [C_out, C_in, K_H, K_W].
// This means, unlike fp32_conv.cu/fp32_conv3d.cu, a single output channel's
// weight slice is NOT one contiguous run across all of C_in -- it's
// contiguous only within one (c_in, c_out) pair's K_H*K_W window. So the
// CTILE per-channel weight pointers are recomputed via one multiply per c_in step
// (not walked via ++ across the whole C_in loop), then walked via ++ across
// the contiguous K_H*K_W window for that c_in -- kh/kw are looped in
// row-major (kh outer, kw inner) order to match that contiguous layout.
//
// The output-pixel -> input-pixel mapping is inverted from regular conv
// (h_in = (h_out + pad - kh*dil) / stride, valid only if that division is
// exact and in range) -- carried over unchanged from the previous kernel,
// which already had this the right way round.
//
// Plain scalar multiply-accumulate, so --fmad=false (see ../../setup.py)
// keeps nvcc from fusing these into FFMA, same as every other kernel here.
// =================================================================================

__global__ void __launch_bounds__(256) conv_transpose2d_fp32_kernel_opt(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C_in, int H_in, int W_in,
    int C_out, int K_H, int K_W,
    int H_out, int W_out,
    int stride_h, int stride_w, int pad_h, int pad_w, int dil_h, int dil_w
) {
    // ----------------------------------------------------------------
    // 1. Thread -> output coordinate mapping (same scheme as
    //    conv2d_fp32_kernel_ampere_opt_v2 / conv3d_fp32_kernel_opt)
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
    int b_idx = blockIdx.z;

    if (h_out >= H_out) return;

    bool valid_w0 = (w_out_0 < W_out);
    bool valid_w1 = (w_out_1 < W_out);
    if (!valid_w0 && !valid_w1) return;

    // ----------------------------------------------------------------
    // 2. Accumulators and base pointers
    // ----------------------------------------------------------------
    float sum0[CTILE] = {0.0f};
    float sum1[CTILE] = {0.0f};

    long long input_batch_offset = (long long)b_idx * C_in * H_in * W_in;
    const float* input_base_ptr = input + input_batch_offset;

    long long weight_stride_cin  = (long long)C_out * K_H * K_W;   // Weight [C_in, C_out, K_H, K_W]
    long long weight_stride_cout = (long long)K_H * K_W;

    const float* w_ptrs[CTILE];

    // ----------------------------------------------------------------
    // 3. Core accumulation loop -- c_in outermost (weight pointers are
    //    recomputed once per c_in, then walked contiguously through kh/kw).
    // ----------------------------------------------------------------
    for (int c_in = 0; c_in < C_in; ++c_in) {
        const float* input_c_ptr = input_base_ptr + (long long)c_in * H_in * W_in;

        #pragma unroll
        for (int k = 0; k < CTILE; ++k) {
            int co = c_out_base + k;
            w_ptrs[k] = (co < C_out)
                ? (weight + (long long)c_in * weight_stride_cin + (long long)co * weight_stride_cout)
                : weight;
        }

        for (int kh = 0; kh < K_H; ++kh) {
            int h_in_scaled = h_out + pad_h - kh * dil_h;
            bool h_valid = (h_in_scaled >= 0) && (h_in_scaled % stride_h == 0);
            int h_in = h_valid ? (h_in_scaled / stride_h) : 0;
            bool row_valid = h_valid && (h_in < H_in);
            const float* row_ptr = row_valid ? (input_c_ptr + (long long)h_in * W_in) : nullptr;

            for (int kw = 0; kw < K_W; ++kw) {
                float in_val0 = 0.0f;
                float in_val1 = 0.0f;

                if (row_valid) {
                    int w_in_scaled0 = w_out_0 + pad_w - kw * dil_w;
                    int w_in_scaled1 = w_out_1 + pad_w - kw * dil_w;

                    if (valid_w0 && w_in_scaled0 >= 0 && (w_in_scaled0 % stride_w == 0)) {
                        int w_in0 = w_in_scaled0 / stride_w;
                        if (w_in0 < W_in) in_val0 = row_ptr[w_in0];
                    }
                    if (valid_w1 && w_in_scaled1 >= 0 && (w_in_scaled1 % stride_w == 0)) {
                        int w_in1 = w_in_scaled1 / stride_w;
                        if (w_in1 < W_in) in_val1 = row_ptr[w_in1];
                    }
                }

                // Weight pointers must advance exactly once per (kh,kw) step
                // regardless of validity, to stay in sync with the
                // contiguous K_H*K_W window for this (c_in, c_out) pair.
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

    // ----------------------------------------------------------------
    // 4. Write back
    // ----------------------------------------------------------------
    long long out_batch_offset = (long long)b_idx * C_out * H_out * W_out;
    long long out_channel_stride = (long long)H_out * W_out;

    auto write_back = [&](int w_curr, float* acc, bool w_valid) {
        if (!w_valid) return;
        long long out_spatial = (long long)h_out * W_out + w_curr;

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
// Launcher (name/signature unchanged -- see src/main.cpp). out_pad_h/w are
// accepted for signature compatibility but unused here, same as the
// previous kernel: output_padding only affects H_out/W_out, already baked
// into the H_out/W_out arguments below by custom_conv_transpose2d_forward.
// -------------------------------------------------------------------------
void launch_conv_transpose2d_fp32(const float* input, const float* weight, const float* bias, float* output,
    int B, int C_in, int H_in, int W_in, int C_out, int K_H, int K_W, int H_out, int W_out,
    int stride_h, int stride_w, int pad_h, int pad_w, int out_pad_h, int out_pad_w, int dil_h, int dil_w) {

    (void)out_pad_h;
    (void)out_pad_w;

    long long total = (long long)B * C_out * H_out * W_out;
    if (total <= 0) return;

    dim3 threads_per_block(16, 16);

    int blocks_w = DIV_CEIL(W_out, 32);
    int blocks_h = DIV_CEIL(H_out, 16);
    int grid_x = blocks_w * blocks_h;
    int grid_y = DIV_CEIL(C_out, CTILE);
    int grid_z = B;

    dim3 blocks(grid_x, grid_y, grid_z);

    conv_transpose2d_fp32_kernel_opt<<<blocks, threads_per_block>>>(
        input, weight, bias, output,
        B, C_in, H_in, W_in, C_out, K_H, K_W, H_out, W_out,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w
    );
}

// =================================================================================
// Autotune benchmark harness (opt-in, NOT part of the real build)
//
// Only compiled when -DCMPEXT3_AUTOTUNE_HARNESS is passed -- see
// autotune_kernel_tile() in ../../setup.py. Exercises the real
// launch_conv_transpose2d_fp32 on synthetic data at tools/bench.py's
// ConvTranspose2d shape, timed with cudaEvents. Never linked into cmpext3.
// =================================================================================
#ifdef CMPEXT3_AUTOTUNE_HARNESS
#include <cstdio>

int main() {
    const int B = 64, C_in = 64, H_in = 64, W_in = 64;
    const int C_out = 64, K_H = 3, K_W = 3;
    const int stride = 2, pad = 1, out_pad = 1, dil = 1;
    const int H_out = (H_in - 1) * stride - 2 * pad + dil * (K_H - 1) + out_pad + 1;
    const int W_out = (W_in - 1) * stride - 2 * pad + dil * (K_W - 1) + out_pad + 1;

    size_t in_elems  = (size_t)B * C_in * H_in * W_in;
    size_t w_elems   = (size_t)C_in * C_out * K_H * K_W;   // weight is [C_in, C_out, K_H, K_W]
    size_t out_elems = (size_t)B * C_out * H_out * W_out;

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
        launch_conv_transpose2d_fp32(d_in, d_w, d_bias, d_out,
            B, C_in, H_in, W_in, C_out, K_H, K_W, H_out, W_out,
            stride, stride, pad, pad, out_pad, out_pad, dil, dil);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        launch_conv_transpose2d_fp32(d_in, d_w, d_bias, d_out,
            B, C_in, H_in, W_in, C_out, K_H, K_W, H_out, W_out,
            stride, stride, pad, pad, out_pad, out_pad, dil, dil);
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
