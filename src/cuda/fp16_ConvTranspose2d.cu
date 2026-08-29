#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define DIV_CEIL(a, b) (((a) + (b) - 1) / (b))

// CTILE = output channels processed per thread (must be even -- channels
// are processed in half2 pairs). Independently tunable via
// -DCONVT2D_FP16_CTILE=.. (see ../../setup.py's autotune_kernel_tile()) --
// prefixed per-file so it can't collide with the other kernels' own CTILE
// overrides.
#ifndef CONVT2D_FP16_CTILE
#define CONVT2D_FP16_CTILE 8
#endif
#define CTILE CONVT2D_FP16_CTILE
#define CTILE_HALF2 (CTILE / 2)

// =================================================================================
// Register-blocked ConvTranspose2d -- FP16
//
// Replaces the previous conv_transpose2d_fp16_ga100_opt_kernel (2 output
// channels per thread, scalar __hfma, single output pixel, no pixel reuse)
// with the same 2-pixel x CTILE-channel thread-coarsening scheme as
// fp32_ConvTranspose2d.cu, upgraded to half2-packed __hfma2 arithmetic:
// channels are processed in pairs (CTILE_HALF2 half2 accumulator slots per pixel),
// each pair's two scalar half weight loads packed into one half2 via
// __halves2half2 (same trick fp16_conv.cu's conv2d_fp16_turing uses for its
// smem-resident fragments) and multiplied against the input value broadcast
// to both lanes via __half2half2.
//
// Weight layout is [C_in, C_out, K_H, K_W] (C_in outermost, same as the
// fp32 kernel) -- see fp32_ConvTranspose2d.cu's header comment for why the
// CTILE per-channel weight pointers are recomputed once per c_in (one multiply
// each) rather than walked with ++ across the whole C_in loop, then walked
// contiguously through the kh/kw window for that c_in. A genuine
// shared-memory-tiled GEMM kernel (like conv2d_fp16_turing/conv3d_fp16_turing)
// isn't a direct fit here without first transposing the weight to a
// [C_out, K] layout -- an extra pass this kernel avoids.
// =================================================================================

__global__ void __launch_bounds__(256) conv_transpose2d_fp16_kernel_opt(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int B, int C_in, int H_in, int W_in,
    int C_out, int K_H, int K_W,
    int H_out, int W_out,
    int stride_h, int stride_w, int pad_h, int pad_w, int dil_h, int dil_w
) {
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

    // CTILE_HALF2 half2 slots per pixel -> CTILE output channels (c_out_base+2j, +2j+1).
    half2 accum0[CTILE_HALF2];
    half2 accum1[CTILE_HALF2];
    #pragma unroll
    for (int j = 0; j < CTILE_HALF2; ++j) {
        accum0[j] = __float2half2_rn(0.0f);
        accum1[j] = __float2half2_rn(0.0f);
    }

    long long input_batch_offset = (long long)b_idx * C_in * H_in * W_in;
    const half* input_base_ptr = input + input_batch_offset;

    long long weight_stride_cin  = (long long)C_out * K_H * K_W;   // Weight [C_in, C_out, K_H, K_W]
    long long weight_stride_cout = (long long)K_H * K_W;

    const half* w_ptrs[CTILE];

    for (int c_in = 0; c_in < C_in; ++c_in) {
        const half* input_c_ptr = input_base_ptr + (long long)c_in * H_in * W_in;

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
            const half* row_ptr = row_valid ? (input_c_ptr + (long long)h_in * W_in) : nullptr;

            for (int kw = 0; kw < K_W; ++kw) {
                half in_val0 = __float2half(0.0f);
                half in_val1 = __float2half(0.0f);

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

                half2 va0 = __half2half2(in_val0);
                half2 va1 = __half2half2(in_val1);

                // Weight pointers must advance exactly once per (kh,kw) step
                // regardless of validity, to stay in sync with the
                // contiguous K_H*K_W window for this (c_in, c_out) pair.
                #pragma unroll
                for (int j = 0; j < CTILE_HALF2; ++j) {
                    half wa = *w_ptrs[2 * j];     w_ptrs[2 * j]++;
                    half wb = *w_ptrs[2 * j + 1]; w_ptrs[2 * j + 1]++;
                    half2 wv = __halves2half2(wa, wb);

                    accum0[j] = __hfma2(va0, wv, accum0[j]);
                    accum1[j] = __hfma2(va1, wv, accum1[j]);
                }
            }
        }
    }

    // ----------------------------------------------------------------
    // Write back
    // ----------------------------------------------------------------
    long long out_batch_offset = (long long)b_idx * C_out * H_out * W_out;
    long long out_channel_stride = (long long)H_out * W_out;

    auto write_back = [&](int w_curr, half2* acc, bool w_valid) {
        if (!w_valid) return;
        long long out_spatial = (long long)h_out * W_out + w_curr;

        #pragma unroll
        for (int j = 0; j < CTILE_HALF2; ++j) {
            int co_a = c_out_base + 2 * j;
            int co_b = co_a + 1;

            half va = acc[j].x;
            half vb = acc[j].y;

            if (co_a < C_out) {
                if (bias) va = __hadd(va, bias[co_a]);
                output[out_batch_offset + (long long)co_a * out_channel_stride + out_spatial] = va;
            }
            if (co_b < C_out) {
                if (bias) vb = __hadd(vb, bias[co_b]);
                output[out_batch_offset + (long long)co_b * out_channel_stride + out_spatial] = vb;
            }
        }
    };

    write_back(w_out_0, accum0, valid_w0);
    write_back(w_out_1, accum1, valid_w1);
}

// -------------------------------------------------------------------------
// Launcher (name/signature unchanged -- see src/main.cpp). out_pad_h/w are
// accepted for signature compatibility but unused, same as before:
// output_padding only affects H_out/W_out, already baked into those
// arguments by custom_conv_transpose2d_forward.
// -------------------------------------------------------------------------
void launch_conv_transpose2d_fp16(const void* input, const void* weight, const void* bias, void* output,
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

    conv_transpose2d_fp16_kernel_opt<<<blocks, threads_per_block>>>(
        (const half*)input, (const half*)weight, (const half*)bias, (half*)output,
        B, C_in, H_in, W_in, C_out, K_H, K_W, H_out, W_out,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w
    );
}

// =================================================================================
// Autotune benchmark harness (opt-in, NOT part of the real build)
//
// Only compiled when -DCMPEXT3_AUTOTUNE_HARNESS is passed -- see
// autotune_kernel_tile() in ../../setup.py. Exercises the real
// launch_conv_transpose2d_fp16 on synthetic data at tools/bench.py's
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

    half *d_in, *d_w, *d_bias, *d_out;
    if (cudaMalloc(&d_in, in_elems * sizeof(half)) != cudaSuccess ||
        cudaMalloc(&d_w, w_elems * sizeof(half)) != cudaSuccess ||
        cudaMalloc(&d_bias, (size_t)C_out * sizeof(half)) != cudaSuccess ||
        cudaMalloc(&d_out, out_elems * sizeof(half)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }

    cudaMemset(d_in, 0x3C, in_elems * sizeof(half));   // 0x3C3C == 1.0h
    cudaMemset(d_w, 0x3C, w_elems * sizeof(half));
    cudaMemset(d_bias, 0, (size_t)C_out * sizeof(half));

    const int warmup = 5, iters = 20;
    for (int i = 0; i < warmup; ++i) {
        launch_conv_transpose2d_fp16(d_in, d_w, d_bias, d_out,
            B, C_in, H_in, W_in, C_out, K_H, K_W, H_out, W_out,
            stride, stride, pad, pad, out_pad, out_pad, dil, dil);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        launch_conv_transpose2d_fp16(d_in, d_w, d_bias, d_out,
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
