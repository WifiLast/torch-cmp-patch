// =================================================================================
// matmul_trt_runtime.h -- see trt_common.h for the full design rationale
// (build-on-first-use, content-hash cache, UNVERIFIED).
//
// Scope matches fp16_matmul.cu/fp32_matmul.cu's launch_matmul_fp16/fp32
// exactly: plain input[M,K] @ weight[K,N] -> output[M,N], NO bias --
// custom_linear_forward calls launch_add_bias_fp16/fp32 as a SEPARATE
// kernel afterward, unchanged by this. `weight` here is expected already
// transposed to [K,N] layout, matching what custom_linear_forward already
// passes to launch_matmul_fp16/fp32 (see main.cpp's weight_t = weight.t().
// contiguous() before its own dispatch).
// =================================================================================
#pragma once

#include <cuda_runtime.h>

bool cmpext3_trt_matmul_forward(
    const void* input, const void* weight, void* output,
    int M, int N, int K,
    bool is_fp16,
    cudaStream_t stream);
