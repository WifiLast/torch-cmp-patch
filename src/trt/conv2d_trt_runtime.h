// =================================================================================
// conv2d_trt_runtime.h -- see conv3d_trt_runtime.h / trt_common.h for the
// full design rationale (build-on-first-use, content-hash cache,
// UNVERIFIED). Same contract, 2D instead of 3D.
// =================================================================================
#pragma once

#include <cuda_runtime.h>

bool cmpext3_trt_conv2d_forward(
    const void* input, const void* weight, const void* bias, void* output,
    int B, int C_in, int H_in, int W_in,
    int C_out, int K_H, int K_W,
    int H_out, int W_out,
    int s_h, int s_w,
    int p_h, int p_w,
    int d_h, int d_w,
    bool is_fp16,
    bool has_bias,
    cudaStream_t stream);
