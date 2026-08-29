// =================================================================================
// attention_trt_runtime.h -- see trt_common.h for the shared design
// rationale (build-on-first-use, cache-forever, UNVERIFIED). Different
// from conv3d/conv2d/matmul in one respect: Q/K/V are genuine per-call
// runtime inputs here, not fixed weights, so the cache key is purely
// shape+scale+dtype -- no device->host copy of Q/K/V needed to compute it
// (unlike the weight-bearing ops, where the cache key has to hash weight
// CONTENT for correctness). This also means the SAME cached engine is
// reused across every call at a given (B,H,S,D,scale), regardless of the
// actual Q/K/V values -- which is correct, since those are exactly the
// engine's runtime bindings, not baked-in constants.
//
// Scope matches fp16_attention.cu/fp32_attention.cu's launch_attention_
// fp16/fp32 exactly: plain scaled dot-product attention, no mask, no
// causal, no dropout -- custom_attention_forward already only reaches this
// far after falling back to stock for anything else (see main.cpp).
//
// Built as three separate layers (MatMul(Q,K^T) -> scale -> Softmax ->
// MatMul(*, V)), since the TensorRT 8.5+/9.x API generation this targets
// has no single fused Attention op (that arrived in a much later opset/
// TensorRT version) -- this is the same decomposition the legacy ONNX
// exporter itself produces for F.scaled_dot_product_attention (see
// tools/export_all_onnx.py's AttentionModule), just built directly via
// TensorRT's C++ network API instead of round-tripping through ONNX.
// =================================================================================
#pragma once

#include <cuda_runtime.h>

bool cmpext3_trt_attention_forward(
    const void* q, const void* k, const void* v, void* output,
    int B, int H, int S, int D,
    float scale,
    bool is_fp16,
    cudaStream_t stream);
