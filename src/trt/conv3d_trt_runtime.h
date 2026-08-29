// =================================================================================
// conv3d_trt_runtime.h
//
// Optional runtime TensorRT execution path for conv3d, used in place of
// fp32_conv3d.cu / fp16_conv3d.cu when CMPEXT3_USE_TENSORRT=1 (see
// cmpext3/__init__.py and main.cpp's custom_conv3d_forward). Only compiled
// in when CMPEXT3_WITH_TENSORRT is defined (setup.py sets this only when a
// TensorRT SDK is found at build time -- see _find_tensorrt_sdk() there),
// so the base extension still builds fine without TensorRT installed.
//
// UNVERIFIED: written against the TensorRT 8.5+/9.x C++ API with no
// TensorRT SDK available anywhere in the environment this was written in,
// and targeting Linux (the stated deployment target) from a Windows
// sandbox that can't even compile-check the Linux build. This is the
// least-verified code in this project -- test it for real before trusting
// it, ideally against tests_hardware/test_conv3d_perf.py-style correctness
// checks (compare its output to the hand-tuned kernel's, not just "did it
// run").
// =================================================================================
#pragma once

#include <cuda_runtime.h>

// Attempts to run conv3d via a TensorRT engine, built fresh from the REAL
// weight/bias tensors on first encounter with a given (shape, weight
// content) combination and cached (in-process AND on-disk, see
// conv3d_trt_runtime.cpp's cache directory resolution) for reuse on every
// later call with the same shape+weights. The ONNX exports in src/onnx/
// used random dummy weights (real weights aren't known until this
// function is actually called) -- this is why the engine is built here,
// weight-aware, rather than loading a pre-built src/trt/*.trt file
// directly; tools/build_trt_engine.py's output is for offline
// benchmarking, not for this runtime path.
//
// input/weight/bias/output are CUDA DEVICE pointers (the same ones
// fp32_conv3d.cu/fp16_conv3d.cu's launch_conv3d_fp32/fp16 already take).
// `stream` should be the caller's current CUDA stream (main.cpp passes
// c10::cuda::getCurrentCUDAStream()) so this integrates with PyTorch's own
// stream ordering instead of racing it on a separate default stream.
//
// Returns true and leaves a complete, correct result in `output` on
// success. Returns false (leaving `output` UNTOUCHED -- the caller must
// not assume any partial writes happened) on ANY failure: TensorRT
// unavailable at runtime, unsupported dtype/shape, engine build failure,
// CUDA error, anything. Callers must always have a working fallback (the
// existing hand-tuned kernel) and treat `false` as "silently try that
// instead," matching every other opportunistic-fast-path pattern already
// used throughout this codebase (e.g. every _patched_* wrapper in
// cmpext3/__init__.py falling back to stock on RuntimeError).
bool cmpext3_trt_conv3d_forward(
    const void* input, const void* weight, const void* bias, void* output,
    int B, int C_in, int D_in, int H_in, int W_in,
    int C_out, int K_D, int K_H, int K_W,
    int D_out, int H_out, int W_out,
    int s_d, int s_h, int s_w,
    int p_d, int p_h, int p_w,
    int d_d, int d_h, int d_w,
    bool is_fp16,
    bool has_bias,
    cudaStream_t stream);
