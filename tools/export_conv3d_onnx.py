#!/usr/bin/env python3
"""
Exports a standalone ONNX model containing a single Conv3d layer, matching
fp32_conv3d.cu's representative shape (tools/bench.py's Conv3d section --
a HunyuanVideo-style VAE-decode shape) at FP32 precision.

Why this instead of hand-written TensorRT C++ API code
(src/cuda/fp32_conv3d_trt_harness.cpp uses addConvolutionNd directly and
could NOT be compile-checked anywhere in this environment -- no TensorRT
SDK installed on this machine at all): this script only needs plain
PyTorch, CPU-only is fine (nn.Conv3d and ONNX export don't touch CUDA), so
it CAN be verified here. It also produces the standard "ONNX -> TensorRT
builder" path the real Stable-Diffusion-WebUI-TensorRT extension in this
repo actually uses (see source/other/Stable-Diffusion-WebUI-TensorRT/
exporter.py's export_onnx, same opset default), rather than hand-built
network-API C++ that has to be trusted without ever compiling.

Once you have TensorRT installed, benchmark the exported model directly
with trtexec -- ships with every TensorRT SDK install, no custom code,
no compiling against NvInfer.h at all:

    python tools/export_conv3d_onnx.py
    trtexec --onnx=fp32_conv3d.onnx --fp32 --iterations=20 --avgRuns=20

trtexec's own output includes a "GPU Compute Time" mean, directly
comparable to fp32_conv3d.cu's own harness's RESULT_MS (same shape, same
FP32 precision, same warmup/steady-state-timing intent) -- see
cmpext3/__init__.py's enable() comment and fp32_conv3d.cu's header comment
for why FFMA/Tensor-Core avoidance is the whole point of the hand-tuned
kernel this is meant to compare against, and why a straightforward FP32
TensorRT engine isn't guaranteed to share that property (TensorRT has no
--fmad=false equivalent).
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn

# tools/bench.py's Conv3d section: HunyuanVideo-style VAE-decode shape.
B, C_IN, D_IN, H_IN, W_IN = 1, 512, 8, 32, 32
C_OUT, K, STRIDE, PADDING = 512, 3, 1, 1


def build_model() -> nn.Module:
    conv = nn.Conv3d(C_IN, C_OUT, kernel_size=K, stride=STRIDE, padding=PADDING, bias=True)
    conv.eval()
    return conv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="fp32_conv3d.onnx", help="Output .onnx path")
    parser.add_argument("--opset", type=int, default=17,
                         help="ONNX opset -- 17 matches this repo's Stable-Diffusion-WebUI-TensorRT "
                              "extension's own export default (see exporter.py's export_onnx)")
    args = parser.parse_args()

    model = build_model()
    dummy_input = torch.randn(B, C_IN, D_IN, H_IN, W_IN, dtype=torch.float32)

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_input,
            args.out,
            input_names=["input"],
            output_names=["output"],
            opset_version=args.opset,
            do_constant_folding=True,
            # Forces the legacy TorchScript-based exporter. torch>=2.5's
            # default (dynamo=True) needs the separate onnxscript package;
            # the legacy path needs only PyTorch itself and matches what
            # this repo's Stable-Diffusion-WebUI-TensorRT extension already
            # relies on (see exporter.py) -- no reason to add a new
            # dependency just for a single-node export like this one.
            dynamo=False,
        )

    print(f"Wrote {args.out}")
    print(f"Shape: input=({B},{C_IN},{D_IN},{H_IN},{W_IN}) "
          f"kernel={K} stride={STRIDE} padding={PADDING} C_out={C_OUT}, dtype=fp32")
    print()
    print("Benchmark with trtexec once TensorRT is installed -- no custom code needed:")
    print(f"    trtexec --onnx={args.out} --fp32 --iterations=20 --avgRuns=20")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
