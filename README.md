# cmpext3 - PyTorch/ComfyUI CUDA Extension for CMP Mining Cards

A PyTorch/ComfyUI extension that bypasses FFMA/Tensor core throttling on CMP mining cards based on Turing chips (TU10X). Forked from [eastmoe/cmp_ext](https://github.com/eastmoe/cmp_ext), which originally targeted the Ampere-based CMP 170HX.

Tested on a CMP 50HX card with SDXL and Anima text-to-image models, at least tripling throughput compared to normal FP16 workloads.

## What it does

- Monkeypatches core `torch`/`F` ops (conv2d, conv3d, linear/matmul, attention, normalization, activations, and more) with hand-tuned CUDA kernels that avoid the FMA throttle mining cards impose.
- Supports FP16, FP32, and BF16. Turing has no native BF16 arithmetic, so BF16 tensors are converted at the kernel boundary and routed through the existing FMA-free FP32 or FP16 kernels.
- Automatically selects, per operation and shape, whichever implementation (custom kernel or stock cuDNN/cuBLAS) is actually faster and numerically correct, measured at runtime rather than assumed in advance.
- Uses a Winograd-based fp32 conv3d kernel where the shape allows it, chosen automatically against the direct kernel.
- Autotunes kernel tile parameters against the installed GPU at build time, with results cached for subsequent installs.

See `Projekt_Description.md` for full technical details, benchmarks, and configuration options.

## Requirements

- A CUDA-capable Turing GPU (developed and tested on CMP 50HX, TU116, compute capability 7.5)
- CUDA toolkit and `nvcc`
- PyTorch

## Installation

A pre-compiled wheel for CUDA 12.8 + Python 3.12 is available on the Releases page.

To build from source:

```
pip install -e . --no-build-isolation
```

The first install runs a one-time kernel autotuning sweep (roughly 1-5 minutes); results are cached in `.cmpext3_autotune_cache.json` for future installs.

## Status

This project is vibecoded (using Claude.ai) to get PyTorch running efficiently on Turing-based mining cards. It may contain errors and can affect output correctness; validate results for your own workloads.

## License

MIT
