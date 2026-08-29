#!/usr/bin/env python3
"""
ONNX -> TensorRT pipeline for conv3d, following the same shape as this
repo's example/Stable-Diffusion-WebUI-TensorRT extension (export to ONNX,
parse with TensorRT's ONNX parser, build an engine with an optimization
profile + timing cache, run it) -- see that project's exporter.py and
utilities.py's Engine class for the original.

DIFFERENT from src/trt/conv3d_trt_runtime.cpp (the runtime C++ path used
by CMPEXT3_USE_TENSORRT=1): that one builds the TensorRT network directly
via the C++ Network API (addConvolutionNd) with the REAL weights baked in
at first-call time, no ONNX involved at all. This script instead goes
through an actual ONNX file on disk, via TensorRT's OnnxParser -- the
"standard" path SD-WebUI-TensorRT itself uses, and the same graph
TensorRT's tactic search sees when built that way (which may or may not
pick different tactics than addConvolutionNd -- that's an open question
this script exists to actually answer, not assume).

Also DIFFERENT from tools/build_trt_engine.py (a thin trtexec wrapper --
still the simplest option if you just want a .trt file). This one uses
TensorRT's Python API directly (Builder/OnnxParser/BuilderConfig), like
SD-WebUI-TensorRT's utilities.py does -- except that file additionally
uses `polygraphy` as a convenience wrapper (not installed here), and its
Engine class targets an older binding-index API (num_bindings,
binding_is_input, context.set_binding_shape) that TensorRT 10+ removed
entirely in favor of the tensor-NAME-based API used below
(get_tensor_name, set_input_shape, set_tensor_address, execute_async_v3)
-- confirmed by inspecting this exact installed TensorRT 11.2.1 build's
Python API surface directly, the same way src/trt/*.cpp's C++-side
TensorRT-10+-breaking-changes were confirmed rather than assumed.

Also note: TensorRT 10+ networks are always "strongly typed" -- there is
no trt.BuilderFlag.FP16 anymore (confirmed absent from this build's
BuilderFlag enum, matching src/trt/*.cpp's C++-side finding). Precision
comes entirely from the ONNX file's own tensor dtypes, so "fp16" here
means exporting the model at fp16 in the first place (see export_onnx()),
not a builder flag.

Usage:
    python tools/trt_onnx_conv3d.py                  # fp16, real bench.py shape
    python tools/trt_onnx_conv3d.py --dtype fp32
    python tools/trt_onnx_conv3d.py --iters 30 --warmup 10
    python tools/trt_onnx_conv3d.py --keep-onnx --keep-engine   # inspect the artifacts after
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent
TIMING_CACHE_DIR = REPO_DIR / "src" / "trt" / "timing_caches"

# tools/bench.py's Conv3d section -- the same HunyuanVideo-style
# VAE-decode shape export_conv3d_onnx.py and build_trt_engine.py's
# KNOWN_INPUT_SHAPES already use.
B, C_IN, D_IN, H_IN, W_IN = 1, 512, 8, 32, 32
C_OUT, K, STRIDE, PADDING = 512, 3, 1, 1

DTYPES = {"fp16": torch.float16, "fp32": torch.float32}
TORCH_TO_NP = {torch.float16: np.float16, torch.float32: np.float32}


def build_model(dtype: torch.dtype) -> nn.Module:
    torch.manual_seed(0)
    conv = nn.Conv3d(C_IN, C_OUT, kernel_size=K, stride=STRIDE, padding=PADDING, bias=True)
    conv = conv.to(dtype=dtype).eval()
    return conv


def export_onnx(model: nn.Module, dtype: torch.dtype, onnx_path: Path, opset: int = 17) -> torch.Tensor:
    """Exports `model` to `onnx_path` and returns the exact dummy input
    used, so the caller can re-run the SAME input through the real module
    for a correctness check against the engine built from this file."""
    dummy_input = torch.randn(B, C_IN, D_IN, H_IN, W_IN, dtype=dtype)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,  # matches export_conv3d_onnx.py -- avoids the onnxscript dependency
        )
    return dummy_input


class Engine:
    """Minimal TensorRT 10+/11 Engine wrapper: build from ONNX, load,
    activate an execution context, allocate matching I/O tensors, run.
    Same role as SD-WebUI-TensorRT's utilities.py Engine class, rebuilt
    against the current tensor-name-based API (see module docstring)."""

    def __init__(self, engine_path: Path):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.tensors: dict[str, torch.Tensor] = {}

    def build(self, onnx_path: Path, timing_cache_path: Path | None = None, workspace_mb: int = 1024) -> None:
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        # No NetworkDefinitionCreationFlag needed -- explicit batch +
        # strongly typed are the only mode TensorRT 10+ supports (see
        # src/trt/*.cpp's identical createNetworkV2(0) fix).
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)

        onnx_bytes = onnx_path.read_bytes()
        if not parser.parse(onnx_bytes):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"Failed to parse {onnx_path}:\n{errors}")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))

        # Timing cache: persists tactic-timing measurements across builds
        # (of this or any other engine with matching layers) on this same
        # GPU, so a later rebuild of a similar network skips re-timing
        # tactics it already has data for. Same role as SD-WebUI-
        # TensorRT's timing_caches/ directory.
        cache_bytes = b""
        if timing_cache_path is not None and timing_cache_path.is_file():
            cache_bytes = timing_cache_path.read_bytes()
        timing_cache = config.create_timing_cache(cache_bytes)
        config.set_timing_cache(timing_cache, ignore_mismatch=False)

        # This model's only input is fully static (no dynamic_axes in
        # export_onnx above), so no IOptimizationProfile is needed --
        # TensorRT infers the single fixed shape straight from the ONNX
        # graph. (tools/export_all_onnx.py's dynamic-axes variant would
        # need one; see that script + build_trt_engine.py's
        # KNOWN_INPUT_SHAPES for the min/opt/max values that would go
        # into it.)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed (see logger output above)")
        self.engine_path.write_bytes(bytes(serialized))

        if timing_cache_path is not None:
            updated = config.get_timing_cache()
            if updated is not None:
                timing_cache_path.parent.mkdir(parents=True, exist_ok=True)
                timing_cache_path.write_bytes(bytes(updated.serialize()))

    def load(self) -> None:
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine from {self.engine_path}")

    def activate(self) -> None:
        self.context = self.engine.create_execution_context()

    def allocate_buffers(self, device: str = "cuda") -> None:
        import tensorrt as trt

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.context.get_tensor_shape(name))
            np_dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            torch_dtype = {np.float16: torch.float16, np.float32: torch.float32}[np_dtype]
            self.tensors[name] = torch.empty(shape, dtype=torch_dtype, device=device)

    def infer(self, feed_dict: dict[str, torch.Tensor], stream: int) -> dict[str, torch.Tensor]:
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)
        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        if not self.context.execute_async_v3(stream):
            raise RuntimeError("TensorRT inference failed")
        return self.tensors


def timed_ms(fn, stream: torch.cuda.Stream, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    for _ in range(iters):
        fn()
    end.record(stream)
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dtype", choices=list(DTYPES), default="fp16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--keep-onnx", action="store_true", help="Keep the exported .onnx file instead of deleting it")
    parser.add_argument("--keep-engine", action="store_true", help="Keep the built .trt engine instead of deleting it")
    parser.add_argument("--rebuild-engine", action="store_true",
                         help="Rebuild the TensorRT engine even if one already exists at the target path "
                              "(conv3d's tactic search can take several minutes; reused by default with --out-dir)")
    parser.add_argument("--out-dir", default=None, help="Directory for .onnx/.trt artifacts (default: a temp dir)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible -- nothing to build/benchmark.")

    dtype = DTYPES[args.dtype]
    out_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="cmpext3_trt_onnx_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{args.dtype}_conv3d.onnx"
    engine_path = out_dir / f"{args.dtype}_conv3d.trt"
    timing_cache_path = TIMING_CACHE_DIR / "conv3d.cache"

    print(f"Building {args.dtype} conv3d: input=({B},{C_IN},{D_IN},{H_IN},{W_IN}) "
          f"kernel={K} stride={STRIDE} padding={PADDING} C_out={C_OUT}")

    model = build_model(dtype).cuda()
    dummy_input = export_onnx(build_model(dtype), dtype, onnx_path)  # CPU copy for export, same seed
    print(f"Exported ONNX -> {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")

    engine = Engine(engine_path)
    if engine_path.is_file() and not args.rebuild_engine:
        print(f"Reusing existing engine -> {engine_path} (pass --rebuild-engine to force a fresh build)")
    else:
        print("Building TensorRT engine (tactic search -- can take several minutes for conv3d)...")
        engine.build(onnx_path, timing_cache_path=timing_cache_path)
        print(f"Built engine -> {engine_path} ({engine_path.stat().st_size / (1024*1024):.1f} MB)")
    engine.load()
    engine.activate()
    engine.allocate_buffers()

    stream = torch.cuda.Stream()
    x = dummy_input.to(device="cuda", dtype=dtype)

    with torch.inference_mode():
        stock_out = model(x)
    torch.cuda.synchronize()

    with torch.cuda.stream(stream):
        trt_tensors = engine.infer({"input": x}, stream.cuda_stream)
    torch.cuda.synchronize()
    trt_out = trt_tensors["output"]

    diff = (trt_out.float() - stock_out.float()).abs()
    rel = (diff.max() / stock_out.float().abs().max().clamp_min(1e-6)).item()
    print(f"\nCorrectness vs stock PyTorch conv3d: max_abs_diff={diff.max().item():.6g}  rel={rel:.6g}")

    def run_stock():
        # Must run on the SAME `stream` the timing events below are
        # recorded on -- otherwise the real cuDNN work happens on PyTorch's
        # default stream while start/end.record(stream) bracket a
        # different, empty stream, measuring near-zero elapsed time
        # instead of the actual kernel duration (caught by testing: this
        # produced an implausible 0.07ms "stock" number before the fix).
        with torch.cuda.stream(stream), torch.inference_mode():
            model(x)

    def run_trt():
        with torch.cuda.stream(stream):
            engine.infer({"input": x}, stream.cuda_stream)

    stock_ms = timed_ms(run_stock, stream, args.warmup, args.iters)
    trt_ms = timed_ms(run_trt, stream, args.warmup, args.iters)

    print(f"\nstock PyTorch (cuDNN): {stock_ms:.4f} ms/call")
    print(f"TensorRT (ONNX-built): {trt_ms:.4f} ms/call")
    print(f"ratio (stock/TRT):     {stock_ms / trt_ms:.2f}x  "
          f"({'TensorRT faster' if trt_ms < stock_ms else 'stock cuDNN faster'})")

    if not args.keep_onnx:
        onnx_path.unlink(missing_ok=True)
    if not args.keep_engine:
        engine_path.unlink(missing_ok=True)
    if not args.keep_onnx and not args.keep_engine:
        try:
            out_dir.rmdir()
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
