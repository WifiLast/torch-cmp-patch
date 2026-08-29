#!/usr/bin/env python3
"""
Tests fp32_attention.cu (attention_fp32_strict_kernel, the naive
one-thread-per-query-row hand-tuned kernel -- see its own file header and
cmpext3/__init__.py's custom_attention_forward comment for why fp16
routes through a different, warp-cooperative kernel instead) against:

  1. stock PyTorch fp32 F.scaled_dot_product_attention (cuDNN/math backend)
  2. cmpext3.ops.attention (the hand-tuned fp32 kernel itself, raw call)
  3. an ONNX-based TensorRT engine for the same attention computation,
     using the same tools/trt_onnx_conv3d.py-style pipeline (export ->
     OnnxParser -> Builder), rebuilt here for attention specifically.

No mask, no dropout, no causal (matches fp32_attention.cu's own scope --
see cmpext3/__init__.py's _patched_sdpa gating). Built as three ONNX
layers (MatMul(Q,K^T) -> scale (baked into Q via mul before export,
avoiding a separate Mul node) -> Softmax -> MatMul(*, V)), same
decomposition src/trt/attention_trt_runtime.cpp's C++ Network-API path
already uses for the same reason (no fused-attention op in this
TensorRT/opset generation) and tools/export_all_onnx.py's own
AttentionModule already exports.

Usage:
    python tools/trt_onnx_attention_test.py                 # default shape
    python tools/trt_onnx_attention_test.py --seq 4096
    python tools/trt_onnx_attention_test.py --iters 30 --warmup 10
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent
TIMING_CACHE_DIR = REPO_DIR / "src" / "trt" / "timing_caches"


class AttentionModule(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v, scale=self.scale)


class Engine:
    """Same minimal TensorRT 10+/11 wrapper as trt_onnx_conv3d.py -- see
    that file's module docstring for why this targets the current
    tensor-name-based API rather than SD-WebUI-TensorRT's older one."""

    def __init__(self, engine_path: Path):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.tensors: dict[str, torch.Tensor] = {}

    def build(self, onnx_path: Path, timing_cache_path: Path | None, workspace_mb: int = 1024) -> None:
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(onnx_path.read_bytes()):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"Failed to parse {onnx_path}:\n{errors}")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))

        cache_bytes = timing_cache_path.read_bytes() if (timing_cache_path and timing_cache_path.is_file()) else b""
        timing_cache = config.create_timing_cache(cache_bytes)
        config.set_timing_cache(timing_cache, ignore_mismatch=False)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed")
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
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--rebuild-engine", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible.")

    dtype = torch.float32
    B, H, S, D = args.batch, args.heads, args.seq, args.dim
    scale = D ** -0.5

    out_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="cmpext3_trt_onnx_attn_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "fp32_attention.onnx"
    engine_path = out_dir / "fp32_attention.trt"
    timing_cache_path = TIMING_CACHE_DIR / "attention.cache"

    print(f"Shape: B={B} H={H} S={S} D={D} scale={scale:.6f} dtype=fp32")

    torch.manual_seed(0)
    q = torch.randn(B, H, S, D, dtype=dtype, device="cuda")
    k = torch.randn(B, H, S, D, dtype=dtype, device="cuda")
    v = torch.randn(B, H, S, D, dtype=dtype, device="cuda")

    model = AttentionModule(scale).eval()
    dummy = (q.cpu(), k.cpu(), v.cpu())
    with torch.inference_mode():
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["q", "k", "v"], output_names=["output"],
            opset_version=17, do_constant_folding=True, dynamo=False,
        )
    print(f"Exported ONNX -> {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")

    engine = Engine(engine_path)
    if engine_path.is_file() and not args.rebuild_engine:
        print(f"Reusing existing engine -> {engine_path}")
    else:
        print("Building TensorRT engine (tactic search)...")
        engine.build(onnx_path, timing_cache_path)
        print(f"Built engine -> {engine_path} ({engine_path.stat().st_size / (1024*1024):.1f} MB)")
    engine.load()
    engine.activate()
    engine.allocate_buffers()

    stream = torch.cuda.Stream()

    with torch.inference_mode():
        stock_out = F.scaled_dot_product_attention(q, k, v, scale=scale)
    torch.cuda.synchronize()

    try:
        import cmpext3
        handtuned_out = cmpext3.ops.attention(q, k, v, scale)
        torch.cuda.synchronize()
        handtuned_diff = (handtuned_out.float() - stock_out.float()).abs()
        handtuned_available = True
    except Exception as exc:
        handtuned_available = False
        handtuned_err = str(exc)

    with torch.cuda.stream(stream):
        trt_tensors = engine.infer({"q": q, "k": k, "v": v}, stream.cuda_stream)
    torch.cuda.synchronize()
    trt_out = trt_tensors["output"]
    trt_diff = (trt_out.float() - stock_out.float()).abs()

    print(f"\nTensorRT (ONNX-built) correctness vs stock: max_abs_diff={trt_diff.max().item():.6g}")
    if handtuned_available:
        print(f"fp32_attention.cu (hand-tuned) correctness vs stock: max_abs_diff={handtuned_diff.max().item():.6g}")
    else:
        print(f"fp32_attention.cu (hand-tuned): call failed -- {handtuned_err}")

    def run_stock():
        with torch.cuda.stream(stream), torch.inference_mode():
            F.scaled_dot_product_attention(q, k, v, scale=scale)

    def run_trt():
        with torch.cuda.stream(stream):
            engine.infer({"q": q, "k": k, "v": v}, stream.cuda_stream)

    stock_ms = timed_ms(run_stock, stream, args.warmup, args.iters)
    trt_ms = timed_ms(run_trt, stream, args.warmup, args.iters)

    print(f"\nstock PyTorch fp32 SDPA:     {stock_ms:.4f} ms/call")
    print(f"TensorRT (ONNX-built) fp32:  {trt_ms:.4f} ms/call  ({stock_ms/trt_ms:.2f}x vs stock)")

    if handtuned_available:
        def run_handtuned():
            with torch.cuda.stream(stream):
                cmpext3.ops.attention(q, k, v, scale)
        handtuned_ms = timed_ms(run_handtuned, stream, args.warmup, args.iters)
        print(f"fp32_attention.cu (hand-tuned): {handtuned_ms:.4f} ms/call  "
              f"({stock_ms/handtuned_ms:.2f}x vs stock, {trt_ms/handtuned_ms:.2f}x vs TRT)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
