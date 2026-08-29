"""
Runtime ONNX-based TensorRT path for conv3d -- a Python-level companion to
src/trt/conv3d_trt_runtime.cpp's C++ Network-API path (addConvolutionNd),
built instead through the same ONNX-export -> OnnxParser -> Builder route
tools/trt_onnx_conv3d.py already validated offline: measured ~8% faster
than stock cuDNN for the real bench.py HunyuanVideo VAE-decode shape --
notably unlike the C++ Network-API path, which lost to the hand-tuned
kernel by 5-15x in earlier testing (see that script's own module
docstring for the full comparison and how the timing was verified).

WHY PYTHON, NOT C++: getting the REAL runtime weight/bias tensors into an
ONNX graph means building an actual ONNX protobuf model. Doing that from
C++ needs libprotobuf-dev + the onnx .proto sources compiled in -- this
machine (checked directly: no protoc, no libprotobuf-dev) has neither,
and installing them is a real new system dependency for the whole
package, not a small addition. `torch.onnx.export` and the `onnx` pip
package already do this correctly from Python with nothing extra needed,
so that's where this lives. C++ is not involved at any point here.

WHY A REFIT-BASED ENGINE, NOT ONE-ENGINE-PER-WEIGHT-TENSOR: an engine
build for a real conv3d shape measured ~8 MINUTES of wall-clock time in
testing (mostly TensorRT's own CPU-side tactic search, not GPU work --
see tools/trt_onnx_conv3d.py's comments). A cache keyed by weight CONTENT
(the first version of this file) would pay that cost again for every
distinct checkpoint/fine-tune/LoRA sharing the same conv3d shape --
unnecessary, and this is exactly the problem
example/Stable-Diffusion-WebUI-TensorRT's own REFIT support
(BuilderFlag.REFIT + trt.Refitter, see its exporter.py's
get_refit_weights/export_lora and utilities.py's Engine.refit_from_dict)
solves for the same reason: their LoRA workflow needs the SAME base UNet
engine to serve many different weight sets without rebuilding. Adopted
here the same way: the engine is built ONCE per STRUCTURE (weight shape +
input shape + dtype + stride/padding/dilation -- content-independent),
with BuilderFlag.REFIT set, and real weight VALUES get swapped in via
trt.Refitter at call time -- measured at ~0.3ms per refit in testing
(Refitter construction + set_named_weights + refit_cuda_engine), vs
minutes for a full rebuild. One structure-keyed engine therefore now
transparently serves every conv3d layer that happens to share that exact
shape/config, not just one specific weight tensor.

WHY STILL NO ON-THE-FLY *BUILDS* AT CALL TIME: the ~8-minute cost above
is for the initial structure build, not refitting -- that part hasn't
changed. Unlike src/trt/conv3d_trt_runtime.cpp's C++ path (which DOES
build on first use -- a real design tradeoff explained in trt_common.h,
appropriate there because addConvolutionNd builds are much faster), an
8-minute stall on a random inference call would be a severe, unacceptable
regression. So the split is still:

  - prebuild(...): explicit, offline, SLOW the first time for a given
    shape/config (minutes), near-instant every time after (the on-disk
    engine already exists, this just confirms it). Call this ahead of
    time -- once per distinct (weight shape, input shape,
    stride/padding/dilation, dtype) you expect to hit -- never from
    inside a model's forward pass.
  - try_forward(...): the actual runtime entry point _patched_conv3d
    calls. FAST -- returns None immediately (no build attempt) if
    prebuild() was never run for this exact structure. When a matching
    engine exists, refits it with the CURRENT call's real weight/bias
    (skipped entirely if unchanged since the last call -- see
    _Engine.refit's identity check) and benchmarks once against
    `native_fn`, same "measure once per structure, trust the cache after
    that" design as trt_faster_than_fallback (src/trt/trt_common.cpp)
    uses on the C++ side -- valid here too since refit is cheap enough
    not to skew a steady-state timing comparison, and performance
    doesn't depend on weight values anyway, only structure.

Cache location: the SAME $XDG_CACHE_HOME/cmpext3/trt_engines (or
$HOME/.cache/cmpext3/trt_engines) directory src/trt/trt_common.cpp's
cache_dir() uses -- distinct filename prefix (conv3d_onnx_ vs conv3d_) so
the two engine caches share a directory without ever colliding.

Opt-in via CMPEXT3_USE_TENSORRT_ONNX=1 (separate from CMPEXT3_USE_TENSORRT,
which only controls the C++-side Network-API path) -- and conv3d
patching itself stays off by default regardless
(CMPEXT3_ENABLE_UNVERIFIED_KERNELS=1), so this only ever engages when
BOTH are explicitly opted into.
"""
from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

_engines: dict[str, "_Engine"] = {}
_decisions: dict[str, bool] = {}
_lock = threading.Lock()

_NP_TO_TORCH = {np.float16: torch.float16, np.float32: torch.float32}
_TORCH_TO_TRT_NAME = {torch.float16: "HALF", torch.float32: "FLOAT"}


def _cache_dir() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "cmpext3" / "trt_engines"


def _structure_key(weight_shape, has_bias: bool, dtype: torch.dtype,
                    input_shape, stride, padding, dilation) -> str:
    """Identifies the ENGINE STRUCTURE only -- deliberately no weight
    content. With REFIT, correctness no longer depends on baking in one
    specific weight tensor's values at build time (see module docstring),
    so a shape/config match is sufficient to reuse an engine, the same
    way trt_faster_than_fallback's C++-side benchmark-decision key
    already excludes weight content for the same underlying reason
    (speed and, now, correctness both depend only on structure)."""
    h = hashlib.sha256()
    h.update(repr((tuple(weight_shape), has_bias, str(dtype), tuple(input_shape),
                    stride, padding, dilation)).encode())
    return "conv3d_onnx_" + h.hexdigest()[:24]


def _engine_path(key: str) -> Path:
    return _cache_dir() / f"{key}.trt"


class _Engine:
    """TensorRT 10+/11 REFIT-capable engine wrapper -- load a
    pre-serialized (structure-only) engine, activate an execution
    context, allocate matching I/O tensors, refit real weight values in
    before each use, run. Trimmed runtime counterpart of
    tools/trt_onnx_conv3d.py's fuller Engine class (that one also builds
    non-refittable engines for benchmarking; see that file for why this
    targets TensorRT's tensor-name-based API, not SD-WebUI-TensorRT's
    older binding-index one)."""

    def __init__(self):
        self.engine = None
        self.context = None
        self.tensors: dict[str, torch.Tensor] = {}
        self._last_refit_id: Optional[tuple] = None

    def load(self, engine_bytes: bytes) -> None:
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError("failed to deserialize TensorRT engine")
        self.context = self.engine.create_execution_context()
        self._allocate_buffers()

    def _allocate_buffers(self) -> None:
        # OUTPUT only -- a persistent buffer TensorRT's kernel writes into
        # directly (no torch-level op touches it, so it's exempt from the
        # inference-tensor version-bump machinery below). INPUT deliberately
        # gets no buffer here; see infer()'s comment for why.
        import tensorrt as trt
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if name == "input":
                continue
            shape = tuple(self.context.get_tensor_shape(name))
            np_dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            self.tensors[name] = torch.empty(shape, dtype=_NP_TO_TORCH[np_dtype], device="cuda")

    def refit(self, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> None:
        """Swaps in real weight/bias VALUES via TensorRT's Refitter --
        cheap (~0.3ms measured) compared to the minutes a full rebuild
        would cost. Skipped entirely if this exact weight/bias tensor
        pair was the last thing refit into this engine (the common case:
        the same nn.Module gets called repeatedly across a forward pass
        with the same underlying parameter tensors) -- identity check via
        (data_ptr, autograd version), not full content comparison, same
        pragmatic tradeoff this codebase already makes elsewhere (e.g.
        trt_common.cpp's engine-cache content hash: cheap enough to be
        right in every case that matters, not formally bulletproof
        against e.g. a freed pointer being reused by an unrelated
        same-shape tensor -- astronomically unlikely in practice)."""
        import tensorrt as trt

        ident = (weight.data_ptr(), int(weight._version),
                 bias.data_ptr() if bias is not None else None,
                 int(bias._version) if bias is not None else None)
        if ident == self._last_refit_id:
            return

        refitter = trt.Refitter(self.engine, trt.Logger(trt.Logger.WARNING))
        all_names = set(refitter.get_all_weights())

        w_np = weight.detach().contiguous().cpu().numpy()
        trt_dtype = getattr(trt.DataType, _TORCH_TO_TRT_NAME[weight.dtype])
        if "weight" not in all_names or not refitter.set_named_weights(
                "weight", trt.Weights(trt_dtype, w_np.ctypes.data, w_np.size)):
            raise RuntimeError("refit: failed to set 'weight'")

        b_np = None  # keep alive until refit_cuda_engine() returns
        if bias is not None:
            b_np = bias.detach().contiguous().cpu().numpy()
            if "bias" not in all_names or not refitter.set_named_weights(
                    "bias", trt.Weights(trt_dtype, b_np.ctypes.data, b_np.size)):
                raise RuntimeError("refit: failed to set 'bias'")

        missing = refitter.get_missing_weights()
        if missing:
            raise RuntimeError(f"refit: missing weights {missing}")
        if not refitter.refit_cuda_engine():
            raise RuntimeError("refit_cuda_engine() failed")

        self._last_refit_id = ident

    def infer(self, input_tensor: torch.Tensor, stream: int) -> torch.Tensor:
        # Points TensorRT directly at the CALLER's tensor instead of
        # copying into a persistent buffer first -- besides being a free
        # copy to skip, a persistent input buffer allocated once (e.g.
        # inside a torch.inference_mode() block, as _patched_conv3d's
        # caller commonly is) becomes an "inference tensor" forever;
        # .copy_()-ing into it from a later call made OUTSIDE inference
        # mode then raises ("Inplace update to inference tensor outside
        # InferenceMode is not allowed") -- confirmed directly by hitting
        # exactly this in a benchmark script that (correctly) exercised
        # both call patterns. input_tensor must be contiguous and already
        # matches the engine's expected shape/dtype (guaranteed by
        # _structure_key matching); the caller keeps it alive for the
        # duration of this call, which execute_async_v3 needs anyway.
        input_tensor = input_tensor.contiguous()
        self.context.set_tensor_address("input", input_tensor.data_ptr())
        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        if not self.context.execute_async_v3(stream):
            raise RuntimeError("TensorRT inference failed")
        return self.tensors["output"]


def _build_engine_bytes(weight_shape, has_bias: bool, dtype: torch.dtype,
                         input_shape, stride, padding, dilation) -> bytes:
    """SLOW (minutes) -- ONNX export (placeholder zero weights; REFIT
    swaps in the real values later, so what's baked in at build time
    doesn't matter) + TensorRT build with BuilderFlag.REFIT. Only ever
    called from prebuild(), never from try_forward()."""
    import tempfile
    import tensorrt as trt

    C_out, C_in, K_D, K_H, K_W = weight_shape
    conv = nn.Conv3d(C_in, C_out, kernel_size=(K_D, K_H, K_W), stride=stride,
                      padding=padding, dilation=dilation, bias=has_bias)
    conv = conv.to(dtype=dtype).eval()  # placeholder weights (whatever nn.Conv3d's default init gives)

    dummy_input = torch.zeros(input_shape, dtype=dtype)
    with tempfile.TemporaryDirectory(prefix="cmpext3_trt_onnx_build_") as tmpdir:
        onnx_path = Path(tmpdir) / "conv3d.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                conv, dummy_input, str(onnx_path),
                input_names=["input"], output_names=["output"],
                opset_version=17, do_constant_folding=True, dynamo=False,
            )
        onnx_bytes = onnx_path.read_bytes()

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # explicit batch + strongly typed: the only mode TRT10+ has
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_bytes):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed: {errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.set_flag(trt.BuilderFlag.REFIT)  # <-- the whole point of this rewrite; see module docstring

    timing_cache_path = _cache_dir() / "onnx_timing_cache.bin"
    cache_bytes = timing_cache_path.read_bytes() if timing_cache_path.is_file() else b""
    timing_cache = config.create_timing_cache(cache_bytes)
    config.set_timing_cache(timing_cache, ignore_mismatch=False)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")

    updated = config.get_timing_cache()
    if updated is not None:
        _cache_dir().mkdir(parents=True, exist_ok=True)
        timing_cache_path.write_bytes(bytes(updated.serialize()))

    return bytes(serialized)


def prebuild(weight: torch.Tensor, bias: Optional[torch.Tensor],
             input_shape, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1),
             force: bool = False) -> str:
    """Explicit, offline warmup: builds (or reuses, unless force=True) a
    REFIT-capable ONNX-based TensorRT engine for this weight shape/dtype +
    input shape/config, and writes it to the on-disk cache try_forward()
    reads from. `weight`/`bias` are used only for their shape/dtype/
    has-bias-ness -- their VALUES are irrelevant (placeholder weights get
    baked in at build time; real values are refit in later, see module
    docstring) -- so ANY correctly-shaped tensor works here, including
    ones from a model you haven't loaded real weights into yet.

    SLOW the first time for a given structure (minutes, real conv3d
    shapes measured around 8 in testing, almost entirely TensorRT's own
    CPU-side tactic search); near-instant on every later call for the
    same structure (just confirms the on-disk engine is already there).
    Call this once per distinct (weight shape, input shape,
    stride/padding/dilation, dtype) you expect to hit, ahead of actual
    inference -- never from inside a forward pass. One successful
    prebuild() then transparently covers every real conv3d layer that
    happens to share that exact shape/config, not just one weight tensor.

    Returns the cache key (also derivable via try_forward's own hashing,
    exposed here mainly so callers can log/verify what got built).
    """
    key = _structure_key(tuple(weight.shape), bias is not None, weight.dtype,
                          input_shape, stride, padding, dilation)
    path = _engine_path(key)
    if path.is_file() and not force:
        return key
    engine_bytes = _build_engine_bytes(tuple(weight.shape), bias is not None, weight.dtype,
                                        input_shape, stride, padding, dilation)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(engine_bytes)
    return key


def _get_engine(key: str) -> Optional[_Engine]:
    with _lock:
        cached = _engines.get(key)
        if cached is not None:
            return cached
        path = _engine_path(key)
        if not path.is_file():
            return None
        engine = _Engine()
        try:
            engine.load(path.read_bytes())
        except Exception:
            return None
        _engines[key] = engine
        return engine


def _time_calls(fn, stream: torch.cuda.Stream, warmup: int, iters: int) -> float:
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


def try_forward(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
                 stride, padding, dilation,
                 native_fn) -> Optional[torch.Tensor]:
    """Fast path for _patched_conv3d: returns None immediately (no engine
    build attempt, negligible overhead) unless prebuild() already
    populated the cache for this EXACT structure (weight shape, input
    shape, stride/padding/dilation, dtype -- NOT weight content, see
    module docstring). When a cached engine exists, refits it with this
    call's real weight/bias (skipped if unchanged since last call, see
    _Engine.refit) and, on the first hit for this structure, benchmarks
    it once against `native_fn` (the existing _C.conv3d dispatch --
    itself possibly already choosing between the hand-tuned kernel and
    the C++ Network-API TRT path), caching which one wins -- same
    "measure once, trust the cache after that" design as
    trt_faster_than_fallback (src/trt/trt_common.cpp) uses on the C++
    side. Returns the correct output tensor either way, or None if this
    path isn't usable at all (no cached engine, or it failed to run).
    """
    key = _structure_key(tuple(weight.shape), bias is not None, weight.dtype,
                          tuple(input.shape), stride, padding, dilation)

    with _lock:
        cached_decision = _decisions.get(key)
    if cached_decision is False:
        return None  # already measured slower than native for this exact structure

    engine = _get_engine(key)
    if engine is None:
        return None  # never prebuilt -- do NOT attempt a build here

    stream = torch.cuda.current_stream()
    try:
        engine.refit(weight, bias)

        if cached_decision is True:
            return engine.infer(input, stream.cuda_stream).clone()

        # First hit for this structure: benchmark once, cache the decision.
        def run_onnx_trt():
            engine.infer(input, stream.cuda_stream)

        def run_native():
            native_fn()

        onnx_ms = _time_calls(run_onnx_trt, stream, warmup=2, iters=5)
        native_ms = _time_calls(run_native, stream, warmup=2, iters=5)
        use_onnx = onnx_ms < native_ms

        with _lock:
            _decisions[key] = use_onnx

        if use_onnx:
            return engine.infer(input, stream.cuda_stream).clone()
        return None
    except Exception:
        with _lock:
            _decisions[key] = False
        return None
