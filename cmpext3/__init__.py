"""
cmpext3 -- monkeypatches PyTorch's CUDA op dispatch with FMA-instruction-free
kernels for CMP mining cards (Turing/TU10x), where the driver throttles
FFMA/Tensor-core throughput based on device ID.

Supports fp16, fp32, and bf16 inputs. Turing has no native bf16 arithmetic
hardware, so bf16 tensors are converted at the native-kernel boundary and
run through an existing FMA-free kernel -- see the per-op guards in
src/main.cpp. This matters because many diffusion/video models (e.g.
HunyuanVideo-based pipelines) run their transformer in bf16, which would
otherwise silently fail every _usable() check and fall through to
throttled stock ops for the whole model.

Every op converts bf16->fp32 (exact in, round-to-nearest-even out) EXCEPT
attention, which converts bf16->fp16 instead: the fp32 attention kernel
(fp32_attention.cu) is a naive one-thread-per-query-row implementation
with no warp-level cooperation across the head-dim reduction, measured at
~21.6s for a single S=17402 call, while the fp16 attention kernel
(fp16_attention.cu) is properly warp-cooperative and tiled, measured at
~6.5ms for the same shape -- it's the same kernel design later forked into
the sageattention/xformers CMP-Turing ports. fp16's narrower exponent
range (vs fp32/bf16) is an accepted tradeoff there since those ports
already validated bf16->fp16->bf16 round-tripping through this exact
kernel design without overflow.

Import this package to enable the patch:

    import cmpext3   # patches torch/torch.nn.functional on import

Or control it explicitly:

    import cmpext3
    cmpext3.disable()
    cmpext3.enable()

Set CMPEXT3_AUTOPATCH=0 in the environment to import without patching.

Every op is patched; which implementation actually runs is measured, not
assumed. The first call for a given op and shape runs BOTH the native
kernel and stock PyTorch, checks their outputs agree, times them, and
keeps the winner for every later call with that shape (see
cmpext3/autoselect.py). A kernel that is slower, or wrong, for a given
shape is simply never used for it again.

This replaced a pair of hardcoded guesses that were wrong in both
directions. conv3d, conv_transpose2d and F.interpolate(mode='nearest')
used to be OFF unless CMPEXT3_ENABLE_UNVERIFIED_KERNELS=1 was set, on the
theory that their untiled kernels had to be slow -- but fp32 conv3d has
since been rewritten and measures ~19.7ms against stock cuDNN's ~282ms on
a real VAE-decode shape (14x FASTER, on the same GPU where the old naive
kernel was 5.5x slower), and fp32 conv_transpose2d wins on some shapes
too. Meanwhile fp32 attention was hardcoded ON at 0.50x -- half of stock's
speed -- and interpolate's kernel turns out to be ~6-9x slower than stock.
Measuring gets all four right without anyone having to keep a table of
verdicts up to date.

Knobs: CMPEXT3_AUTOSELECT=0 restores the old "always use the kernel"
behavior, CMPEXT3_DISABLE_OPS=conv3d,interpolate pins named ops to stock,
CMPEXT3_VERBOSE=1 prints each decision as it's made, and
cmpext3.autoselect_report() dumps them all at any point. An explicit
CMPEXT3_ENABLE_UNVERIFIED_KERNELS=0 is still honored as an opt-out for
those three ops.

Also opt-in, once conv3d patching itself is on: CMPEXT3_USE_TENSORRT_ONNX=1
tries a pre-built ONNX-based TensorRT engine (see trt_onnx.py) ahead of
the hand-tuned kernel, per exact (weight content, input shape) -- but
ONLY when trt_onnx.prebuild(...) was already run for that exact case;
this never builds an engine on the fly (measured ~8 minutes for a real
shape in testing, entirely unacceptable inside a live forward pass), so
without an explicit prebuild() call this is a no-op. Separate from
CMPEXT3_USE_TENSORRT (main.cpp / src/trt/conv3d_trt_runtime.cpp's C++
Network-API path, which DOES build on first use -- see that file for why
that tradeoff is fine there but not here).

fp32 conv3d has a second hand-tuned kernel behind it, on by default and
needing no opt-in of its own: the F(2x2x2,3x3x3) Winograd implementation
in src/cuda/fp32_conv3d_winograd.cu. It only handles 3x3x3 / stride 1 /
padding 1 / dilation 1 / batch 1 with even spatial dims, and it isn't
universally faster even there, so custom_conv3d_forward measures it
against the direct fp32_conv3d.cu once per shape, caches the winner for
the rest of the process, and prints one line to stderr saying which it
picked. Set CMPEXT3_DISABLE_WINOGRAD=1 to pin fp32 conv3d to the direct
kernel -- an escape hatch for bisecting a suspected Winograd bug, not an
accuracy fallback: measured against a float64 reference, Winograd is
2-4x MORE accurate than the direct kernel (and than cuDNN) at every shape
tried. See README.md for the numbers and why.

On by default: enable() also raises the GPU's power limit to its board
maximum and locks clocks at max via NVML (nvml_boost.py), restored on
disable(). This is a *different* mechanism from the kernel patches above
-- it does not bypass the FFMA/Tensor-Core instruction-pattern throttle
those exist for, it just makes sure the card isn't also sitting at a
lower power/clock cap on top of that. It mutates global, system-wide GPU
state (not just this process) and typically needs root/Administrator for
the underlying NVML calls to actually take effect (failures are logged as
warnings and otherwise ignored). Set CMPEXT3_NVML_BOOST=0 to opt out, e.g.
on a machine shared with other workloads/users. See nvml_boost.py's
docstring for the full picture, and nvml_boost.query_throttle_reasons()
for a diagnostic read of what NVML itself can (and can't) see.

SAFETY
------
None of the custom kernels have a backward pass implemented (see
src/main.cpp -- they are raw CUDA writes via data_ptr(), not
torch::autograd::Function subclasses). Every patched call therefore falls
back to the stock PyTorch op whenever autograd is live for the tensors
involved (i.e. grad mode is enabled and any input/weight/bias actually
requires_grad), so training correctness is unaffected -- you just don't
get the speedup outside torch.no_grad()/torch.inference_mode(). This
matches the README's use case (SDXL/Anima inference in ComfyUI), not
training.

Every wrapper also falls back to the stock op on any RuntimeError/TypeError
from the native kernel (unsupported shape, odd output-channel count for
fp16 linear, groups != 1 for conv, attn_mask/dropout/is_causal for SDPA,
etc.) -- see the individual TORCH_CHECK calls in src/main.cpp for exactly
what each custom kernel supports.

Deliberately NOT auto-patched: torch.tanh, torch.erf, F.softmax,
F.softplus, F.softshrink. These are used pervasively far outside
diffusion/transformer model code (losses, RL, generic math), and their
custom implementations are approximations (see src/main.cpp's "Custom Tanh
based on exp expansion and FP16 RCP"). They're still available for manual,
opt-in use via cmpext3.ops.<name>.

cmpext3.ops.group_norm_silu(input, num_groups, weight=None, bias=None,
eps=1e-5) is a fused GroupNorm+SiLU op (SiLU applied in the same kernel's
write-back, right after the normalize+affine step) -- not monkeypatched
onto anything, since F.group_norm has no activation argument to key off of.
Call it directly in model code in place of F.group_norm(...) followed by
F.silu(...): that exact pair is the first two steps of the
GroupNorm -> SiLU -> Conv block a diffusion UNet/VAE decoder repeats
throughout, and fusing them skips one full global-memory round trip. conv2d
and conv3d also get a 1x1(x1) fast path for free (custom_conv2d_forward /
custom_conv3d_forward in src/main.cpp): a unit kernel with stride=1,
padding=0, dilation=1 is pure channel-mixing with no spatial window at all,
so it's reshaped to channels-last and dispatched straight to the tiled
linear/matmul kernel instead of the conv windowing machinery, which would
otherwise do real work for nothing.
"""
from __future__ import annotations

import os
from typing import Any, Callable

import torch
import torch.nn.functional as F

try:
    from . import _native as _C
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "cmpext3 native extension not built. Run: pip install -e . --no-build-isolation"
    ) from exc

# Raw, unpatched access to every kernel in src/main.cpp (linear, bmm, conv2d,
# conv3d, conv_transpose2d, upsample_scaling, attention, embedding,
# group_norm, layer_norm, rmsnorm, gelu, silu, swish, mish, softmax,
# softplus, softsign, softshrink, tanh, erf) for manual use regardless of
# what enable() patches.
ops = _C

from . import autoselect  # noqa: E402  (needs _C imported first)

_SUPPORTED_DTYPES = (torch.float16, torch.float32, torch.bfloat16)
_ORIGINALS: dict[tuple[Any, str], Callable] = {}
_ENABLED = False


def _grad_safe(*tensors: Any) -> bool:
    """True if swapping in a kernel with no autograd support is safe."""
    if hasattr(torch, "is_inference_mode_enabled") and torch.is_inference_mode_enabled():
        return True
    if not torch.is_grad_enabled():
        return True
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.requires_grad:
            return False
    return True


def _usable(*tensors: Any) -> bool:
    """True if every tensor is a CUDA fp16/fp32/bf16 tensor the kernels target.

    bf16 has no native arithmetic on Turing; the native kernels convert it to
    fp32 at the tensor boundary and run the existing FMA-free fp32 path (see
    the per-op "bf16 has no native arithmetic..." guards in src/main.cpp).
    """
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            continue
        if not t.is_cuda or t.dtype not in _SUPPORTED_DTYPES:
            return False
    return True


# Ops to leave on stock PyTorch regardless of what autoselect measures, as a
# comma-separated list of op names: CMPEXT3_DISABLE_OPS=conv3d,interpolate.
# Rarely needed now that a losing kernel simply isn't selected, but it's the
# blunt instrument for pinning one op to stock while debugging.
_OP_ALIASES = {"scaled_dot_product_attention": "attention"}
_DISABLED_OPS = frozenset(
    part.strip() for part in os.environ.get("CMPEXT3_DISABLE_OPS", "").split(",") if part.strip()
)

# CMPEXT3_ENABLE_UNVERIFIED_KERNELS used to gate conv3d/conv_transpose2d/
# interpolate off by default. It no longer gates anything on (autoselect
# measures them instead -- see enable()), but an explicit "0" is still
# honored as "keep those three on stock", so nobody who set it deliberately
# gets kernels switched on under them by upgrading.
if os.environ.get("CMPEXT3_ENABLE_UNVERIFIED_KERNELS") in ("0", "false", "False"):
    _DISABLED_OPS = _DISABLED_OPS | frozenset({"conv3d", "conv_transpose2d", "interpolate"})


def _install(target: Any, name: str, wrapper: Callable) -> None:
    if name in _DISABLED_OPS or _OP_ALIASES.get(name, name) in _DISABLED_OPS:
        return
    key = (target, name)
    if key not in _ORIGINALS:
        _ORIGINALS[key] = getattr(target, name)
    setattr(target, name, wrapper)


def _restore(target: Any, name: str) -> None:
    original = _ORIGINALS.pop((target, name), None)
    if original is not None:
        setattr(target, name, original)


_DTYPE_TAG = {torch.float16: "fp16", torch.float32: "fp32", torch.bfloat16: "bf16"}


def _hashable(v: Any) -> Any:
    """List -> tuple, so a conv parameter can be a dict key either way."""
    return tuple(v) if type(v) is list else v


def _dispatch(op: str, ref: torch.Tensor, key: tuple,
              native_call: Callable, stock_call: Callable) -> Any:
    """Run whichever of the native kernel and stock PyTorch is faster here.

    See cmpext3/autoselect.py for how that's decided (once per signature,
    correctness checked before speed). With CMPEXT3_AUTOSELECT=0 this
    collapses to what every wrapper used to do inline: native kernel, stock
    only if it raises.
    """
    if autoselect.enabled():
        try:
            return autoselect.choose(f"{op} {_DTYPE_TAG.get(ref.dtype, ref.dtype)}",
                                     key, native_call, stock_call)
        except TypeError as exc:
            # An unhashable key -- some argument this wrapper folded into the
            # signature isn't a scalar/tuple after all. Not worth crashing a
            # forward pass over: skip the measurement for this call and use
            # the kernel, exactly as if autoselect were switched off.
            if "unhashable" not in str(exc):
                raise
    try:
        return native_call()
    except (RuntimeError, TypeError):
        return stock_call()


# ---------------------------------------------------------------------------
# Wrappers. Each checks grad-safety/dtype/device up front, decides between its
# native kernel and stock PyTorch via _dispatch (which measures both the first
# time it sees a given signature), and still falls back on any
# RuntimeError/TypeError the native kernel raises for a shape or argument
# combination it doesn't support.
# ---------------------------------------------------------------------------

def _patched_linear(input, weight, bias=None):
    orig = _ORIGINALS[(F, "linear")]
    if not (_grad_safe(input, weight, bias) and _usable(input, weight)):
        return orig(input, weight, bias)
    if input.dim() < 2 or weight.dim() != 2:
        return orig(input, weight, bias)
    if input.dtype == torch.float16 and weight.size(0) % 2 != 0:
        return orig(input, weight, bias)
    return _dispatch("linear", input, (input.shape, input.dtype, weight.shape, bias is not None),
                     lambda: _C.linear(input, weight, bias),
                     lambda: orig(input, weight, bias))


def _patched_matmul(input, other, *, out=None):
    orig = _ORIGINALS[(torch, "matmul")]
    if out is not None or not (_grad_safe(input, other) and _usable(input, other)):
        return orig(input, other) if out is None else orig(input, other, out=out)
    if input.dtype != other.dtype:
        return orig(input, other)
    if input.dim() == 2 and other.dim() == 2:
        native = lambda: _C.bmm(input.unsqueeze(0), other.unsqueeze(0)).squeeze(0)  # noqa: E731
    elif input.dim() == 3 and other.dim() == 3:
        native = lambda: _C.bmm(input, other)  # noqa: E731
    else:
        return orig(input, other)
    return _dispatch("matmul", input, (input.shape, input.dtype, other.shape),
                     native, lambda: orig(input, other))


def _patched_bmm(input, mat2, *, out=None):
    orig = _ORIGINALS[(torch, "bmm")]
    if out is not None or not (_grad_safe(input, mat2) and _usable(input, mat2)):
        return orig(input, mat2) if out is None else orig(input, mat2, out=out)
    if input.dtype != mat2.dtype:
        return orig(input, mat2)
    return _dispatch("bmm", input, (input.shape, input.dtype, mat2.shape),
                     lambda: _C.bmm(input, mat2),
                     lambda: orig(input, mat2))


def _patched_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    orig = _ORIGINALS[(F, "conv2d")]
    if groups != 1 or not (_grad_safe(input, weight, bias) and _usable(input, weight)):
        return orig(input, weight, bias, stride, padding, dilation, groups)
    return _dispatch("conv2d", input,
                     (input.shape, input.dtype, weight.shape, bias is not None,
                      _hashable(stride), _hashable(padding), _hashable(dilation)),
                     lambda: _C.conv2d(input, weight, bias, stride, padding, dilation, groups),
                     lambda: orig(input, weight, bias, stride, padding, dilation, groups))


def _norm3(v):
    """int or (int,int,int) -> (int,int,int), for a stable cache key
    regardless of which form the caller used for the same effective
    value (trt_onnx.py's cache key needs this to be consistent)."""
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v, v)


_CMPEXT3_USE_TENSORRT_ONNX = os.environ.get("CMPEXT3_USE_TENSORRT_ONNX", "0") not in ("0", "", "false", "False")


def _patched_conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    orig = _ORIGINALS[(F, "conv3d")]
    if groups != 1 or not (_grad_safe(input, weight, bias) and _usable(input, weight)):
        return orig(input, weight, bias, stride, padding, dilation, groups)

    def call_native():
        return _C.conv3d(input, weight, bias, stride, padding, dilation, groups)

    # cmpext3/trt_onnx.py's ONNX-based TensorRT path -- see that module's
    # docstring for why it's Python-side, not C++, and why it only ever
    # uses an engine that was already built ahead of time via
    # cmpext3.trt_onnx.prebuild(...) (an on-the-fly build measured ~8
    # minutes in testing -- unacceptable inside a live forward pass).
    # fp16/fp32 only (matches what's actually been tested); no dynamic
    # shapes, so this only ever engages for a shape prebuild() was
    # explicitly run for.
    if _CMPEXT3_USE_TENSORRT_ONNX and weight.dtype in (torch.float16, torch.float32):
        try:
            from . import trt_onnx
            result = trt_onnx.try_forward(
                input, weight, bias,
                _norm3(stride), _norm3(padding), _norm3(dilation),
                call_native,
            )
            if result is not None:
                return result
        except Exception:
            pass  # fall through to the native path below, same as every other failure mode here

    return _dispatch("conv3d", input,
                     (input.shape, input.dtype, weight.shape, bias is not None,
                      _hashable(stride), _hashable(padding), _hashable(dilation)),
                     call_native,
                     lambda: orig(input, weight, bias, stride, padding, dilation, groups))


def _patched_conv_transpose2d(input, weight, bias=None, stride=1, padding=0,
                               output_padding=0, groups=1, dilation=1):
    orig = _ORIGINALS[(F, "conv_transpose2d")]
    if groups != 1 or not (_grad_safe(input, weight, bias) and _usable(input, weight)):
        return orig(input, weight, bias, stride, padding, output_padding, groups, dilation)
    return _dispatch("conv_transpose2d", input,
                     (input.shape, input.dtype, weight.shape, bias is not None,
                      _hashable(stride), _hashable(padding),
                      _hashable(output_padding), _hashable(dilation)),
                     lambda: _C.conv_transpose2d(input, weight, bias, stride, padding,
                                                 output_padding, dilation, groups),
                     lambda: orig(input, weight, bias, stride, padding, output_padding,
                                  groups, dilation))


def _patched_interpolate(input, size=None, scale_factor=None, mode="nearest", **kwargs):
    orig = _ORIGINALS[(F, "interpolate")]
    target = size if size is not None else scale_factor
    if (mode != "nearest" or target is None or not isinstance(input, torch.Tensor)
            or input.dim() != 4 or kwargs.get("align_corners") is not None
            or not (_grad_safe(input) and _usable(input))):
        return orig(input, size=size, scale_factor=scale_factor, mode=mode, **kwargs)
    return _dispatch("interpolate", input, (input.shape, input.dtype, _hashable(target)),
                     lambda: _C.upsample_scaling(input, target),
                     lambda: orig(input, size=size, scale_factor=scale_factor, mode=mode, **kwargs))


def _patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                   scale=None, **kwargs):
    orig = _ORIGINALS[(F, "scaled_dot_product_attention")]
    if (attn_mask is not None or dropout_p != 0.0 or is_causal
            or query.dim() != 4 or not _grad_safe(query, key, value)
            or not _usable(query, key, value)):
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                     is_causal=is_causal, scale=scale, **kwargs)
    return _dispatch("attention", query, (query.shape, query.dtype, key.shape, value.shape, scale),
                     lambda: _C.attention(query, key, value, scale),
                     lambda: orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                                  is_causal=is_causal, scale=scale, **kwargs))


def _patched_embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0,
                        scale_grad_by_freq=False, sparse=False):
    orig = _ORIGINALS[(F, "embedding")]
    if (max_norm is not None or weight.dim() != 2
            or not (_grad_safe(weight) and _usable(weight))):
        return orig(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
    return _dispatch("embedding", weight,
                     (input.shape, weight.shape, weight.dtype, padding_idx,
                      scale_grad_by_freq, sparse),
                     lambda: _C.embedding(input, weight,
                                          padding_idx if padding_idx is not None else -1,
                                          scale_grad_by_freq, sparse),
                     lambda: orig(input, weight, padding_idx, max_norm, norm_type,
                                  scale_grad_by_freq, sparse))


def _patched_group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    orig = _ORIGINALS[(F, "group_norm")]
    if not (input.is_contiguous() and _grad_safe(input, weight, bias) and _usable(input)):
        return orig(input, num_groups, weight, bias, eps)
    return _dispatch("group_norm", input, (input.shape, input.dtype, num_groups,
                      weight is not None, bias is not None, eps),
                     lambda: _C.group_norm(input, num_groups, weight, bias, eps),
                     lambda: orig(input, num_groups, weight, bias, eps))


def _patched_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    orig = _ORIGINALS[(F, "layer_norm")]
    if not (input.is_contiguous() and _grad_safe(input, weight, bias) and _usable(input)):
        return orig(input, normalized_shape, weight, bias, eps)
    return _dispatch("layer_norm", input, (input.shape, input.dtype, tuple(normalized_shape),
                      weight is not None, bias is not None, eps),
                     lambda: _C.layer_norm(input, list(normalized_shape), weight, bias, eps),
                     lambda: orig(input, normalized_shape, weight, bias, eps))


# Keyword arguments each native (pybind11) elementwise op actually accepts,
# beyond `input`. Anything outside this set must never reach native_fn: a
# pybind11 overload-resolution failure builds its TypeError message by
# repr()-ing every argument, and repr() on a CUDA tensor runs
# masked_select/nonzero plus a .item() per printed element -- ~1000x slower
# than the op itself. This bit for real: F.silu(x, inplace=True) (a normal,
# common call pattern) used to hit exactly this path on every call before
# `inplace` was added to the native silu/mish signatures, costing ~61ms/call
# instead of microseconds. Checking the kwarg set up front means an
# unsupported kwarg falls back to stock immediately, without ever
# constructing that exception.
_ELEMENTWISE_SUPPORTED_KWARGS: dict[str, frozenset[str]] = {
    "gelu": frozenset(),
    "silu": frozenset({"inplace"}),
    "mish": frozenset({"inplace"}),
    "softsign": frozenset(),
}


def _make_elementwise_patch(op_name: str, fallback_key: tuple[Any, str]) -> Callable:
    native_fn = getattr(_C, op_name)
    supported_kwargs = _ELEMENTWISE_SUPPORTED_KWARGS.get(op_name, frozenset())

    def wrapper(input, *args, **kwargs):
        orig = _ORIGINALS[fallback_key]
        if not (isinstance(input, torch.Tensor) and _grad_safe(input) and _usable(input)):
            return orig(input, *args, **kwargs)
        if not set(kwargs) <= supported_kwargs:
            return orig(input, *args, **kwargs)
        if kwargs.get("inplace"):
            # An in-place op can't be benchmarked: every timing iteration
            # would feed the previous iteration's output back in, and the two
            # candidates would fight over the same buffer -- so the "compare
            # both outputs" correctness gate would be comparing garbage too.
            # Use the kernel directly, exactly as this wrapper did before
            # autoselect existed. The out-of-place form of the same op and
            # shape still gets measured normally.
            try:
                return native_fn(input, *args, **kwargs)
            except (RuntimeError, TypeError):
                return orig(input, *args, **kwargs)
        return _dispatch(op_name, input, (input.shape, input.dtype, args),
                         lambda: native_fn(input, *args, **kwargs),
                         lambda: orig(input, *args, **kwargs))

    return wrapper


def enable() -> None:
    """Patch torch/torch.nn.functional to route eligible calls through cmpext3."""
    global _ENABLED
    if _ENABLED:
        return
    _install(F, "linear", _patched_linear)
    _install(torch, "matmul", _patched_matmul)
    _install(torch, "bmm", _patched_bmm)
    _install(F, "conv2d", _patched_conv2d)

    # conv3d, conv_transpose2d and interpolate used to be excluded here
    # unless CMPEXT3_ENABLE_UNVERIFIED_KERNELS=1 was set, because their
    # kernels had never been benchmarked and were assumed to be as bad as
    # the old untiled fp32 attention kernel (which took 21.6s where the
    # rewritten one takes 6.5ms). Installing them is no longer a bet:
    # autoselect measures each of them against stock the first time it sees
    # a given shape, checks the outputs match before it even times them, and
    # keeps the winner (cmpext3/autoselect.py). A kernel that loses is never
    # used a second time, so the worst case for a bad kernel is one probe --
    # which is a far better deal than a hardcoded guess in either direction.
    # It cuts both ways: fp32 attention was hardcoded ON at HALF stock's
    # speed (0.50x, benchmark/results_nv.json) and now gets dropped on the
    # same evidence that gets conv3d switched on (14x faster than cuDNN).
    _install(F, "conv3d", _patched_conv3d)
    _install(F, "conv_transpose2d", _patched_conv_transpose2d)
    _install(F, "interpolate", _patched_interpolate)

    if hasattr(F, "scaled_dot_product_attention"):
        _install(F, "scaled_dot_product_attention", _patched_sdpa)
    _install(F, "embedding", _patched_embedding)
    _install(F, "group_norm", _patched_group_norm)
    _install(F, "layer_norm", _patched_layer_norm)
    _install(F, "gelu", _make_elementwise_patch("gelu", (F, "gelu")))
    _install(F, "silu", _make_elementwise_patch("silu", (F, "silu")))
    if hasattr(F, "mish"):
        _install(F, "mish", _make_elementwise_patch("mish", (F, "mish")))
    _install(F, "softsign", _make_elementwise_patch("softsign", (F, "softsign")))
    _ENABLED = True

    # On by default -- see nvml_boost.py's module docstring for exactly
    # what this can and can't do (it does NOT bypass the FFMA/Tensor-Core
    # instruction-pattern throttle the kernels above exist for; it only
    # raises the power limit / locks clocks to max where the board and
    # privileges allow it, on top of that). It mutates global,
    # system-wide GPU state (not just this process), so set
    # CMPEXT3_NVML_BOOST=0 to opt out (e.g. on a machine shared with other
    # workloads/users, or in an environment lacking root/Administrator
    # where the underlying NVML calls would just fail anyway).
    if os.environ.get("CMPEXT3_NVML_BOOST", "1") not in ("0", "", "false", "False"):
        from . import nvml_boost
        nvml_boost.boost()


def disable() -> None:
    """Restore every patched function to the stock PyTorch implementation."""
    global _ENABLED
    for target, name in list(_ORIGINALS.keys()):
        _restore(target, name)
    from . import nvml_boost
    nvml_boost.restore()
    _ENABLED = False


def is_enabled() -> bool:
    return _ENABLED


def autoselect_report() -> str:
    """Every native-vs-stock decision made so far, as printable text.

    Nothing is decided until an op actually runs, so this is empty right
    after enable() and fills in as the model executes. Useful for answering
    "is cmpext3 actually doing anything for my workload?" -- and for spotting
    an op where the kernel lost, which is a tuning lead rather than a bug.
    """
    return autoselect.format_report()


if os.environ.get("CMPEXT3_AUTOPATCH", "1") != "0":
    enable()
