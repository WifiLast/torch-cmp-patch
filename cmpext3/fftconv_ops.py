"""FFT-based N-D convolution -- the large-kernel counterpart to this
package's dense conv2d/conv3d CUDA kernels (src/cuda/*conv*.cu), which are
tuned for small (3x3-style) kernels and get relatively worse as kernel
width grows: a direct/im2col kernel is O(N*K) per spatial dim, FFT-conv is
O(N log N), so past some kernel width the algorithmic win outweighs FFT's
fixed transform overhead.

Ported from other/amd_tools/source/cmp_ext_turing/amd_tuned_torch/
fftconv_ops.py (itself vendored there from fft-conv-pytorch, mainly to
preserve the caller's dtype instead of upstream's silent upcast-to-fp32).
The core algorithm is pure PyTorch -- `torch.fft.rfftn`/`irfftn`, `F.pad`,
`torch.kron`, plain `@`/`einsum` -- no HIP/CK/CUDA-C at all, so despite
being written for RDNA3/ROCm it isn't actually ROCm-specific: CUDA's
`torch.fft` already goes through cuFFT, the same way ROCm's goes through
hipFFT, so this ports essentially unchanged.

WHAT WAS DELIBERATELY *NOT* PORTED, AND WHY:

  - `rocfft_ops`/`vkfft_ops` routing: amd_tuned_torch contests torch.fft
    against a direct ctypes binding of librocfft, because on ROCm
    torch.fft's own hipFFT plan-cache overhead was worth bypassing for
    some shapes. There is no CUDA equivalent need here -- torch.fft on
    CUDA already calls cuFFT about as directly as this package could, and
    building a raw cuFFT ctypes binding just to re-discover "it's the same
    library, sometimes plan-cache overhead matters" is not worth the
    surface area for a port with no hardware to measure it on.
  - The `kernel_select`-based "pad the transform to a smoother FFT length"
    contest: a real optimization (a length whose factorization has no good
    radix kernel costs several times a nearby smooth length), but it's
    built entirely on amd_tuned_torch's own `kernel_select` measure-once-
    cache-the-winner infrastructure, which doesn't exist in cmpext3 (this
    package's equivalent, `cmpext3/autoselect.py`, is a strict 2-way
    native-vs-stock chooser, not the N-way contest this needs). Dropped
    for this initial port rather than rebuilt from scratch sight-unseen;
    `fft_conv` below is still fully correct without it, just doesn't get
    the extra smooth-length speedup. Revisit if profiling on real hardware
    shows a specific shape's transform length is the bottleneck.
  - Everything else in the upstream module (env-var gates for those two
    features, the `_contest`/`x_dtype` plumbing) went with them.

WHAT WAS KEPT, VERBATIM WHERE POSSIBLE, BECAUSE IT'S GENUINE ALGORITHM/
CORRECTNESS CONTENT, NOT ROCm PLUMBING: `to_ntuple`, `complex_matmul`
(including its depthwise fast path and the einsum-over-`@` layout fix --
see that function's own docstring for the 6-80x numbers upstream measured
on RDNA3; the *reason* those numbers exist -- `@` over non-contiguous
1-wide-dim views forcing a pathological batched-GEMM shape -- is
hardware-independent, so the fix almost certainly still matters here even
though the exact multiplier hasn't been re-measured on a CMP card), and
`fft_conv`/`fft_conv1d`/`fft_conv2d`/`fft_conv3d`/`fft_convnd` themselves,
including the mixed-precision policy (fp16/bf16 storage, fp32 only for the
transform itself) and the kernel-wider-than-input validation.

WHAT'S EXPOSED. `fft_conv1d`/`fft_conv2d`/`fft_conv3d`/`fft_convnd` --
explicit, `F.convNd`-shaped functions for a caller building a long-kernel
block (Hyena-style global convolution, a long-kernel 1D audio/vocoder
layer, a large 3D kernel in a video-diffusion VAE) to call directly, same
"plain library surface" posture as `cmpext3.ops.group_norm_silu`. There is
also a conservative auto-patch integration: `_patched_conv2d`/
`_patched_conv3d` in cmpext3/__init__.py route to `fftconv_candidate`
instead of the native CUDA kernel once the kernel's widest spatial axis
crosses `CMPEXT3_FFTCONV2D_MIN_KERNEL`/`CMPEXT3_FFTCONV3D_MIN_KERNEL` --
see that module for the thresholds and gating.

UNVALIDATED ON A REAL CMP/TURING CARD, same as every other port in this
project done without hardware access: the algorithm is unchanged from
upstream's own (independently, non-ROCm-specifically tested) fft-conv-
pytorch lineage, but the mixed-precision boundary and the grouped/
depthwise paths have not been re-verified numerically or for speed here.
Before relying on this for a real large-kernel workload: compare its
output against `F.conv1d/2d/3d` at a kernel width small enough for stock
to still be fast, and benchmark against stock at your actual (large)
kernel width and shape -- there is no calibrated crossover for this
hardware, only the upstream-benchmark-derived static defaults in
`maybe_fft_conv`/`maybe_fft_conv1d` and cmpext3/__init__.py's auto-patch
thresholds.
"""
from __future__ import annotations

from math import ceil, floor
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor


def available() -> bool:
    """Always True -- this module has no compiled extension or optional
    dependency to gate on; it is pure PyTorch."""
    return True


def to_ntuple(val: Union[int, Iterable[int]], n: int) -> Tuple[int, ...]:
    """Casts to a tuple of length `n`. A bare `str` is excluded from the
    `Iterable` branch even though Python considers it iterable, so
    `to_ntuple("same", n=2)` raises instead of silently iterating
    characters -- not reachable through this module's own call sites
    (padding="same" is handled before to_ntuple ever sees the string) but
    not a foot-gun worth keeping either."""
    if isinstance(val, Iterable) and not isinstance(val, str):
        out = tuple(val)
        if len(out) == n:
            return out
        raise ValueError(f"Cannot cast tuple of length {len(out)} to length {n}.")
    return n * (val,)


def complex_matmul(a: Tensor, b: Tensor, groups: int = 1) -> Tensor:
    """Grouped complex-valued frequency-domain multiply: for every frequency
    bin, `out[n, o] = sum_i a[n, i] * b[o, i]` summed over the input channels
    belonging to output channel `o`'s group. `a` is [N, Cin, *freq] out of
    rfftn, `b` is [Cout, Cin/groups, *freq]; the result is [N, Cout, *freq].

    WRITTEN AS AN EINSUM, NOT A RESHAPED `@`. The naive formulation (move the
    channel axis to the end, add a length-1 axis, call `@`) asks the BLAS
    backend for a batched GEMM over every frequency bin whose operands are
    non-contiguous views with a degenerate 1-wide dimension -- a
    pathological request. Measured on RDNA3 (complex64, the shapes
    `fft_conv` actually produces): 27.7ms for a dense 1D conv's [8,128] x
    [128,128] over 4609 bins vs. 3.5ms for the same contraction written as
    an einsum over contiguous operands (6-8x, bit-identical output -- same
    contraction order). The win is layout, not arithmetic; there is no
    reason to expect this to be CUDA-specific, since the underlying issue
    (a degenerate batched-GEMM shape) is a BLAS-dispatch problem, not a
    ROCm one.

    DEPTHWISE IS NOT A MATMUL AT ALL. When `groups == Cin` (Cin/groups == 1,
    Cout == groups) the per-bin contraction is over a single input channel,
    i.e. a scalar complex product, and dispatching it as a batched GEMM over
    N*C*bins 1x1 matrices is the worst case of the paragraph above (measured
    62.7ms vs. 0.78ms, 80x, for the plain elementwise multiply below).

    Both branches are ordinary differentiable primitives (`einsum`, `*`), so
    `fft_conv`'s real (no custom kernel) autograd is unaffected."""
    n, cin = a.shape[0], a.shape[1]
    cout, freq = b.shape[0], a.shape[2:]
    a = a.reshape(n, cin, -1)
    b = b.reshape(cout, b.shape[1], -1)

    if b.shape[1] == 1 and cout == groups:
        # Depthwise: one input channel per output channel, no sum to do.
        out = a * b.squeeze(1).unsqueeze(0)
    else:
        out = torch.einsum(
            "ngif,goif->ngof",
            a.view(n, groups, cin // groups, -1),
            b.view(groups, cout // groups, b.shape[1], -1),
        ).reshape(n, cout, -1)

    return out.reshape(n, cout, *freq)


def fft_conv(
    signal: Tensor,
    kernel: Tensor,
    bias: Optional[Tensor] = None,
    padding: Union[int, Iterable[int], str] = 0,
    padding_mode: str = "constant",
    stride: Union[int, Iterable[int]] = 1,
    dilation: Union[int, Iterable[int]] = 1,
    groups: int = 1,
) -> Tensor:
    """N-D convolution via FFT: signal_fr = rfft(signal), kernel_fr =
    rfft(kernel), multiply in the frequency domain, irfft back. Wins over
    direct/im2col convolution for large kernels (see module docstring) --
    cmpext3's hand-tuned CUDA conv2d/conv3d kernels already cover the
    small-kernel case.

    Same call signature and semantics as F.convNd (padding="same" also
    supported, stride=1/dilation=1 only), N inferred from
    `signal.ndim - 2`.

    MIXED PRECISION: fp16/bf16 STORAGE, fp32 TRANSFORM. `signal`/`kernel`
    stay in the caller's own dtype through every pure-data-movement step --
    the `torch.kron` dilation expansion and both `F.pad` calls, which touch
    the largest tensors here (the padded signal in particular, since a
    long-kernel/long-sequence workload is exactly what this targets) and
    carry no precision-sensitive math of their own. Only `rfftn`/
    `complex_matmul`/`irfftn` (the actual transform and frequency-domain
    pointwise multiply, where low precision would dominate the numerical
    error, and which `torch.fft.rfftn`/`irfftn` only support in
    float32/float64 regardless) run in float32 -- cast right before
    `rfftn`, cast back to `signal`'s original dtype immediately after
    `irfftn`, so nothing after that point (the stride/kernel-size crop, the
    bias add) pays for float32 either. Output dtype always matches
    `signal`'s."""
    n = signal.ndim - 2
    stride_ = to_ntuple(stride, n=n)
    dilation_ = to_ntuple(dilation, n=n)
    if isinstance(padding, str):
        if padding != "same":
            raise ValueError(f"Padding mode {padding} not supported.")
        if stride != 1 or dilation != 1:
            raise ValueError("stride must be 1 for padding='same'.")
        padding_ = tuple((k - 1) / 2 for k in kernel.shape[2:])
    else:
        padding_ = to_ntuple(padding, n=n)

    out_dtype = signal.dtype

    # --- fp16/bf16 storage: dilation expansion + padding, still in
    # `signal`'s original dtype (see docstring's MIXED PRECISION section).
    offset = torch.zeros(1, 1, *dilation_, device=signal.device, dtype=signal.dtype)
    offset[(slice(None), slice(None), *((0,) * n))] = 1.0

    cutoff = tuple(slice(None, -d + 1 if d != 1 else None) for d in dilation_)
    kernel = torch.kron(kernel, offset)[(slice(None), slice(None)) + cutoff]

    signal_padding = [r(p) for p in padding_[::-1] for r in (floor, ceil)]
    signal = F.pad(signal, signal_padding, mode=padding_mode)

    signal_size = signal.size()

    # --- A kernel wider than the (already padded, already dilation-expanded)
    # signal has no valid output position at all. F.convNd rejects this
    # outright; so does this, with the same complaint -- without this check
    # F.pad would silently accept the negative pad it implies and truncate
    # the kernel, and the crop below would slice to an empty, wrongly-shaped
    # tensor instead of failing.
    invalid = [
        (i - 2, int(signal_size[i]), kernel.size(i))
        for i in range(2, signal.ndim)
        if kernel.size(i) > signal_size[i]
    ]
    if invalid:
        got = " x ".join(str(int(signal_size[i])) for i in range(2, signal.ndim))
        want = " x ".join(str(kernel.size(i)) for i in range(2, signal.ndim))
        raise ValueError(
            f"Calculated padded input size per channel: ({got}). "
            f"Kernel size: ({want}). Kernel size can't be greater than "
            f"actual input size (axis {invalid[0][0]})."
        )

    if signal.size(-1) % 2 != 0:
        signal = F.pad(signal, [0, 1])

    kernel_padding = [
        pad
        for i in reversed(range(2, signal.ndim))
        for pad in [0, signal.size(i) - kernel.size(i)]
    ]
    padded_kernel = F.pad(kernel, kernel_padding)

    # --- fp32 transform + pointwise multiply: the only part of this
    # function where precision actually matters, and the only part
    # torch.fft requires it for regardless of `signal`'s own dtype.
    fft_dims = tuple(range(2, signal.ndim))
    signal_fr = torch.fft.rfftn(signal.float(), dim=fft_dims)
    kernel_fr = torch.fft.rfftn(padded_kernel.float(), dim=fft_dims)

    kernel_fr.imag *= -1
    output_fr = complex_matmul(signal_fr, kernel_fr, groups=groups)
    out = torch.fft.irfftn(output_fr, dim=fft_dims)

    # --- back to fp16/bf16 storage immediately: the crop and bias-add
    # below are again pure data movement over what's now an
    # activation-sized tensor, not a reason to keep paying for float32.
    out = out.to(out_dtype)

    crop_slices = (slice(None), slice(None)) + tuple(
        slice(0, (signal_size[i] - kernel.size(i) + 1), stride_[i - 2])
        for i in range(2, signal.ndim)
    )
    out = out[crop_slices].contiguous()

    if bias is not None:
        bias_shape = tuple([1, -1] + (signal.ndim - 2) * [1])
        out = out + bias.to(out_dtype).view(bias_shape)

    return out


def fft_conv1d(signal: Tensor, kernel: Tensor, bias: Optional[Tensor] = None,
                padding: Union[int, Iterable[int], str] = 0, padding_mode: str = "constant",
                stride: Union[int, Iterable[int]] = 1, dilation: Union[int, Iterable[int]] = 1,
                groups: int = 1) -> Tensor:
    """F.conv1d-equivalent via FFT. `signal` [B,Ci,L], `kernel` [Co,Ci/groups,K]."""
    if signal.dim() != 3 or kernel.dim() != 3:
        raise ValueError("fft_conv1d expects 3D [B,C,L] signal and kernel tensors.")
    return fft_conv(signal, kernel, bias=bias, padding=padding, padding_mode=padding_mode,
                     stride=stride, dilation=dilation, groups=groups)


def fft_conv2d(signal: Tensor, kernel: Tensor, bias: Optional[Tensor] = None,
                padding: Union[int, Iterable[int], str] = 0, padding_mode: str = "constant",
                stride: Union[int, Iterable[int]] = 1, dilation: Union[int, Iterable[int]] = 1,
                groups: int = 1) -> Tensor:
    """F.conv2d-equivalent via FFT. `signal` [B,Ci,H,W], `kernel` [Co,Ci/groups,Kh,Kw]."""
    if signal.dim() != 4 or kernel.dim() != 4:
        raise ValueError("fft_conv2d expects 4D [B,C,H,W] signal and kernel tensors.")
    return fft_conv(signal, kernel, bias=bias, padding=padding, padding_mode=padding_mode,
                     stride=stride, dilation=dilation, groups=groups)


def fft_conv3d(signal: Tensor, kernel: Tensor, bias: Optional[Tensor] = None,
                padding: Union[int, Iterable[int], str] = 0, padding_mode: str = "constant",
                stride: Union[int, Iterable[int]] = 1, dilation: Union[int, Iterable[int]] = 1,
                groups: int = 1) -> Tensor:
    """F.conv3d-equivalent via FFT. `signal` [B,Ci,D,H,W], `kernel` [Co,Ci/groups,Kd,Kh,Kw]."""
    if signal.dim() != 5 or kernel.dim() != 5:
        raise ValueError("fft_conv3d expects 5D [B,C,D,H,W] signal and kernel tensors.")
    return fft_conv(signal, kernel, bias=bias, padding=padding, padding_mode=padding_mode,
                     stride=stride, dilation=dilation, groups=groups)


def fft_convnd(signal: Tensor, kernel: Tensor, bias: Optional[Tensor] = None,
                padding: Union[int, Iterable[int], str] = 0, padding_mode: str = "constant",
                stride: Union[int, Iterable[int]] = 1, dilation: Union[int, Iterable[int]] = 1,
                groups: int = 1) -> Tensor:
    """F.convNd-equivalent via FFT for ANY number of spatial axes -- `signal`
    [B,Ci,*spatial], `kernel` [Co,Ci/groups,*kernel_spatial], N inferred as
    `signal.dim() - 2`. Requires at least one spatial axis."""
    if signal.dim() < 3:
        raise ValueError(
            f"fft_convnd expects a [B,C,*spatial] signal with at least one "
            f"spatial axis, got a {signal.dim()}D tensor."
        )
    if kernel.dim() != signal.dim():
        raise ValueError(
            f"fft_convnd expects signal and kernel of equal rank, got "
            f"{signal.dim()}D signal and {kernel.dim()}D kernel."
        )
    return fft_conv(signal, kernel, bias=bias, padding=padding, padding_mode=padding_mode,
                     stride=stride, dilation=dilation, groups=groups)


def fftconv_candidate(input: Tensor, weight: Tensor, bias: Optional[Tensor],
                       stride, padding, dilation, groups: int) -> Optional[Tensor]:
    """Thunk-friendly `fft_conv` wrapper: returns None instead of raising so
    an ineligible or failing call simply declines rather than breaking
    whatever contest it's a candidate in (matches cmpext3.autoselect's own
    "raise RuntimeError/TypeError to decline" convention for a native-kernel
    candidate). Used by `_patched_conv2d`/`_patched_conv3d`'s large-kernel
    auto-patch path in cmpext3/__init__.py.

    None if the ranks disagree, or if the call fails for any reason --
    including the kernel-wider-than-input case `fft_conv` rejects outright."""
    if input.dim() < 3 or weight.dim() != input.dim():
        return None
    try:
        return fft_conv(input, weight, bias=bias, padding=padding,
                         stride=stride, dilation=dilation, groups=groups)
    except (RuntimeError, ValueError, TypeError):
        return None


def maybe_fft_conv(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
                    stride: Union[int, Tuple[int, ...]] = 1, padding: Union[int, Tuple[int, ...]] = 0,
                    dilation: Union[int, Tuple[int, ...]] = 1, groups: int = 1,
                    min_kernel: int = 128) -> Optional[Tensor]:
    """Static-heuristic direct/FFT conv switch: routes to `fft_conv` only
    when `weight`'s widest spatial axis is at or above `min_kernel`, no
    timing/measurement involved -- for a caller that wants FFT-conv's
    algorithmic-complexity argument applied directly, without cmpext3's
    autoselect machinery. Default of 128 is upstream fft-conv-pytorch's own
    1D crossover estimate; a 2D/3D caller should pass a smaller value (the
    auto-patch integration in cmpext3/__init__.py uses
    CMPEXT3_FFTCONV2D_MIN_KERNEL=32 / CMPEXT3_FFTCONV3D_MIN_KERNEL=7 as its
    own starting points -- see that module for why those are unverified
    guesses, not measurements, on a CMP/Turing card).

    `fft_conv`'s entire computation is ordinary differentiable PyTorch
    (rfftn/irfftn/pad/kron/matmul), so this has a real, correct backward
    pass and is safe to call under autograd -- no `_grad_safe` check,
    unlike every hand-written CUDA kernel elsewhere in this package.

    None when `weight`'s widest axis is below `min_kernel`, or the call
    fails for any reason (e.g. an unsupported padding_mode/padding string)."""
    if input.dim() < 3 or weight.dim() != input.dim():
        return None
    if max(weight.shape[2:]) < min_kernel:
        return None
    try:
        return fft_conv(input, weight, bias=bias, padding=padding,
                         stride=stride, dilation=dilation, groups=groups)
    except (RuntimeError, ValueError, TypeError):
        return None
