"""Use whichever implementation is actually fastest, decided at runtime.

Every kernel in this package is a bet: FMA-free hand-tuned CUDA beats stock
cuDNN/cuBLAS *on a throttled CMP mining card*. The bet pays off spectacularly
for some ops and loses badly for others, and which is which depends on the
shape, not just the op. Measured on the CMP 50HX (benchmark/results_nv.json):

    group_norm fp16   5.53x faster      attention fp32   0.50x  (2x SLOWER)
    silu       fp16   5.91x faster      conv3d   fp32   ~19.7ms vs 282ms cuDNN
    conv2d     fp32   3.75x faster      conv3d   fp32    0.14ms vs 0.07ms direct
                                                         at 8 channels

Before this module the answer was hardcoded per op, one way or the other, for
every shape: conv3d/conv_transpose2d/interpolate were switched off wholesale
behind CMPEXT3_ENABLE_UNVERIFIED_KERNELS because nobody had measured them,
while fp32 attention stayed switched ON at half the speed of stock because
nobody had re-measured it after it was written. Both are the same mistake.

So: don't decide up front. The first call for a given (op, shapes, dtypes,
parameters) runs BOTH implementations, keeps the faster one for every later
call with that signature, and never runs the loser again. This is the same
contract src/main.cpp's cmpext3_alt_kernel_faster applies one level down,
between two native conv3d kernels -- see that function; the two layers
compose (the native side of a conv3d decision here is itself whichever of
Winograd/direct won there).

Correctness comes first, and comes free: both outputs already exist during
the probe, so they get compared before either one is timed. A native kernel
whose output doesn't match stock is never selected no matter how fast it is
-- which is what makes it safe to stop gating "unverified" kernels behind an
environment variable and simply let them prove themselves per shape.

Knobs (all optional):
    CMPEXT3_AUTOSELECT=0        Skip all of this: always use the native
                                kernel where it's eligible, falling back to
                                stock only on an exception. The behavior this
                                package had before this module existed.
    CMPEXT3_VERBOSE=1           Print each decision to stderr as it's made
                                (also turns on the equivalent line from the
                                native conv3d Winograd/direct choice).
    CMPEXT3_AUTOSELECT_BUDGET_MS=20
                                Roughly how long to spend timing each
                                candidate. Bigger is more reliable on noisy
                                shapes and slower to start up.
    CMPEXT3_AUTOSELECT_TOLERANCE=<float>
                                How far a kernel's output may sit from stock's
                                (relative to the output's peak) and still be
                                eligible. Overrides the per-dtype defaults --
                                see _TOLERANCE below for the one kernel this
                                actually decides today.

cmpext3.autoselect_report() dumps every decision made so far.
"""
from __future__ import annotations

import math
import os
import sys
import time
from typing import Any, Callable

import torch


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in ("0", "", "false", "False")


_ENABLED = _flag("CMPEXT3_AUTOSELECT", "1")
_VERBOSE = _flag("CMPEXT3_VERBOSE", "0")

try:
    _BUDGET_S = max(1.0, float(os.environ.get("CMPEXT3_AUTOSELECT_BUDGET_MS", "20"))) / 1000.0
except ValueError:
    _BUDGET_S = 0.020

# Past this many distinct signatures for one op, stop probing new ones and
# reuse what that op has already shown (see _fallback_verdict). Without a cap,
# an op whose shape changes every call -- attention over a variable sequence
# length is the obvious one -- would pay the probe cost forever and grow this
# dict without bound. 64 is well past the number of distinct signatures a
# diffusion UNet/VAE actually presents per op, so in practice the cap only
# ever engages for genuinely dynamic shapes.
_MAX_SIGNATURES_PER_OP = 64

# How far a native kernel's output may sit from stock's before it's rejected
# outright, as a fraction of the reference's largest absolute value. These are
# deliberately loose -- the job is to catch a kernel that's WRONG (bad
# indexing, uninitialized memory, a scope guard that let through a shape it
# can't handle), not to police float32 rounding. For scale: the fp16 conv3d
# kernel's measured worst case is ~3e-4 relative, more than 100x inside the
# fp16 bound below, and the fp32 Winograd kernel's is ~5e-6.
_TOLERANCE = {
    torch.float32: 1e-3,
    torch.float16: 3e-2,
    torch.bfloat16: 5e-2,
}

# One number to override the whole table, for deciding the trade-off yourself.
# The case this exists for, concretely: fp16_conv3d.cu accumulates in half
# (half2 accum[][] + __hfma2 -- that IS the design, it's how the kernel dodges
# the FFMA throttle), so its error grows with the reduction length: measured
# against an fp32 reference at 8x32x32 it is off by 3.1 at 64 channels and
# 25.5 at 512, where cuDNN -- which accumulates in fp32 -- stays at ~0.4. At
# 512 channels that is 4.3% of the output's peak, over the bound above, so the
# kernel gets dropped there despite being ~2.3x faster. Raise this if that
# trade is worth it for your workload, and look at the pictures before you do.
_TOLERANCE_OVERRIDE: float | None = None
try:
    if os.environ.get("CMPEXT3_AUTOSELECT_TOLERANCE"):
        _TOLERANCE_OVERRIDE = float(os.environ["CMPEXT3_AUTOSELECT_TOLERANCE"])
except ValueError:
    _TOLERANCE_OVERRIDE = None


class Decision:
    """Why one implementation is being used for one call signature."""

    __slots__ = ("op", "use_native", "native_ms", "stock_ms", "note")

    def __init__(self, op, use_native, native_ms=None, stock_ms=None, note=""):
        self.op = op
        self.use_native = use_native
        self.native_ms = native_ms
        self.stock_ms = stock_ms
        self.note = note

    @property
    def speedup(self) -> float | None:
        if self.native_ms and self.stock_ms:
            return self.stock_ms / self.native_ms
        return None

    def __repr__(self) -> str:
        which = "cmpext3" if self.use_native else "stock"
        if self.native_ms is not None and self.stock_ms is not None:
            return (f"<{self.op}: {which} "
                    f"({self.native_ms * 1000:.3f}ms vs {self.stock_ms * 1000:.3f}ms stock)>")
        return f"<{self.op}: {which} ({self.note})>"


_decisions: dict[Any, Decision] = {}
_probe_counts: dict[str, int] = {}
_warned: set[str] = set()


def enabled() -> bool:
    return _ENABLED


def reset() -> None:
    """Forget every decision, so the next call re-measures. Mainly for tests."""
    _decisions.clear()
    _probe_counts.clear()
    _warned.clear()


def report() -> list[Decision]:
    """Every decision made so far, worst speedup first."""
    return sorted(_decisions.values(), key=lambda d: d.speedup or 0.0)


def format_report() -> str:
    if not _decisions:
        return "cmpext3: no kernel decisions made yet (no eligible op has run)."
    lines = [f"cmpext3: {len(_decisions)} kernel decision(s), slowest native first:"]
    for d in report():
        if d.native_ms is not None and d.stock_ms is not None:
            lines.append(f"  {'cmpext3' if d.use_native else 'stock  '}  {d.op:<24} "
                         f"{d.native_ms * 1000:9.3f} ms vs {d.stock_ms * 1000:9.3f} ms stock "
                         f"({d.speedup:.2f}x)")
        else:
            lines.append(f"  {'cmpext3' if d.use_native else 'stock  '}  {d.op:<24} {d.note}")
    return "\n".join(lines)


def _log(message: str) -> None:
    if _VERBOSE:
        sys.stderr.write(f"[cmpext3] {message}\n")
        sys.stderr.flush()


def _warn_once(tag: str, message: str) -> None:
    if tag in _warned:
        return
    _warned.add(tag)
    sys.stderr.write(f"[cmpext3] {message}\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time(fn: Callable) -> tuple[float, Any]:
    """Seconds per call, plus the last result -- so the caller can use it.

    Iteration count is derived from one measured call rather than fixed, to
    keep the probe's cost bounded no matter the op: a 0.02ms silu gets many
    iterations, a 280ms stock conv3d gets exactly one. A fixed count would
    either be statistically useless for the fast ops or add seconds of
    startup for the slow ones.
    """
    _sync()
    start = time.perf_counter()
    out = fn()
    _sync()
    single = time.perf_counter() - start
    if single <= 0.0 or single >= _BUDGET_S:
        return single, out

    iters = min(50, max(1, int(_BUDGET_S / single)))
    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
    _sync()
    return (time.perf_counter() - start) / iters, out


# ---------------------------------------------------------------------------
# Correctness gate
# ---------------------------------------------------------------------------

def _mismatch(native_out: Any, stock_out: Any) -> str | None:
    """None if the two agree; a short human-readable reason if they don't."""
    if not isinstance(native_out, torch.Tensor) or not isinstance(stock_out, torch.Tensor):
        # Nothing here returns non-tensors today; if that ever changes, decline
        # to judge rather than guess.
        return None
    if tuple(native_out.shape) != tuple(stock_out.shape):
        return f"shape {tuple(native_out.shape)} vs stock {tuple(stock_out.shape)}"

    a = native_out.detach().float()
    b = stock_out.detach().float()
    if not torch.isfinite(a).all().item():
        return "native output contains NaN/Inf"

    diff = (a - b).abs().max().item()
    scale = b.abs().max().item()
    if not math.isfinite(diff):
        return "native output contains NaN/Inf"
    rel = diff / scale if scale > 0.0 else diff
    tol = _TOLERANCE_OVERRIDE if _TOLERANCE_OVERRIDE is not None else \
        _TOLERANCE.get(stock_out.dtype, 1e-3)
    if rel > tol:
        return f"max relative difference {rel:.2e} exceeds {tol:.0e}"
    return None


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _fallback_verdict(op: str) -> bool:
    """Verdict for a signature that won't be probed (see _MAX_SIGNATURES_PER_OP).

    Goes with the majority of what this op has already shown rather than
    assuming either way, so a dynamically-shaped op inherits the answer its own
    measured shapes gave instead of a hardcoded guess.
    """
    seen = [d for d in _decisions.values() if d.op == op]
    if not seen:
        return True
    return sum(1 for d in seen if d.use_native) * 2 >= len(seen)


def _record(key, op, use_native, native_ms=None, stock_ms=None, note="") -> Decision:
    decision = Decision(op, use_native, native_ms, stock_ms, note)
    _decisions[key] = decision
    _log(f"{op}: using {'the cmpext3 kernel' if use_native else 'stock'} -- "
         + (f"{native_ms * 1000:.3f} ms vs {stock_ms * 1000:.3f} ms stock"
            if native_ms is not None and stock_ms is not None else note))
    return decision


def choose(op: str, key: Any, native_call: Callable, stock_call: Callable) -> Any:
    """Run whichever of the two is faster for this signature, and return it.

    `native_call` may raise RuntimeError/TypeError for a shape its kernel
    doesn't handle -- that's a permanent "use stock here", not an error.
    `stock_call` must always work. Every call returns a real result: probe,
    cache hit, and cache miss alike.
    """
    decision = _decisions.get(key)
    if decision is not None:
        if not decision.use_native:
            return stock_call()
        try:
            return native_call()
        except (RuntimeError, TypeError):
            # A one-off failure (transient OOM, say) on a decision that was
            # measured good. Serve this call from stock and leave the decision
            # alone so the next one tries the kernel again.
            return stock_call()

    if _probe_counts.get(op, 0) >= _MAX_SIGNATURES_PER_OP:
        use_native = _fallback_verdict(op)
        _record(key, op, use_native,
                note=f"not probed ({_MAX_SIGNATURES_PER_OP} signatures already measured "
                     f"for {op}; following that majority)")
        if use_native:
            try:
                return native_call()
            except (RuntimeError, TypeError):
                return stock_call()
        return stock_call()

    _probe_counts[op] = _probe_counts.get(op, 0) + 1

    # Untimed warmup for both, which is also where an inapplicable shape or a
    # first-call CUDA module load gets paid for instead of landing in a
    # measurement.
    try:
        native_out = native_call()
    except (RuntimeError, TypeError) as exc:
        _record(key, op, False, note=f"native kernel raised {type(exc).__name__}: {exc}")
        return stock_call()
    stock_out = stock_call()

    reason = _mismatch(native_out, stock_out)
    if reason is not None:
        _record(key, op, False, note=f"output mismatch -- {reason}")
        _warn_once(
            f"mismatch:{op}",
            f"{op}: the native kernel's output does not match stock ({reason}); "
            f"using stock for this shape. This is a kernel bug, not a tuning "
            f"decision -- please report it with the shape involved.",
        )
        return stock_out

    native_ms, native_out = _time(native_call)
    stock_ms, stock_out = _time(stock_call)
    use_native = native_ms < stock_ms
    _record(key, op, use_native, native_ms, stock_ms)
    return native_out if use_native else stock_out
