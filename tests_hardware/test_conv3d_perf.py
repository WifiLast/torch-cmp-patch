#!/usr/bin/env python3
"""
Standalone hardware regression check for cmpext3's rewritten conv3d kernels
(src/cuda/fp32_conv3d.cu, src/cuda/fp16_conv3d.cu) -- no pytest required,
just run it directly on the machine with the CMP 50HX.

Background: the previous conv3d kernels were naive (one thread per output
voxel, zero shared-memory/register reuse) and measured ~460ms/call vs. stock
cuDNN's ~83ms/call on a real VAE-decode shape -- 5.5x *slower*, which is why
conv3d has been gated behind CMPEXT3_ENABLE_UNVERIFIED_KERNELS=1 rather than
enabled by default (see cmpext3/__init__.py's enable() and README.md). Both
kernels have been rewritten (fp32: register-blocked/pointer-hoisted,
extending fp32_conv.cu's proven conv2d design; fp16: shared-memory-tiled
GEMM, extending fp16_conv.cu's proven conv2d design) to close that gap. This
script exists to make the "did it actually get faster, and is it still
correct" question reproducible on real hardware, the same way
test_attention_bf16_routing.py did for the attention-kernel bf16 routing fix.

fp32 additionally has a third kernel behind it: the F(2x2x2,3x3x3) Winograd
implementation in src/cuda/fp32_conv3d_winograd.cu, used only for the narrow
slice of shapes it handles and only where it was measured faster than the
direct kernel for that exact shape. Check 5 below covers that path
specifically -- it is the one check here that would still pass if Winograd
never ran at all, so it verifies selection as well as numerics.

Run on the machine with the actual GPU:

    python test_conv3d_perf.py
    python test_conv3d_perf.py --dtype fp16   # just one precision

Exits 0 if everything passes, 1 if any check fails or the environment can't
run it (no CUDA / cmpext3 not built).
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import tempfile
import time
from typing import Callable, Tuple

# conv3d's F.conv3d monkeypatch is opt-in; the checks below need it engaged
# so they can observe the actual routing/perf behavior, not just the raw op.
os.environ.setdefault("CMPEXT3_ENABLE_UNVERIFIED_KERNELS", "1")
# Check 5 reads the native kernel-choice line off stderr, and both the C++ and
# the Python layer keep quiet unless asked (they'd otherwise print a line per
# op per shape in a real model run).
os.environ.setdefault("CMPEXT3_VERBOSE", "1")


def _make_conv3d_inputs(b, c_in, d, h, w, c_out, k, dtype, device, seed=0):
    import torch
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(b, c_in, d, h, w, dtype=dtype, device=device, generator=g)
    weight = torch.randn(c_out, c_in, k, k, k, dtype=dtype, device=device, generator=g)
    bias = torch.randn(c_out, dtype=dtype, device=device, generator=g)
    return x, weight, bias


def _stock_conv3d(x, weight, bias, stride, padding):
    """conv3d from stock PyTorch/cuDNN, guaranteed NOT to be the kernel under test.

    `import cmpext3` auto-patches F.conv3d on import (CMPEXT3_AUTOPATCH
    defaults to on -- see cmpext3/__init__.py's last two lines), so a plain
    F.conv3d call anywhere in this file routes straight back into the native
    kernel. For fp16/bf16 that still compared something real (the half
    kernel against the fp32 one, since the reference casts to float first),
    but for fp32 it compared the kernel against ITSELF and reported a
    bit-exact 0.0 error no matter what the kernel did. Every "vs. fp32
    reference" check below goes through here instead.
    """
    import cmpext3
    import torch.nn.functional as F
    was_enabled = cmpext3.is_enabled()
    try:
        if was_enabled:
            cmpext3.disable()
        return F.conv3d(x, weight, bias, stride=stride, padding=padding)
    finally:
        if was_enabled:
            cmpext3.enable()


def _timed(fn: Callable, warmup: int, iters: int) -> Tuple[object, float]:
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = None
    for _ in range(iters):
        out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - start) / iters


@contextlib.contextmanager
def _capture_c_stderr():
    """Capture writes to file descriptor 2, not just sys.stderr.

    The one-line "which conv3d kernel won" report comes from a
    fprintf(stderr, ...) inside cmpext3_alt_kernel_faster (src/main.cpp),
    so contextlib.redirect_stderr would never see it -- that only rebinds
    the Python object. Swapping the underlying fd is the only way to read
    what the extension printed. Yields a one-element list that holds the
    captured text once the block exits.
    """
    import torch
    sink: list[str] = []
    torch.cuda.synchronize()
    sys.stderr.flush()
    buf = tempfile.TemporaryFile(mode="w+b")
    saved_fd = os.dup(2)
    try:
        os.dup2(buf.fileno(), 2)
        yield sink
    finally:
        # The report is printed from the C++ side during the call itself,
        # but sync anyway so nothing is still in flight when fd 2 is
        # restored underneath it.
        torch.cuda.synchronize()
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        buf.seek(0)
        sink.append(buf.read().decode("utf-8", "replace"))
        buf.close()


class Check:
    def __init__(self):
        self.failures: list[str] = []

    def ok(self, label: str) -> None:
        print(f"  [PASS] {label}")

    def fail(self, label: str, detail: str) -> None:
        print(f"  [FAIL] {label}")
        print(f"         {detail}")
        self.failures.append(label)

    def require(self, label: str, condition: bool, detail: str) -> None:
        if condition:
            self.ok(label)
        else:
            self.fail(label, detail)


def check_environment(check: Check):
    import torch
    print("=" * 78)
    print("Environment")
    print("=" * 78)
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        check.fail("cuda available", "No CUDA device visible -- nothing further can run.")
        return None
    dev = torch.device("cuda:0")
    print(f"device: {torch.cuda.get_device_name(dev)}")
    print(f"compute capability: {torch.cuda.get_device_capability(dev)}")
    return dev


def check_nan_inf(check: Check, dev, dtype_name: str, dtype, shape) -> None:
    import cmpext3
    import torch
    print("\n" + "=" * 78)
    print(f"Check 1 [{dtype_name}]: conv3d output has no NaN/Inf")
    print("=" * 78)
    b, c_in, d, h, w, c_out, k = shape
    x, weight, bias = _make_conv3d_inputs(b, c_in, d, h, w, c_out, k, dtype, dev)
    out = cmpext3.ops.conv3d(x, weight, bias, 1, 1)
    finite = torch.isfinite(out).all().item()
    check.require(f"no NaN/Inf in {dtype_name} conv3d output", finite,
                   "conv3d produced NaN/Inf -- check the fp32_conv3d.cu/fp16_conv3d.cu index arithmetic.")


def check_numerics(check: Check, dev, dtype_name: str, dtype, shape) -> None:
    import cmpext3
    import torch
    import torch.nn.functional as F
    print("\n" + "=" * 78)
    print(f"Check 2 [{dtype_name}]: conv3d matches an fp32 F.conv3d reference (numerics, not speed)")
    print("=" * 78)
    # Small shape -- this checks correctness, not performance.
    b, c_in, d, h, w, c_out, k = shape
    x, weight, bias = _make_conv3d_inputs(b, c_in, d, h, w, c_out, k, dtype, dev)

    got = cmpext3.ops.conv3d(x, weight, bias, 1, 1).float()
    want = _stock_conv3d(x.float(), weight.float(), bias.float(), 1, 1)

    max_abs_err = (got - want).abs().max().item()
    cos_sim = F.cosine_similarity(got.reshape(1, -1), want.reshape(1, -1)).item()
    print(f"  max abs error vs fp32 reference: {max_abs_err:.6f}")
    print(f"  cosine similarity vs fp32 reference: {cos_sim:.6f}")
    threshold = 0.999 if dtype == torch.float32 else 0.99
    check.require(
        f"{dtype_name} conv3d numerically close to fp32 reference", cos_sim > threshold,
        f"cosine similarity {cos_sim:.4f} is below {threshold} -- new kernel likely has an "
        "indexing/accumulation bug (see fp32_conv3d.cu / fp16_conv3d.cu).",
    )


def check_1x1x1_fastpath(check: Check, dev, dtype_name: str, dtype) -> None:
    """A 1x1x1 kernel with stride=1/padding=0/dilation=1 hits a different code
    path entirely in custom_conv3d_forward: it's pure channel mixing, so it
    gets reshaped to NDHWC and dispatched straight to the tiled linear/matmul
    kernel (see cmpext3/__init__.py's module docstring) instead of going
    through fp32_conv3d.cu/fp16_conv3d.cu at all. Verify that path matches
    F.conv3d's fp32 reference on its own -- a bug here wouldn't show up in
    check_numerics, which uses a 3x3x3 kernel and never takes this branch.
    """
    import cmpext3
    import torch
    import torch.nn.functional as F
    print("\n" + "=" * 78)
    print(f"Check 2b [{dtype_name}]: 1x1x1 conv3d fast path (dispatches to the linear kernel)")
    print("=" * 78)
    b, c_in, d, h, w, c_out = 2, 16, 4, 5, 6, 12
    x, weight, bias = _make_conv3d_inputs(b, c_in, d, h, w, c_out, 1, dtype, dev)

    got = cmpext3.ops.conv3d(x, weight, bias, 1, 0).float()
    want = _stock_conv3d(x.float(), weight.float(), bias.float(), 1, 0)

    assert got.shape == want.shape, f"shape mismatch: got {tuple(got.shape)}, want {tuple(want.shape)}"
    cos_sim = F.cosine_similarity(got.reshape(1, -1), want.reshape(1, -1)).item()
    print(f"  cosine similarity vs fp32 reference: {cos_sim:.6f}")
    threshold = 0.999 if dtype == torch.float32 else 0.99
    check.require(
        f"{dtype_name} 1x1x1 conv3d (linear fast path) matches fp32 reference", cos_sim > threshold,
        f"cosine similarity {cos_sim:.4f} is below {threshold} -- check custom_conv3d_forward's "
        "1x1x1 branch in src/main.cpp (NDHWC permute / weight.view / output permute-back).",
    )


def check_routes_to_native_kernel(check: Check, dev, dtype_name: str, dtype, shape) -> None:
    """F.conv3d's monkeypatch falls back to stock on any RuntimeError from the
    native kernel (see cmpext3/__init__.py's _patched_conv3d) -- so a broken
    new kernel could pass correctness by silently never running at all. This
    counts native-kernel invocations while F.conv3d is patched to prove the
    call actually reaches the new kernel rather than silently falling back.

    The count is ">= 1", not "== 1": cmpext3/autoselect.py runs the kernel
    several times on the first call for a shape, timing it against stock to
    decide which to keep. How many times is deliberately not fixed (it scales
    the iteration count to the op's cost), so only "did it run at all" is a
    stable assertion here. Which one autoselect then picked is reported below
    and checked properly by Check 5.
    """
    import cmpext3
    import torch.nn.functional as F
    print("\n" + "=" * 78)
    print(f"Check 3 [{dtype_name}]: F.conv3d monkeypatch actually reaches the native kernel")
    print("=" * 78)

    b, c_in, d, h, w, c_out, k = shape
    x, weight, bias = _make_conv3d_inputs(b, c_in, d, h, w, c_out, k, dtype, dev)

    call_count = {"n": 0}
    orig_native_conv3d = cmpext3.ops.conv3d

    def counting_conv3d(*args, **kwargs):
        call_count["n"] += 1
        return orig_native_conv3d(*args, **kwargs)

    was_enabled = cmpext3.is_enabled()
    cmpext3.ops.conv3d = counting_conv3d
    try:
        if was_enabled:
            cmpext3.disable()
        cmpext3.enable()
        F.conv3d(x, weight, bias, stride=1, padding=1)
    finally:
        cmpext3.ops.conv3d = orig_native_conv3d
        if not was_enabled:
            cmpext3.disable()

    import cmpext3 as _cmpext3
    print(f"  native conv3d invocations: {call_count['n']} "
          f"(includes cmpext3/autoselect.py's one-time probe)")
    check.require(
        f"F.conv3d ({dtype_name}) invoked the native kernel, not a silent stock fallback",
        call_count["n"] >= 1,
        f"got {call_count['n']} native conv3d calls -- _patched_conv3d never reached the "
        "kernel at all (check groups/_usable/_grad_safe guards, CMPEXT3_DISABLE_OPS, and "
        "whether the native call is raising a RuntimeError).",
    )
    for decision in _cmpext3.autoselect.report():
        if decision.op.startswith("conv3d"):
            print(f"  autoselect kept: {decision!r}")


def check_winograd_path(check: Check, dev) -> None:
    """The Winograd conv3d kernel (src/cuda/fp32_conv3d_winograd.cu) is only
    used where custom_conv3d_forward MEASURED it faster than the direct
    fp32_conv3d.cu for that exact shape (see cmpext3_alt_kernel_faster in
    src/main.cpp) -- so "did it run at all" is a real question, and none of
    the checks above answer it: Check 2's small shape is one where Winograd
    deliberately loses and is not used, and Check 4 would pass either way.

    This uses a shape touched by no other check in this file, so the
    once-per-process benchmark line is guaranteed to be emitted here rather
    than already cached from an earlier call: 128 channels at 8x32x32,
    measured 1.49ms Winograd vs. 2.27ms direct on the CMP 50HX. (The
    crossover sits around 32-64 channels at this resolution -- below it the
    two extra transform passes and the U/V workspace cost more than the
    ~5.6x multiply saving is worth, which is exactly why the choice is
    benchmarked per shape instead of assumed.)

    fp32 only -- there is no fp16/bf16 Winograd kernel; bf16 reaches this
    one anyway by converting at the boundary (see custom_conv3d_forward).
    """
    import cmpext3
    import torch
    import torch.nn.functional as F
    print("\n" + "=" * 78)
    print("Check 5 [FP32]: the Winograd kernel is selected where it wins, and is correct there")
    print("=" * 78)

    b, c_in, d, h, w, c_out, k = 1, 128, 8, 32, 32, 128, 3
    x, weight, bias = _make_conv3d_inputs(b, c_in, d, h, w, c_out, k, torch.float32, dev, seed=7)

    with _capture_c_stderr() as captured:
        got = cmpext3.ops.conv3d(x, weight, bias, 1, 1)
    report = captured[0]
    sys.stderr.write(report)          # don't swallow it -- it's useful output
    sys.stderr.flush()

    line = next((ln for ln in report.splitlines()
                 if "Winograd" in ln and f"C{c_in} {d}x{h}x{w}" in ln), None)
    print(f"  kernel-choice report: {line if line else '(none emitted)'}")

    check.require(
        "conv3d fp32 benchmarked Winograd against the direct kernel for this shape",
        line is not None,
        "no benchmark line was printed for this shape -- either the Winograd launcher "
        "rejected it before timing (check fp32_conv3d_winograd.cu's guard: B=1, 3x3x3, "
        "stride/dilation 1, padding 1, even D/H/W, tile count and C_out both divisible "
        "by 8), or some earlier call in this process already cached the decision.",
    )
    check.require(
        "Winograd won on a shape it is supposed to win on (so it is actually in use)",
        line is not None and line.rstrip().endswith("-- using it"),
        "the direct kernel was measured faster here. Not automatically a bug -- the "
        "dispatch is doing its job by picking the winner -- but this shape was chosen "
        "because Winograd won it by ~1.5x, so a flip means fp32_conv3d_winograd.cu "
        "regressed (or fp32_conv3d.cu got much faster).",
    )

    want = _stock_conv3d(x.float(), weight.float(), bias.float(), 1, 1)
    max_abs_err = (got - want).abs().max().item()
    rel_err = max_abs_err / want.std().item()
    cos_sim = F.cosine_similarity(got.reshape(1, -1), want.reshape(1, -1)).item()
    print(f"  max abs error vs fp32 reference: {max_abs_err:.6f}  "
          f"(output std {want.std().item():.2f} -> {rel_err:.2e} relative)")
    print(f"  cosine similarity vs fp32 reference: {cos_sim:.7f}")
    # Bounded relative to the output scale rather than by a fixed absolute
    # number, because absolute error here grows with C_in for any kernel.
    # For reference, at this shape (measured against a float64 reference):
    # Winograd 1.5e-04, the direct kernel 7.1e-04, stock cuDNN 3.1e-04 --
    # i.e. Winograd is the most accurate of the three, not the least, so a
    # bound comfortably below the direct kernel's own error is still a fair
    # one. A real indexing or transform bug misses by orders of magnitude
    # more than this, not by a factor of two.
    check.require(
        "Winograd conv3d numerically matches the fp32 reference",
        cos_sim > 0.9999 and rel_err < 1e-4,
        f"cosine similarity {cos_sim:.7f} / relative error {rel_err:.2e} -- check the "
        "G/B^T/A^T transforms and the tile indexing in fp32_conv3d_winograd.cu.",
    )

    # Out-of-scope shape: stride 2 is outside the Winograd guard, so this must
    # fall through to fp32_conv3d.cu. Worth checking explicitly because the
    # failure mode of a bad guard isn't a crash -- it's a silently wrong
    # result from a kernel that assumed stride 1.
    x2, weight2, bias2 = _make_conv3d_inputs(1, 16, 8, 16, 16, 16, 3, torch.float32, dev, seed=9)
    got2 = cmpext3.ops.conv3d(x2, weight2, bias2, 2, 1)
    want2 = _stock_conv3d(x2, weight2, bias2, 2, 1)
    cos2 = F.cosine_similarity(got2.reshape(1, -1), want2.reshape(1, -1)).item()
    print(f"  stride-2 (out of Winograd scope) cosine similarity: {cos2:.7f}")
    check.require(
        "a stride-2 conv3d falls back to the direct kernel and stays correct",
        cos2 > 0.999,
        f"cosine similarity {cos2:.7f} -- fp32_conv3d_winograd.cu's guard let through a "
        "shape it cannot handle (it must return false, having written nothing, for "
        "anything but stride 1).",
    )


def check_speed_vs_stock(check: Check, dev, dtype_name: str, dtype, shape) -> None:
    import cmpext3
    import torch.nn.functional as F
    print("\n" + "=" * 78)
    print(f"Check 4 [{dtype_name}]: F.conv3d monkeypatch beats stock on the VAE-decode shape (S={shape})")
    print("=" * 78)

    b, c_in, d, h, w, c_out, k = shape
    x, weight, bias = _make_conv3d_inputs(b, c_in, d, h, w, c_out, k, dtype, dev)

    was_enabled = cmpext3.is_enabled()
    try:
        cmpext3.disable()
        _, stock_time = _timed(lambda: F.conv3d(x, weight, bias, stride=1, padding=1), warmup=3, iters=10)

        cmpext3.enable()
        _, patched_time = _timed(lambda: F.conv3d(x, weight, bias, stride=1, padding=1), warmup=3, iters=10)
    finally:
        cmpext3.disable()
        if was_enabled:
            cmpext3.enable()

    speedup = stock_time / patched_time if patched_time > 0 else float("inf")
    print(f"  stock F.conv3d (cuDNN):        {stock_time * 1000:9.3f} ms/iter")
    print(f"  patched F.conv3d ({dtype_name} kernel): {patched_time * 1000:9.3f} ms/iter")
    print(f"  speedup:                       {speedup:9.2f} x")

    # 0.98, not 1.0: cmpext3/autoselect.py guarantees "never materially slower
    # than stock", not "always faster" -- where the kernel loses or is rejected
    # for a shape, the patched path IS stock and the ratio sits at 1.00 plus
    # timing noise, which would make a >= 1.0 assertion a coin flip. That is
    # the case for fp16 here: the kernel is 2.3x faster at this shape but
    # accumulates in half precision, so at 512 channels it misses stock by 4.3%
    # and gets dropped on accuracy (see cmpext3/autoselect.py's _TOLERANCE).
    import cmpext3 as _cmpext3
    for decision in _cmpext3.autoselect.report():
        if decision.op.startswith("conv3d"):
            print(f"  autoselect kept: {decision!r}")
    check.require(
        f"{dtype_name} conv3d is never materially slower than stock cuDNN",
        speedup >= 0.98,
        f"patched conv3d ({patched_time*1000:.2f} ms) is slower than stock cuDNN "
        f"({stock_time*1000:.2f} ms). autoselect should have fallen back to stock for "
        "this shape rather than letting a losing kernel run -- check whether "
        "CMPEXT3_AUTOSELECT is switched off, or whether the probe measured a "
        "different shape than the one being timed here.",
    )


def run_dtype_suite(check: Check, dev, dtype_name: str, dtype) -> None:
    import torch
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print(f"\n[Info] {dtype_name} not supported on this device, skipping.")
        return
    small_shape = (1, 8, 4, 8, 8, 8, 3)         # b, c_in, d, h, w, c_out, k -- fast, for correctness checks
    real_shape = (1, 512, 8, 32, 32, 512, 3)    # bench.py's HunyuanVideo-style VAE-decode shape

    check_nan_inf(check, dev, dtype_name, dtype, small_shape)
    check_numerics(check, dev, dtype_name, dtype, small_shape)
    check_1x1x1_fastpath(check, dev, dtype_name, dtype)
    check_routes_to_native_kernel(check, dev, dtype_name, dtype, small_shape)
    if dtype == torch.float32:
        check_winograd_path(check, dev)
    check_speed_vs_stock(check, dev, dtype_name, dtype, real_shape)


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16", "all"], default="all")
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print("torch is not importable -- cannot run.")
        return 1

    try:
        import cmpext3  # noqa: F401
    except ImportError as exc:
        print(f"cmpext3 not importable ({exc}) -- build it first: pip install -e . --no-build-isolation")
        return 1

    check = Check()
    dev = check_environment(check)
    if dev is None:
        return _summarize(check)

    dtype_map = {
        "fp32": ("FP32", torch.float32),
        "fp16": ("FP16", torch.float16),
        "bf16": ("BF16", torch.bfloat16),
    }
    dtypes = dtype_map.values() if args.dtype == "all" else [dtype_map[args.dtype]]

    for dtype_name, dtype in dtypes:
        run_dtype_suite(check, dev, dtype_name, dtype)

    return _summarize(check)


def _summarize(check: "Check") -> int:
    print("\n" + "=" * 78)
    if check.failures:
        print(f"RESULT: {len(check.failures)} check(s) FAILED:")
        for label in check.failures:
            print(f"  - {label}")
        print("=" * 78)
        return 1
    print("RESULT: all checks passed.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
