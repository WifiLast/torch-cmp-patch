#!/usr/bin/env python3
"""
Standalone hardware regression check for cmpext3's rewritten conv_transpose2d
kernels (src/cuda/fp32_ConvTranspose2d.cu, src/cuda/fp16_ConvTranspose2d.cu)
-- no pytest required, just run it directly on the machine with the CMP 50HX.

Mirrors test_conv3d_perf.py's structure and reasoning. Background: the
previous conv_transpose2d kernels were one-thread-per-(output-pixel,
1-or-2-output-channels) with no register/pixel-level reuse, which is why
conv_transpose2d has been gated behind CMPEXT3_ENABLE_UNVERIFIED_KERNELS=1
rather than enabled by default (see cmpext3/__init__.py's enable() and
README.md). custom_upsample_smart (F.interpolate(mode='nearest')) is built
directly on top of conv_transpose2d (an all-ones kernel + matching stride),
so this same rewrite covers both ops -- both are exercised below.

Run on the machine with the actual GPU:

    python test_conv_transpose2d_perf.py
    python test_conv_transpose2d_perf.py --dtype fp16   # just one precision

Exits 0 if everything passes, 1 if any check fails or the environment can't
run it (no CUDA / cmpext3 not built).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Callable, Tuple

# conv_transpose2d/interpolate's F.* monkeypatch is opt-in; the checks below
# need it engaged so they can observe the actual routing/perf behavior.
os.environ.setdefault("CMPEXT3_ENABLE_UNVERIFIED_KERNELS", "1")


def _make_convT_inputs(b, c_in, h, w, c_out, k, dtype, device, seed=0):
    import torch
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(b, c_in, h, w, dtype=dtype, device=device, generator=g)
    # ConvTranspose2d weight layout: [C_in, C_out, K_H, K_W]
    weight = torch.randn(c_in, c_out, k, k, dtype=dtype, device=device, generator=g)
    bias = torch.randn(c_out, dtype=dtype, device=device, generator=g)
    return x, weight, bias


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


# ---------------------------------------------------------------------------
# ConvTranspose2d checks
# ---------------------------------------------------------------------------

def check_nan_inf(check: Check, dev, dtype_name: str, dtype, shape) -> None:
    import cmpext3
    import torch
    print("\n" + "=" * 78)
    print(f"Check 1 [{dtype_name}]: conv_transpose2d output has no NaN/Inf")
    print("=" * 78)
    b, c_in, h, w, c_out, k = shape
    x, weight, bias = _make_convT_inputs(b, c_in, h, w, c_out, k, dtype, dev)
    out = cmpext3.ops.conv_transpose2d(x, weight, bias, 2, 1, 1)
    finite = torch.isfinite(out).all().item()
    check.require(f"no NaN/Inf in {dtype_name} conv_transpose2d output", finite,
                   "conv_transpose2d produced NaN/Inf -- check fp32_ConvTranspose2d.cu/fp16_ConvTranspose2d.cu.")


def check_numerics(check: Check, dev, dtype_name: str, dtype, shape) -> None:
    import cmpext3
    import torch
    import torch.nn.functional as F
    print("\n" + "=" * 78)
    print(f"Check 2 [{dtype_name}]: conv_transpose2d matches an fp32 F.conv_transpose2d reference")
    print("=" * 78)
    b, c_in, h, w, c_out, k = shape
    x, weight, bias = _make_convT_inputs(b, c_in, h, w, c_out, k, dtype, dev)

    got = cmpext3.ops.conv_transpose2d(x, weight, bias, 2, 1, 1).float()
    want = F.conv_transpose2d(x.float(), weight.float(), bias.float(),
                               stride=2, padding=1, output_padding=1)

    max_abs_err = (got - want).abs().max().item()
    cos_sim = F.cosine_similarity(got.reshape(1, -1), want.reshape(1, -1)).item()
    print(f"  max abs error vs fp32 reference: {max_abs_err:.6f}")
    print(f"  cosine similarity vs fp32 reference: {cos_sim:.6f}")
    threshold = 0.999 if dtype == torch.float32 else 0.99
    check.require(
        f"{dtype_name} conv_transpose2d numerically close to fp32 reference", cos_sim > threshold,
        f"cosine similarity {cos_sim:.4f} is below {threshold} -- likely an indexing/weight-layout "
        "bug (weight is [C_in, C_out, K_H, K_W], not [C_out, C_in, K_H, K_W] -- see the kernel header "
        "comment for why that matters here).",
    )


def check_routes_to_native_kernel(check: Check, dev, dtype_name: str, dtype, shape) -> None:
    """Same rationale as test_conv3d_perf.py's routing check: _patched_conv_transpose2d
    falls back to stock on any RuntimeError, so a broken kernel could pass
    correctness by silently never running. Count native-kernel invocations
    while F.conv_transpose2d is patched.
    """
    import cmpext3
    import torch.nn.functional as F
    print("\n" + "=" * 78)
    print(f"Check 3 [{dtype_name}]: F.conv_transpose2d monkeypatch actually reaches the native kernel")
    print("=" * 78)

    b, c_in, h, w, c_out, k = shape
    x, weight, bias = _make_convT_inputs(b, c_in, h, w, c_out, k, dtype, dev)

    call_count = {"n": 0}
    orig_native = cmpext3.ops.conv_transpose2d

    def counting_conv_transpose2d(*args, **kwargs):
        call_count["n"] += 1
        return orig_native(*args, **kwargs)

    was_enabled = cmpext3.is_enabled()
    cmpext3.ops.conv_transpose2d = counting_conv_transpose2d
    try:
        if was_enabled:
            cmpext3.disable()
        cmpext3.enable()
        F.conv_transpose2d(x, weight, bias, stride=2, padding=1, output_padding=1)
    finally:
        cmpext3.ops.conv_transpose2d = orig_native
        if not was_enabled:
            cmpext3.disable()

    check.require(
        f"F.conv_transpose2d ({dtype_name}) invoked the native kernel, not a silent stock fallback",
        call_count["n"] == 1,
        f"expected exactly 1 native conv_transpose2d call, got {call_count['n']} -- if 0, "
        "_patched_conv_transpose2d is falling back to stock.",
    )


def check_speed_vs_stock(check: Check, dev, dtype_name: str, dtype, shape) -> None:
    import cmpext3
    import torch.nn.functional as F
    print("\n" + "=" * 78)
    print(f"Check 4 [{dtype_name}]: F.conv_transpose2d monkeypatch beats stock (S={shape})")
    print("=" * 78)

    b, c_in, h, w, c_out, k = shape
    x, weight, bias = _make_convT_inputs(b, c_in, h, w, c_out, k, dtype, dev)

    was_enabled = cmpext3.is_enabled()
    try:
        cmpext3.disable()
        _, stock_time = _timed(
            lambda: F.conv_transpose2d(x, weight, bias, stride=2, padding=1, output_padding=1),
            warmup=3, iters=10)

        cmpext3.enable()
        _, patched_time = _timed(
            lambda: F.conv_transpose2d(x, weight, bias, stride=2, padding=1, output_padding=1),
            warmup=3, iters=10)
    finally:
        cmpext3.disable()
        if was_enabled:
            cmpext3.enable()

    speedup = stock_time / patched_time if patched_time > 0 else float("inf")
    print(f"  stock F.conv_transpose2d (cuDNN):        {stock_time * 1000:9.3f} ms/iter")
    print(f"  patched F.conv_transpose2d ({dtype_name} kernel): {patched_time * 1000:9.3f} ms/iter")
    print(f"  speedup:                                 {speedup:9.2f} x")

    check.require(
        f"{dtype_name} conv_transpose2d is at least as fast as stock cuDNN",
        speedup >= 1.0,
        f"patched conv_transpose2d ({patched_time*1000:.2f} ms) is slower than stock cuDNN "
        f"({stock_time*1000:.2f} ms).",
    )


def run_convT_suite(check: Check, dev, dtype_name: str, dtype) -> None:
    import torch
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print(f"\n[Info] {dtype_name} not supported on this device, skipping.")
        return
    small_shape = (1, 8, 8, 8, 8, 3)          # b, c_in, h, w, c_out, k -- fast, for correctness checks
    real_shape = (64, 64, 64, 64, 64, 3)      # bench.py's ConvTranspose2d 2x-upsample shape

    check_nan_inf(check, dev, dtype_name, dtype, small_shape)
    check_numerics(check, dev, dtype_name, dtype, small_shape)
    check_routes_to_native_kernel(check, dev, dtype_name, dtype, small_shape)
    check_speed_vs_stock(check, dev, dtype_name, dtype, real_shape)


# ---------------------------------------------------------------------------
# F.interpolate(mode='nearest') checks -- built on the same ConvTranspose2d
# kernel via custom_upsample_smart (main.cpp).
# ---------------------------------------------------------------------------

def check_interpolate(check: Check, dev, dtype_name: str, dtype) -> None:
    import cmpext3
    import torch
    import torch.nn.functional as F
    print("\n" + "=" * 78)
    print(f"Check 5 [{dtype_name}]: F.interpolate(mode='nearest') numerics + speed (built on conv_transpose2d)")
    print("=" * 78)

    g = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(4, 320, 32, 32, dtype=dtype, device=dev, generator=g)
    target = (64, 64)  # 2x upsample -- common VAE-decoder pattern

    got = cmpext3.ops.upsample_scaling(x, target).float()
    want = F.interpolate(x.float(), size=target, mode="nearest")
    cos_sim = F.cosine_similarity(got.reshape(1, -1), want.reshape(1, -1)).item()
    print(f"  cosine similarity vs F.interpolate reference: {cos_sim:.6f}")
    check.require(f"{dtype_name} upsample_scaling matches F.interpolate", cos_sim > 0.999,
                  f"cosine similarity {cos_sim:.4f} too low.")

    was_enabled = cmpext3.is_enabled()
    try:
        cmpext3.disable()
        _, stock_time = _timed(lambda: F.interpolate(x, size=target, mode="nearest"), warmup=3, iters=10)
        cmpext3.enable()
        _, patched_time = _timed(lambda: F.interpolate(x, size=target, mode="nearest"), warmup=3, iters=10)
    finally:
        cmpext3.disable()
        if was_enabled:
            cmpext3.enable()

    speedup = stock_time / patched_time if patched_time > 0 else float("inf")
    print(f"  stock F.interpolate:        {stock_time * 1000:9.3f} ms/iter")
    print(f"  patched F.interpolate ({dtype_name}): {patched_time * 1000:9.3f} ms/iter")
    print(f"  speedup:                    {speedup:9.2f} x")
    check.require(f"{dtype_name} F.interpolate is at least as fast as stock", speedup >= 1.0,
                  f"patched ({patched_time*1000:.2f} ms) is slower than stock ({stock_time*1000:.2f} ms).")


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
        run_convT_suite(check, dev, dtype_name, dtype)
        if dtype in (torch.float32, torch.float16):
            # upsample_scaling only ever runs fp32/fp16 natively (bf16 input
            # would hit the same bf16->fp32 boundary conversion as conv3d).
            check_interpolate(check, dev, dtype_name, dtype)

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
