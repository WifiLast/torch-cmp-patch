#!/usr/bin/env python3
"""
Standalone hardware regression check for cmpext3's fused GroupNorm+SiLU op
(cmpext3.ops.group_norm_silu, added to src/cuda/fp32_groupnorm.cu /
fp16_groupnorm.cu) -- no pytest required, just run it directly on the
machine with the CMP 50HX.

Unlike conv3d/conv_transpose2d, this isn't a monkeypatch target -- F.group_norm
has no activation argument to key a patch off of -- so there's no "does it
route to the native kernel" question the way test_conv3d_perf.py has; it's a
plain explicit op call, either it works or it raises. This script checks
numerics against the F.group_norm(...) + F.silu(...) pair it replaces, and
that fusing them is actually faster than the two separate calls.

Run on the machine with the actual GPU:

    python test_group_norm_silu.py
    python test_group_norm_silu.py --dtype fp16

Exits 0 if everything passes, 1 if any check fails or the environment can't
run it (no CUDA / cmpext3 not built).
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Callable, Tuple


def _make_gn_input(n, c, h, w, dtype, device, seed=0):
    import torch
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n, c, h, w, dtype=dtype, device=device, generator=g)
    weight = torch.randn(c, dtype=dtype, device=device, generator=g)
    bias = torch.randn(c, dtype=dtype, device=device, generator=g)
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


def check_nan_inf(check: Check, dev, dtype_name: str, dtype, shape, groups) -> None:
    import cmpext3
    import torch
    print("\n" + "=" * 78)
    print(f"Check 1 [{dtype_name}]: group_norm_silu output has no NaN/Inf")
    print("=" * 78)
    n, c, h, w = shape
    x, weight, bias = _make_gn_input(n, c, h, w, dtype, dev)
    out = cmpext3.ops.group_norm_silu(x, groups, weight, bias, 1e-5)
    finite = torch.isfinite(out).all().item()
    check.require(f"no NaN/Inf in {dtype_name} group_norm_silu output", finite,
                   "group_norm_silu produced NaN/Inf -- check the apply_silu epilogue in "
                   "fp32_groupnorm.cu/fp16_groupnorm.cu.")


def check_numerics(check: Check, dev, dtype_name: str, dtype, shape, groups) -> None:
    import cmpext3
    import torch
    import torch.nn.functional as F
    print("\n" + "=" * 78)
    print(f"Check 2 [{dtype_name}]: group_norm_silu matches F.group_norm(...) + F.silu(...)")
    print("=" * 78)
    n, c, h, w = shape
    x, weight, bias = _make_gn_input(n, c, h, w, dtype, dev)

    got = cmpext3.ops.group_norm_silu(x, groups, weight, bias, 1e-5).float()
    want = F.silu(F.group_norm(x.float(), groups, weight.float(), bias.float(), eps=1e-5))

    max_abs_err = (got - want).abs().max().item()
    cos_sim = F.cosine_similarity(got.reshape(1, -1), want.reshape(1, -1)).item()
    print(f"  max abs error vs F.group_norm+F.silu reference: {max_abs_err:.6f}")
    print(f"  cosine similarity vs reference: {cos_sim:.6f}")
    threshold = 0.999 if dtype == torch.float32 else 0.99
    check.require(
        f"{dtype_name} group_norm_silu numerically close to reference", cos_sim > threshold,
        f"cosine similarity {cos_sim:.4f} is below {threshold} -- check the SiLU formula "
        "(gn_silu_vec4/gn_silu_scalar in fp32_groupnorm.cu, gn_silu_h2/gn_silu_h1 in "
        "fp16_groupnorm.cu) matches F.silu's x*sigmoid(x).",
    )

    # Also verify the un-fused apply_silu=False path is unaffected (this op
    # shares one kernel with plain group_norm, gated by a bool -- make sure
    # adding the epilogue didn't change the base normalize+affine result).
    got_plain = cmpext3.ops.group_norm(x, groups, weight, bias, 1e-5).float()
    want_plain = F.group_norm(x.float(), groups, weight.float(), bias.float(), eps=1e-5)
    cos_sim_plain = F.cosine_similarity(got_plain.reshape(1, -1), want_plain.reshape(1, -1)).item()
    print(f"  (sanity) plain group_norm cosine similarity: {cos_sim_plain:.6f}")
    check.require(
        f"{dtype_name} plain group_norm (apply_silu=False) still matches reference", cos_sim_plain > threshold,
        f"cosine similarity {cos_sim_plain:.4f} is below {threshold} -- the apply_silu plumbing "
        "broke the existing group_norm path.",
    )


def check_speed_vs_unfused(check: Check, dev, dtype_name: str, dtype, shape, groups) -> None:
    import cmpext3
    import torch.nn.functional as F
    print("\n" + "=" * 78)
    print(f"Check 3 [{dtype_name}]: group_norm_silu beats separate group_norm+silu calls (S={shape})")
    print("=" * 78)
    n, c, h, w = shape
    x, weight, bias = _make_gn_input(n, c, h, w, dtype, dev)

    _, unfused_time = _timed(
        lambda: F.silu(cmpext3.ops.group_norm(x, groups, weight, bias, 1e-5)),
        warmup=5, iters=20)
    _, fused_time = _timed(
        lambda: cmpext3.ops.group_norm_silu(x, groups, weight, bias, 1e-5),
        warmup=5, iters=20)

    speedup = unfused_time / fused_time if fused_time > 0 else float("inf")
    print(f"  group_norm(...) then silu(...):  {unfused_time * 1000:9.3f} ms/iter")
    print(f"  group_norm_silu(...) (fused):    {fused_time * 1000:9.3f} ms/iter")
    print(f"  speedup:                         {speedup:9.2f} x")

    check.require(
        f"{dtype_name} fused group_norm_silu is at least as fast as the two-call baseline",
        speedup >= 1.0,
        f"fused ({fused_time*1000:.2f} ms) is slower than the two separate native calls "
        f"({unfused_time*1000:.2f} ms) -- the epilogue fusion isn't paying for itself.",
    )


def run_suite(check: Check, dev, dtype_name: str, dtype) -> None:
    import torch
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print(f"\n[Info] {dtype_name} not supported on this device, skipping.")
        return
    small_shape = (1, 32, 8, 8)          # n, c, h, w -- fast, for correctness checks
    real_shape = (64, 512, 128, 128)     # bench.py's GroupNorm shape
    groups = 32

    check_nan_inf(check, dev, dtype_name, dtype, small_shape, groups)
    check_numerics(check, dev, dtype_name, dtype, small_shape, groups)
    check_speed_vs_unfused(check, dev, dtype_name, dtype, real_shape, groups)


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
        run_suite(check, dev, dtype_name, dtype)

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
