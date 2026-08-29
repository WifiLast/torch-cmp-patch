#!/usr/bin/env python3
"""
Standalone hardware regression check for cmpext3's bf16 attention routing
fix -- no pytest required, just run it directly.

Unlike tests/test_cmpext3_monkeypatch.py (which stubs the native extension
with MagicMocks so *dispatch logic* can be tested on any machine), this
script needs a real GPU and the actual compiled cmpext3._native extension --
it exists specifically to catch the class of bug this repo already hit
once: custom_attention_forward's bf16 boundary conversion silently routing
through the wrong dtype kernel.

Background: fp32_attention.cu's kernel is a naive one-thread-per-query-row
implementation with no warp-level cooperation across the head-dim
reduction -- measured at ~21.6s for a single S=17402 attention call.
fp16_attention.cu's kernel is properly warp-cooperative and tiled --
measured at ~6.5ms for the same shape (it's the same design later forked
into the sageattention/xformers CMP-Turing ports). bf16 must convert to
fp16, not fp32, or every bf16 attention call (e.g. an entire bf16 diffusion
transformer) silently eats the ~3000x slowdown.

Run on the machine with the actual GPU:

    python test_attention_bf16_routing.py
    python test_attention_bf16_routing.py --seqlen 4096   # use a bigger, slower-to-prove shape

Exits 0 if everything passes, 1 if any check fails or the environment
can't run it (no CUDA / cmpext3 not built).
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Callable, Tuple


def _make_qkv(b, h, s, d, dtype, device, seed=0):
    import torch
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(b, h, s, d, dtype=dtype, device=device, generator=g)
    k = torch.randn(b, h, s, d, dtype=dtype, device=device, generator=g)
    v = torch.randn(b, h, s, d, dtype=dtype, device=device, generator=g)
    return q, k, v


def _reference_attention(q, k, v, scale: float):
    """Exact softmax attention in fp32 -- ground truth for the bf16/fp16 kernel path."""
    import torch
    q32, k32, v32 = q.float(), k.float(), v.float()
    scores = torch.matmul(q32, k32.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v32)


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


def print_nvml_diagnostics() -> None:
    """Informational only (no pass/fail) -- reports what NVML itself can
    see about clocks/power/throttling. NVML cannot see the CMP driver's
    FFMA/Tensor-Core instruction-pattern throttle (that's invisible to it
    by design), so a clean report here does NOT mean the card isn't being
    throttled by that separate mechanism -- it only rules out a
    power/thermal/sw cap on top of it. See cmpext3/nvml_boost.py.
    """
    try:
        import cmpext3.nvml_boost as nvml_boost
    except ImportError:
        return
    print("\n" + "=" * 78)
    print("NVML diagnostics (informational -- see cmpext3/nvml_boost.py for scope)")
    print("=" * 78)
    if not nvml_boost.is_available():
        print("  pynvml not available -- skipping (see source/other/nvml/pynvml.py)")
        return
    info = nvml_boost.query_throttle_reasons()
    if not info:
        print("  could not query NVML (see warning above, if any)")
        return
    print(f"  SM clock: {info['sm_clock_mhz']} MHz (max {info['max_sm_clock_mhz']} MHz)")
    print(f"  power: {info['power_usage_w']:.1f} W (limit {info['power_limit_w']:.1f} W, "
          f"board max {info['power_limit_max_w']:.1f} W)")
    print(f"  GPU utilization: {info['gpu_util_pct']}%")
    active = info["throttle_reasons_active"]
    if active:
        print(f"  NVML-visible throttle reasons active: {sorted(active.keys())}")
    else:
        print("  no NVML-visible throttle reasons active (does not rule out the "
              "FFMA/Tensor-Core instruction-pattern throttle -- NVML can't see that one)")


def check_nan_inf(check: Check, dev, seqlen: int, head_dim: int) -> None:
    import cmpext3
    import torch
    print("\n" + "=" * 78)
    print("Check 1: bf16 attention output has no NaN/Inf")
    print("=" * 78)
    q, k, v = _make_qkv(1, 4, seqlen, head_dim, torch.bfloat16, dev)
    scale = head_dim ** -0.5
    out = cmpext3.ops.attention(q, k, v, scale)
    finite = torch.isfinite(out).all().item()
    check.require("no NaN/Inf in bf16 attention output", finite,
                   "bf16 attention produced NaN/Inf -- see the earlier NaN debugging session.")


def check_numerics(check: Check, dev, head_dim: int) -> None:
    import cmpext3
    import torch
    import torch.nn.functional as F
    print("\n" + "=" * 78)
    print("Check 2: bf16 attention matches an fp32 reference (numerics, not speed)")
    print("=" * 78)
    # Smaller S here -- this checks correctness, not performance, and the
    # fp32 reference itself is a plain O(S^2) matmul we don't want to wait on.
    q, k, v = _make_qkv(1, 4, 512, head_dim, torch.bfloat16, dev)
    scale = head_dim ** -0.5
    got = cmpext3.ops.attention(q, k, v, scale).float()
    want = _reference_attention(q, k, v, scale)
    cos_sim = F.cosine_similarity(got.reshape(1, -1), want.reshape(1, -1)).item()
    print(f"  cosine similarity vs fp32 reference: {cos_sim:.6f}")
    check.require(
        "bf16 attention numerically close to fp32 reference", cos_sim > 0.99,
        f"cosine similarity {cos_sim:.4f} is too low -- expected close but not exact "
        "given the bf16->fp16->bf16 round trip.",
    )


def check_bf16_routes_to_fast_kernel(check: Check, dev, seqlen: int, head_dim: int) -> None:
    import cmpext3
    import torch
    print("\n" + "=" * 78)
    print(f"Check 3: bf16 routes to the fast (fp16) kernel, not the naive fp32 one  (S={seqlen})")
    print("=" * 78)
    q, k, v = _make_qkv(1, 2, seqlen, head_dim, torch.bfloat16, dev)
    scale = head_dim ** -0.5

    _, bf16_time = _timed(lambda: cmpext3.ops.attention(q, k, v, scale), warmup=2, iters=5)

    # Deliberately invoke the naive fp32 kernel directly (not through the
    # bf16 boundary) to get a *live* "how slow is the slow path on this
    # exact machine/shape" baseline, instead of hardcoding a number from a
    # different run.
    q32, k32, v32 = q.float(), k.float(), v.float()
    _, fp32_time = _timed(lambda: cmpext3.ops.attention(q32, k32, v32, scale), warmup=0, iters=1)

    print(f"  bf16 (should route through fp16_attention.cu): {bf16_time * 1000:9.2f} ms/iter")
    print(f"  fp32 (naive kernel, deliberately invoked):     {fp32_time * 1000:9.2f} ms/iter")

    check.require(
        f"bf16 attention completes in under 1s at S={seqlen}",
        bf16_time < 1.0,
        f"bf16 attention took {bf16_time:.3f}s/iter -- if this regressed back to routing "
        "through the fp32 kernel, it would be in the hundreds-of-ms-to-seconds range.",
    )
    check.require(
        "bf16 is at least 20x faster than the explicit fp32 path",
        bf16_time < fp32_time / 20,
        f"bf16 path ({bf16_time * 1000:.2f} ms) is not dramatically faster than the explicit "
        f"fp32 path ({fp32_time * 1000:.2f} ms) at the same shape. This is the exact regression "
        "that made F.scaled_dot_product_attention take 21.6s at S=17402: bf16 silently routing "
        "through fp32_attention.cu's naive kernel instead of fp16_attention.cu's warp-cooperative "
        "one. Check custom_attention_forward's bf16 guard in src/main.cpp converts to "
        "torch::kFloat16, not torch::kFloat32.",
    )


def check_sdpa_patch(check: Check, dev, seqlen: int, head_dim: int) -> None:
    import cmpext3
    import torch
    import torch.nn.functional as F
    print("\n" + "=" * 78)
    print("Check 4: the actual F.scaled_dot_product_attention monkeypatch is fast for bf16")
    print("=" * 78)
    was_enabled = cmpext3.is_enabled()
    if not was_enabled:
        cmpext3.enable()
    try:
        q, k, v = _make_qkv(1, 4, seqlen, head_dim, torch.bfloat16, dev)
        _, elapsed = _timed(lambda: F.scaled_dot_product_attention(q, k, v), warmup=2, iters=5)
        print(f"  F.scaled_dot_product_attention (bf16, patched): {elapsed * 1000:9.2f} ms/iter")
        check.require(
            f"patched SDPA completes in under 1s at S={seqlen}",
            elapsed < 1.0,
            f"F.scaled_dot_product_attention took {elapsed:.3f}s/iter for bf16 at S={seqlen} -- "
            "this is the original symptom (measured at 21.6s for S=17402 before the fix).",
        )
    finally:
        if not was_enabled:
            cmpext3.disable()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seqlen", type=int, default=2048,
                         help="Sequence length for the speed checks. Large enough to make the "
                              "naive fp32 kernel's cost obvious without being painfully slow. "
                              "Use your real sequence length (e.g. 17402) to reproduce exactly.")
    parser.add_argument("--head-dim", type=int, default=128)
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
        _summarize(check)
        return 1

    print_nvml_diagnostics()

    check_nan_inf(check, dev, args.seqlen, args.head_dim)
    check_numerics(check, dev, args.head_dim)
    check_bf16_routes_to_fast_kernel(check, dev, args.seqlen, args.head_dim)
    check_sdpa_patch(check, dev, args.seqlen, args.head_dim)

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
