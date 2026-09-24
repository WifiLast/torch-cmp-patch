#!/usr/bin/env python3
"""
Standalone hardware regression check for cmpext3's causal attention support
-- no pytest required, just run it directly.

Needs a real GPU and the actual compiled cmpext3._native extension, like
test_attention_bf16_routing.py.

Background: F.scaled_dot_product_attention(..., is_causal=True) used to bail
straight to stock PyTorch in cmpext3/__init__.py's _patched_sdpa -- the
native fp16_attention.cu kernel had no masking logic, only the plain
(non-causal) case. That meant cmpext3 never engaged at all for LLM-style
decoder attention (e.g. a Qwen3-backbone model), which calls SDPA with
is_causal=True almost exclusively. Causal masking was ported into
fp16_attention.cu's tile loop from sageattention's turing_fma_free fork of
this same kernel (other/amd_tools/source/sageattention-1.0.6/sageattention/
csrc/turing_fma_free/attention_fp16_turing.cu), which added it independently
for the same reason (SageAttention's sageattn() exposes is_causal).

Run on the machine with the actual GPU:

    python test_attention_causal.py
    python test_attention_causal.py --seqlen 4096

Exits 0 if everything passes, 1 if any check fails or the environment can't
run it (no CUDA / cmpext3 not built).
"""
from __future__ import annotations

import argparse
import sys
from typing import Callable, Tuple


def _make_qkv(b, h, s, d, dtype, device, seed=0):
    import torch
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(b, h, s, d, dtype=dtype, device=device, generator=g)
    k = torch.randn(b, h, s, d, dtype=dtype, device=device, generator=g)
    v = torch.randn(b, h, s, d, dtype=dtype, device=device, generator=g)
    return q, k, v


def _reference_causal_attention(q, k, v, scale: float):
    """Exact causal softmax attention in fp32 -- ground truth for the kernel."""
    import torch
    q32, k32, v32 = q.float(), k.float(), v.float()
    s = q32.shape[-2]
    scores = torch.matmul(q32, k32.transpose(-2, -1)) * scale
    mask = torch.triu(torch.ones(s, s, dtype=torch.bool, device=q.device), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v32)


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


def check_causal_numerics(check: Check, dev, dtype, head_dim: int, seqlen: int) -> None:
    import cmpext3
    import torch
    import torch.nn.functional as F
    print("\n" + "=" * 78)
    print(f"Check: causal ops.attention matches an fp32 causal reference  "
          f"(dtype={dtype}, S={seqlen})")
    print("=" * 78)
    q, k, v = _make_qkv(1, 4, seqlen, head_dim, dtype, dev)
    scale = head_dim ** -0.5
    got = cmpext3.ops.attention(q, k, v, scale, True).float()
    want = _reference_causal_attention(q, k, v, scale)
    cos_sim = F.cosine_similarity(got.reshape(1, -1), want.reshape(1, -1)).item()
    print(f"  cosine similarity vs fp32 causal reference: {cos_sim:.6f}")
    check.require(
        f"causal attention ({dtype}) numerically close to fp32 causal reference", cos_sim > 0.99,
        f"cosine similarity {cos_sim:.4f} is too low -- future keys are leaking into the "
        "softmax (masking bug in fp16_attention.cu's causal tile-skip / per-j guard), or "
        "the causal path silently computed full (non-causal) attention instead.",
    )


def check_causal_differs_from_noncausal(check: Check, dev, head_dim: int, seqlen: int) -> None:
    import cmpext3
    import torch
    print("\n" + "=" * 78)
    print("Check: causal output actually differs from non-causal output")
    print("=" * 78)
    q, k, v = _make_qkv(1, 4, seqlen, head_dim, torch.float16, dev)
    scale = head_dim ** -0.5
    causal_out = cmpext3.ops.attention(q, k, v, scale, True)
    plain_out = cmpext3.ops.attention(q, k, v, scale, False)
    max_diff = (causal_out.float() - plain_out.float()).abs().max().item()
    check.require(
        "causal and non-causal outputs differ (masking actually engaged)",
        max_diff > 1e-3,
        f"max abs diff between causal and non-causal output was only {max_diff:.2e} -- "
        "suggests the `causal` flag never reached the kernel launch (e.g. a stale binding "
        "still ignoring the argument), not that masking is merely subtle.",
    )


def check_fp32_causal_falls_back(check: Check, dev, head_dim: int) -> None:
    """fp32_attention.cu has no causal-masking logic (see its TORCH_CHECK guard
    in custom_attention_forward) -- a direct ops.attention(..., causal=True)
    call on fp32 tensors must raise, not silently return wrong (non-causal)
    output, so _patched_sdpa's fallback-to-stock contract stays correct."""
    import cmpext3
    import torch
    print("\n" + "=" * 78)
    print("Check: fp32 + causal raises (no silent wrong answer)")
    print("=" * 78)
    q, k, v = _make_qkv(1, 2, 64, head_dim, torch.float32, dev)
    scale = head_dim ** -0.5
    raised = False
    try:
        cmpext3.ops.attention(q, k, v, scale, True)
    except RuntimeError:
        raised = True
    check.require(
        "fp32 causal attention raises RuntimeError instead of computing a wrong answer",
        raised,
        "cmpext3.ops.attention(fp32, causal=True) did not raise -- either the fp32 kernel "
        "silently gained (untested) causal support, or the TORCH_CHECK(!causal) guard in "
        "custom_attention_forward (src/main.cpp) was removed/bypassed.",
    )


def check_sdpa_patch_is_causal(check: Check, dev, head_dim: int, seqlen: int) -> None:
    import cmpext3
    import torch
    import torch.nn.functional as F
    print("\n" + "=" * 78)
    print("Check: F.scaled_dot_product_attention(is_causal=True) is now patched, not bypassed")
    print("=" * 78)
    was_enabled = cmpext3.is_enabled()
    if not was_enabled:
        cmpext3.enable()
    try:
        q, k, v = _make_qkv(1, 4, seqlen, head_dim, torch.float16, dev)
        got = F.scaled_dot_product_attention(q, k, v, is_causal=True).float()
        want = _reference_causal_attention(q, k, v, head_dim ** -0.5)
        cos_sim = F.cosine_similarity(got.reshape(1, -1), want.reshape(1, -1)).item()
        print(f"  cosine similarity vs fp32 causal reference: {cos_sim:.6f}")
        check.require(
            "patched SDPA with is_causal=True matches the causal reference", cos_sim > 0.99,
            f"cosine similarity {cos_sim:.4f} -- _patched_sdpa is either still bailing to "
            "stock for is_causal (check the `is_causal` bail-out condition was removed from "
            "cmpext3/__init__.py's _patched_sdpa) or the kernel path is wrong.",
        )
    finally:
        if not was_enabled:
            cmpext3.disable()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seqlen", type=int, default=512,
                         help="Sequence length for the numerics checks. Kept small since the "
                              "fp32 reference is a plain O(S^2) matmul.")
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()

    try:
        import torch
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

    check_causal_numerics(check, dev, torch.float16, args.head_dim, args.seqlen)
    check_causal_numerics(check, dev, torch.bfloat16, args.head_dim, args.seqlen)
    check_causal_differs_from_noncausal(check, dev, args.head_dim, args.seqlen)
    check_fp32_causal_falls_back(check, dev, args.head_dim)
    check_sdpa_patch_is_causal(check, dev, args.head_dim, args.seqlen)

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
