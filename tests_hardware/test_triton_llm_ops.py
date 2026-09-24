#!/usr/bin/env python3
"""
Standalone hardware regression check for cmpext3.rope_ops and
cmpext3.fused_norm_ops -- no pytest required, just run it directly.

Both are plain-Triton ports of other/amd_tools's RDNA3 tooling (see each
module's own docstring for exactly what changed and why they aren't
actually ROCm-specific despite that origin). Needs a real GPU, a working
Triton install, and the actual compiled cmpext3._native extension (for the
F.rms_norm/scaled_dot_product_attention comparisons below).

Run on the machine with the actual GPU:

    python test_triton_llm_ops.py

Exits 0 if everything passes, 1 if any check fails or the environment
can't run it (no CUDA / triton not importable / cmpext3 not built).
"""
from __future__ import annotations

import sys


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
    try:
        import triton  # noqa: F401
        print(f"triton: {triton.__version__}")
    except ImportError:
        print("triton: not importable")
    return dev


def check_rope_numerics(check: Check, dev, dtype) -> None:
    import torch
    from cmpext3 import rope_ops

    print("\n" + "=" * 78)
    print(f"Check: rope_ops.rotary_embedding matches a manual rotate-half reference  (dtype={dtype})")
    print("=" * 78)
    if not rope_ops.available():
        check.fail("rope_ops.available()", "triton is not importable -- skipping rope_ops checks.")
        return

    torch.manual_seed(0)
    num_tokens, num_heads, num_kv_heads, head_size = 17, 4, 2, 64
    base, max_pos = 10000.0, 64

    q = torch.randn(num_tokens, num_heads * head_size, dtype=dtype, device=dev)
    k = torch.randn(num_tokens, num_kv_heads * head_size, dtype=dtype, device=dev)
    positions = torch.arange(num_tokens, device=dev)
    cache = rope_ops.compute_cos_sin_cache(base, head_size, max_pos).to(dev)

    q_ref, k_ref = q.clone(), k.clone()

    def manual_neox_rope(x, n_heads):
        # x: [num_tokens, n_heads*head_size] -> per-head rotate-half, same
        # convention as rope_ops' kernel (cos/sin laid out as [cos|sin]).
        xh = x.view(num_tokens, n_heads, head_size).float()
        cos = cache[positions, : head_size // 2].float()
        sin = cache[positions, head_size // 2 :].float()
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        x1, x2 = xh[..., : head_size // 2], xh[..., head_size // 2 :]
        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin
        return torch.cat([out1, out2], dim=-1).to(x.dtype).view(num_tokens, n_heads * head_size)

    q_want = manual_neox_rope(q_ref, num_heads)
    k_want = manual_neox_rope(k_ref, num_kv_heads)

    with torch.no_grad():
        q_got, k_got = rope_ops.rotary_embedding(positions, q, k, head_size, cache)

    q_cos = torch.nn.functional.cosine_similarity(q_got.float().reshape(1, -1), q_want.float().reshape(1, -1)).item()
    k_cos = torch.nn.functional.cosine_similarity(k_got.float().reshape(1, -1), k_want.float().reshape(1, -1)).item()
    print(f"  query cosine similarity vs manual rotate-half reference: {q_cos:.6f}")
    print(f"  key   cosine similarity vs manual rotate-half reference: {k_cos:.6f}")
    check.require(f"rope_ops query matches reference ({dtype})", q_cos > 0.999,
                   f"query cosine similarity {q_cos:.4f} too low.")
    check.require(f"rope_ops key matches reference ({dtype})", k_cos > 0.999,
                   f"key cosine similarity {k_cos:.4f} too low.")
    check.require("rotary_embedding mutated query/key in place", q_got is q and k_got is k,
                   "rotary_embedding should return the same tensors it was given (in-place contract), "
                   "except on the bf16/Turing upcast path where a copy-back is expected instead.")


def check_fused_norm_numerics(check: Check, dev, dtype) -> None:
    import torch
    import torch.nn.functional as F
    from cmpext3 import fused_norm_ops

    print("\n" + "=" * 78)
    print(f"Check: fused_norm_ops.fused_add_rms_norm matches unfused add+F.rms_norm  (dtype={dtype})")
    print("=" * 78)
    if not fused_norm_ops.available():
        check.fail("fused_norm_ops.available()", "triton is not importable -- skipping fused_norm_ops checks.")
        return
    if not hasattr(F, "rms_norm"):
        check.fail("F.rms_norm available", "this torch build has no F.rms_norm -- can't build a reference.")
        return

    torch.manual_seed(0)
    rows, hidden = 23, 256
    eps = 1e-6
    x = torch.randn(rows, hidden, dtype=dtype, device=dev)
    residual = torch.randn(rows, hidden, dtype=dtype, device=dev)
    weight = torch.randn(hidden, dtype=dtype, device=dev)

    want_residual = residual + x
    want_out = F.rms_norm(want_residual.float(), (hidden,), weight.float(), eps).to(dtype)

    got_out, got_residual = fused_norm_ops.fused_add_rms_norm(x, residual, weight, eps)

    out_cos = F.cosine_similarity(got_out.float().reshape(1, -1), want_out.float().reshape(1, -1)).item()
    res_max_diff = (got_residual.float() - want_residual.float()).abs().max().item()
    print(f"  normed-output cosine similarity vs reference: {out_cos:.6f}")
    print(f"  new-residual max abs diff vs reference:        {res_max_diff:.2e}")
    check.require(f"fused_add_rms_norm output matches reference ({dtype})", out_cos > 0.999,
                   f"cosine similarity {out_cos:.4f} too low.")
    check.require(f"fused_add_rms_norm new residual matches reference ({dtype})", res_max_diff < 1e-2,
                   f"max abs diff {res_max_diff:.2e} too high for the plain residual add.")


def check_fused_norm_backward(check: Check, dev) -> None:
    import torch
    from cmpext3 import fused_norm_ops

    print("\n" + "=" * 78)
    print("Check: fused_add_rms_norm backward runs and produces finite gradients")
    print("=" * 78)
    if not fused_norm_ops.available():
        check.fail("fused_norm_ops.available()", "triton is not importable -- skipping.")
        return

    torch.manual_seed(0)
    rows, hidden = 8, 128
    x = torch.randn(rows, hidden, dtype=torch.float32, device=dev, requires_grad=True)
    residual = torch.randn(rows, hidden, dtype=torch.float32, device=dev, requires_grad=True)
    weight = torch.randn(hidden, dtype=torch.float32, device=dev, requires_grad=True)

    out, new_residual = fused_norm_ops.fused_add_rms_norm(x, residual, weight)
    loss = out.sum() + new_residual.sum()
    loss.backward()

    grads_finite = all(
        t.grad is not None and torch.isfinite(t.grad).all().item() for t in (x, residual, weight)
    )
    check.require("gradients for x/residual/weight are present and finite", grads_finite,
                   "one of x.grad/residual.grad/weight.grad is missing or contains NaN/Inf -- "
                   "check the backward kernel's HAS_RESIDUAL_GRAD path and atomic_add accumulation.")


def main() -> int:
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

    import torch

    check = Check()
    dev = check_environment(check)
    if dev is None:
        return _summarize(check)

    for dtype in (torch.float16, torch.float32, torch.bfloat16):
        check_rope_numerics(check, dev, dtype)
    for dtype in (torch.float16, torch.float32, torch.bfloat16):
        check_fused_norm_numerics(check, dev, dtype)
    check_fused_norm_backward(check, dev)

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
