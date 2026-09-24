#!/usr/bin/env python3
"""
Standalone hardware regression check for cmpext3.fftconv_ops -- no pytest
required, just run it directly.

Ported from other/amd_tools's amd_tuned_torch.fftconv_ops (RDNA3/ROCm), but
the algorithm itself is pure PyTorch (torch.fft/F.pad/torch.kron), so it
needs no rebuild -- just a real GPU with torch.fft/cuFFT working, and the
actual compiled cmpext3._native extension (for the F.conv2d/conv3d
comparisons below).

Run on the machine with the actual GPU:

    python test_fftconv_ops.py

Exits 0 if everything passes, 1 if any check fails or the environment
can't run it (no CUDA / cmpext3 not built).
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
    return dev


def _cosine(a, b) -> float:
    import torch.nn.functional as F
    return F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1)).item()


def check_fft_conv1d_numerics(check: Check, dev, dtype) -> None:
    import torch
    import torch.nn.functional as F
    from cmpext3 import fftconv_ops

    print("\n" + "=" * 78)
    print(f"Check: fft_conv1d matches F.conv1d  (dtype={dtype})")
    print("=" * 78)
    torch.manual_seed(0)
    b, cin, cout, length, kernel = 2, 4, 6, 256, 33
    x = torch.randn(b, cin, length, dtype=dtype, device=dev)
    w = torch.randn(cout, cin, kernel, dtype=dtype, device=dev)
    bias = torch.randn(cout, dtype=dtype, device=dev)

    want = F.conv1d(x, w, bias, padding=kernel // 2)
    got = fftconv_ops.fft_conv1d(x, w, bias, padding=kernel // 2)

    cos = _cosine(got, want)
    print(f"  cosine similarity vs F.conv1d: {cos:.6f}")
    check.require(f"fft_conv1d matches F.conv1d ({dtype})", cos > 0.999,
                   f"cosine similarity {cos:.4f} too low.")
    check.require("fft_conv1d output dtype matches input", got.dtype == dtype,
                   f"got dtype {got.dtype}, expected {dtype}.")


def check_fft_conv2d_numerics(check: Check, dev, dtype) -> None:
    import torch
    import torch.nn.functional as F
    from cmpext3 import fftconv_ops

    print("\n" + "=" * 78)
    print(f"Check: fft_conv2d matches F.conv2d  (dtype={dtype})")
    print("=" * 78)
    torch.manual_seed(0)
    b, cin, cout, h, w_, kernel = 1, 3, 5, 64, 64, 11
    x = torch.randn(b, cin, h, w_, dtype=dtype, device=dev)
    weight = torch.randn(cout, cin, kernel, kernel, dtype=dtype, device=dev)

    want = F.conv2d(x, weight, padding=kernel // 2)
    got = fftconv_ops.fft_conv2d(x, weight, padding=kernel // 2)

    cos = _cosine(got, want)
    print(f"  cosine similarity vs F.conv2d: {cos:.6f}")
    check.require(f"fft_conv2d matches F.conv2d ({dtype})", cos > 0.999,
                   f"cosine similarity {cos:.4f} too low.")


def check_fft_conv_depthwise(check: Check, dev) -> None:
    import torch
    import torch.nn.functional as F
    from cmpext3 import fftconv_ops

    print("\n" + "=" * 78)
    print("Check: fft_conv1d depthwise (groups==Cin) matches F.conv1d")
    print("=" * 78)
    torch.manual_seed(0)
    c, length, kernel = 8, 512, 65
    x = torch.randn(1, c, length, dtype=torch.float32, device=dev)
    w = torch.randn(c, 1, kernel, dtype=torch.float32, device=dev)

    want = F.conv1d(x, w, padding=kernel // 2, groups=c)
    got = fftconv_ops.fft_conv1d(x, w, padding=kernel // 2, groups=c)

    cos = _cosine(got, want)
    print(f"  cosine similarity vs F.conv1d (depthwise): {cos:.6f}")
    check.require("fft_conv1d depthwise matches F.conv1d", cos > 0.999,
                   f"cosine similarity {cos:.4f} too low -- check complex_matmul's "
                   "depthwise fast path (groups == Cin) in cmpext3/fftconv_ops.py.")


def check_kernel_wider_than_input_raises(check: Check, dev) -> None:
    import torch
    from cmpext3 import fftconv_ops

    print("\n" + "=" * 78)
    print("Check: a kernel wider than the (padded) input raises ValueError, not a wrong answer")
    print("=" * 78)
    x = torch.randn(1, 2, 8, dtype=torch.float32, device=dev)
    w = torch.randn(2, 2, 64, dtype=torch.float32, device=dev)
    raised = False
    try:
        fftconv_ops.fft_conv1d(x, w)
    except ValueError:
        raised = True
    check.require("fft_conv1d raises ValueError for kernel wider than input", raised,
                   "expected a ValueError (matching F.conv1d's own rejection), got none -- "
                   "check fft_conv's `invalid` size guard in cmpext3/fftconv_ops.py.")


def check_autopatch_engages_for_large_kernel(check: Check, dev) -> None:
    import torch
    import torch.nn.functional as F
    import cmpext3

    print("\n" + "=" * 78)
    print("Check: F.conv2d auto-patch actually routes a large kernel through fft_conv2d")
    print("=" * 78)
    was_enabled = cmpext3.is_enabled()
    if not was_enabled:
        cmpext3.enable()
    try:
        torch.manual_seed(0)
        min_kernel = cmpext3._FFTCONV2D_MIN_KERNEL
        x = torch.randn(1, 2, 96, 96, dtype=torch.float32, device=dev)
        w = torch.randn(3, 2, min_kernel, min_kernel, dtype=torch.float32, device=dev)
        want = cmpext3._ORIGINALS[(F, "conv2d")](x, w, None, 1, min_kernel // 2, 1, 1)
        with torch.no_grad():
            got = F.conv2d(x, w, padding=min_kernel // 2)
        cos = _cosine(got, want)
        print(f"  min_kernel={min_kernel}, cosine similarity vs stock F.conv2d: {cos:.6f}")
        check.require("F.conv2d(large kernel) matches stock F.conv2d", cos > 0.999,
                       f"cosine similarity {cos:.4f} too low.")
    finally:
        if not was_enabled:
            cmpext3.disable()


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
        check_fft_conv1d_numerics(check, dev, dtype)
    for dtype in (torch.float16, torch.float32):
        check_fft_conv2d_numerics(check, dev, dtype)
    check_fft_conv_depthwise(check, dev)
    check_kernel_wider_than_input_raises(check, dev)
    check_autopatch_engages_for_large_kernel(check, dev)

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
