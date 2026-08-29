#!/usr/bin/env python3
"""
Exports one ONNX model per ACTIVE kernel in src/cuda/ (fp16 and fp32 each)
into src/onnx/, named to match 1:1 (fp16_conv.cu -> src/onnx/fp16_conv.onnx,
etc.). Shapes/parameters come from tools/bench.py's own representative
values for each op -- the same source of truth setup.py's autotuner and
export_conv3d_onnx.py already use.

SCOPE: only the 17 ops actually compiled into cmpext3._native per
setup.py's `sources` list (attention, conv, conv3d, ConvTranspose2d, emb,
gelu, groupnorm, layernorm, matmul, mish, rmsnorm, silu, softmax, softplus,
softshrink, softsign, swish). Deliberately EXCLUDED: every bf16_*.cu file
(vestigial -- cmpext3/__init__.py's own docstring says these bf16-native
paths "were removed since Turing hardware can't run [native bf16
arithmetic]"; they exist on disk but were never added to setup.py's
sources) and fp16_upsample.cu/fp32_upsample.cu/bf16_upsample.cu (present on
disk but commented out in setup.py -- F.interpolate is served by the
ConvTranspose2d kernel via custom_upsample_smart instead, already covered
by the ConvTranspose2d export). tanh/erf (src/cuda-base/, not src/cuda/)
are out of scope per the same "src/cuda kernels" request that scoped this.

WEIGHT-BEARING OPS PRODUCE LARGE FILES: matmul (~100MB total for both
dtypes -- 4096x4096 weight), emb (~200MB total -- 32000x1024 embedding
table), conv3d (~40MB total -- 512x512x3x3x3 weight). Pure-elementwise ops
(gelu/silu/softmax/softplus/softsign/softshrink/swish/mish/attention) have
NO learnable weights, so their files are tiny regardless of input shape.
src/onnx/ is gitignored by default (see the .gitignore this script's first
run adds next to it) -- these are generated artifacts, not source, the
same way the autotune cache is. Re-run this script to regenerate; don't
hand-edit the .onnx files.

DYNAMIC BATCH DIMENSION: several bench.py shapes are large enough (e.g.
Linear's M=4096, GroupNorm's N=64) that tracing them directly on CPU is
needlessly slow/memory-heavy for a structural export. Where safe (leading
"count" dimension only -- never the dims baked into a weight tensor's
shape, which the file size depends on regardless of trace batch), this
script traces with a SMALL leading dimension and marks it dynamic via
dynamic_axes, then prints the matching `trtexec --shapes=` flag to use for
a benchmark at the REAL bench.py shape. The exported graph is valid at
either size; only the trace itself is cheaper.

Usage:
    python tools/export_all_onnx.py
    python tools/export_all_onnx.py --only conv3d attention   # subset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR.parent / 'src' / 'onnx'

DTYPES = [("fp16", torch.float16), ("fp32", torch.float32)]


class RMSNormModule(nn.Module):
    """Manual formula, matching tools/bench.py's own torch_rmsnorm_func
    reference exactly -- avoids any dependency on nn.RMSNorm's availability
    across torch versions (added in 2.4; this project doesn't otherwise
    require that recent a torch)."""
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.randn(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwishModule(nn.Module):
    def __init__(self, beta: float):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class AttentionModule(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v, scale=self.scale)


def _elementwise_spec(name, module_fn, full_n=8192, trace_n=64):
    """Shared spec builder for the pure-elementwise ops (bench.py's
    Section III): all use the same (N_elem, N_elem) shape with no weights.
    """
    def build(dtype):
        module = module_fn().to(dtype).eval()
        dummy = torch.randn(trace_n, trace_n, dtype=dtype)
        return (
            module, (dummy,),
            {"input": {0: "dim0", 1: "dim1"}},
            ["input"], ["output"],
            f"{full_n}x{full_n}",
        )
    return {"name": name, "build": build}


OPS = [
    # ---- weight-bearing ops (large files; see module docstring) --------
    {
        "name": "matmul",
        "build": lambda dtype: (
            nn.Linear(4096, 4096, bias=True).to(dtype).eval(),
            (torch.randn(64, 4096, dtype=dtype),),
            {"input": {0: "M"}},
            ["input"], ["output"],
            "4096x4096",
        ),
    },
    {
        # Spatial dims (not just batch) are shrunk for the trace itself --
        # channel counts (128/64) stay at bench.py's real values since
        # those are baked into the weight shape (and therefore file size)
        # regardless of trace input size, but a real fp16 CPU conv2d
        # forward at the full 128x128 spatial size measurably slows
        # tracing (this is exactly what was found hanging on fp16 conv3d
        # below before its spatial dims were shrunk the same way). H/W are
        # marked dynamic so the graph is still valid at the full
        # bench.py shape for a later trtexec --shapes= benchmark.
        "name": "conv",
        "build": lambda dtype: (
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1).to(dtype).eval(),
            (torch.randn(2, 128, 16, 16, dtype=dtype),),
            {"input": {0: "N", 2: "H", 3: "W"}},
            ["input"], ["output"],
            "64x128x128x128",
        ),
    },
    {
        # C_in/C_out stay at bench.py's real 512/512 (weight shape, and
        # therefore file size, depends only on these -- not on trace input
        # size). D/H/W are shrunk 4x/4x/4x for the trace itself: a real
        # fp16 CPU conv3d forward at the full 8x32x32 spatial size over a
        # 512x512x3x3x3 kernel is impractically slow to trace (confirmed
        # directly -- this hung for several minutes before this fix).
        # D/H/W marked dynamic so the graph is still valid at the full
        # bench.py shape for a later trtexec --shapes= benchmark.
        "name": "conv3d",
        "build": lambda dtype: (
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1).to(dtype).eval(),
            (torch.randn(1, 512, 2, 8, 8, dtype=dtype),),
            {"input": {2: "D", 3: "H", 4: "W"}},
            ["input"], ["output"],
            "1x512x8x32x32",
        ),
    },
    {
        "name": "ConvTranspose2d",
        "build": lambda dtype: (
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1,
                                output_padding=1, dilation=1).to(dtype).eval(),
            (torch.randn(2, 64, 64, 64, dtype=dtype),),
            {"input": {0: "N"}},
            ["input"], ["output"],
            "64x64x64x64",
        ),
    },
    {
        "name": "emb",
        "build": lambda dtype: (
            nn.Embedding(32000, 1024).to(dtype).eval(),
            (torch.randint(0, 32000, (8, 1024), dtype=torch.long),),
            {"input": {0: "batch"}},
            ["input"], ["output"],
            "512x1024",
        ),
    },
    {
        "name": "groupnorm",
        "build": lambda dtype: (
            nn.GroupNorm(32, 512).to(dtype).eval(),
            (torch.randn(2, 512, 128, 128, dtype=dtype),),
            {"input": {0: "N"}},
            ["input"], ["output"],
            "64x512x128x128",
        ),
    },
    {
        "name": "layernorm",
        "build": lambda dtype: (
            nn.LayerNorm(1024).to(dtype).eval(),
            (torch.randn(4, 128, 1024, dtype=dtype),),
            {"input": {0: "B"}},
            ["input"], ["output"],
            "64x128x1024",
        ),
    },
    {
        "name": "rmsnorm",
        "build": lambda dtype: (
            RMSNormModule(1024, eps=1e-6).to(dtype).eval(),
            (torch.randn(4, 128, 1024, dtype=dtype),),
            {"input": {0: "B"}},
            ["input"], ["output"],
            "64x128x1024",
        ),
    },
    # ---- no learnable weights -> tiny files regardless of shape --------
    {
        "name": "attention",
        "build": lambda dtype: (
            AttentionModule(scale=128 ** -0.5).to(dtype).eval(),
            (torch.randn(1, 8, 1024, 128, dtype=dtype),
             torch.randn(1, 8, 1024, 128, dtype=dtype),
             torch.randn(1, 8, 1024, 128, dtype=dtype)),
            {"q": {0: "B"}, "k": {0: "B"}, "v": {0: "B"}},
            ["q", "k", "v"], ["output"],
            "8x8x1024x128 (each of q,k,v)",
        ),
    },
    _elementwise_spec("gelu", lambda: nn.GELU()),
    _elementwise_spec("silu", lambda: nn.SiLU()),
    _elementwise_spec("swish", lambda: SwishModule(beta=10.0)),
    _elementwise_spec("mish", lambda: nn.Mish()),
    _elementwise_spec("softmax", lambda: nn.Softmax(dim=-1)),
    _elementwise_spec("softplus", lambda: nn.Softplus(beta=1.0, threshold=20.0)),
    _elementwise_spec("softsign", lambda: nn.Softsign()),
    _elementwise_spec("softshrink", lambda: nn.Softshrink(lambd=0.5)),
]


def _ensure_gitignore():
    gi = OUT_DIR / '.gitignore'
    if not gi.exists():
        gi.write_text(
            "# Generated by tools/export_all_onnx.py -- regenerate, don't hand-edit\n"
            "# or commit: weight-bearing exports (matmul/emb/conv3d) run into the\n"
            "# tens-to-hundreds of MB each. See that script's module docstring.\n"
            "*.onnx\n"
        )


def export_one(name, dtype_name, dtype, build_fn):
    module, dummy_args, dynamic_axes, input_names, output_names, full_shape = build_fn(dtype)
    out_path = OUT_DIR / f"{dtype_name}_{name}.onnx"

    with torch.inference_mode():
        torch.onnx.export(
            module,
            dummy_args,
            str(out_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes or None,
            dynamo=False,
        )

    size_mb = out_path.stat().st_size / (1024 * 1024)

    try:
        import onnx
        onnx.checker.check_model(onnx.load(str(out_path)))
        checked = "checker OK"
    except ImportError:
        checked = "onnx package not installed, skipped checker"

    return out_path, size_mb, checked, full_shape


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", default=None,
                         help="Only export these op names (e.g. --only conv3d attention)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_gitignore()

    ops = OPS if not args.only else [o for o in OPS if o["name"] in args.only]
    if args.only:
        missing = set(args.only) - {o["name"] for o in OPS}
        if missing:
            print(f"Unknown op name(s): {sorted(missing)}")
            return 1

    results = []
    failures = []
    for op in ops:
        for dtype_name, dtype in DTYPES:
            label = f"{dtype_name}_{op['name']}"
            try:
                path, size_mb, checked, full_shape = export_one(op["name"], dtype_name, dtype, op["build"])
                print(f"  {label:28s} {size_mb:9.2f} MB  [{checked}]  (bench shape: {full_shape})")
                results.append((label, path, size_mb))
            except Exception as exc:
                print(f"  {label:28s} FAILED: {exc}")
                failures.append((label, str(exc)))

    total_mb = sum(r[2] for r in results)
    print()
    print(f"Wrote {len(results)}/{len(ops) * len(DTYPES)} files to {OUT_DIR} ({total_mb:.1f} MB total)")
    if failures:
        print(f"{len(failures)} FAILED:")
        for label, err in failures:
            print(f"  - {label}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
