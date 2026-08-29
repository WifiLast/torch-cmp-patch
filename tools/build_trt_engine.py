#!/usr/bin/env python3
"""
Builds a serialized TensorRT engine from an ONNX model in src/onnx/,
writing it to src/trt/. Thin wrapper around trtexec (ships with any
TensorRT SDK install) -- the same tool setup.py's autotuner uses for its
TensorRT side-comparison (see _run_trt_candidate() in ../setup.py).

WARNING -- UNVERIFIED, NEEDS YOUR HARDWARE:
A TensorRT engine is NOT a portable ahead-of-time artifact the way ONNX
is. TensorRT profiles/benchmarks candidate kernel "tactics" against the
actual GPU present at BUILD time, and the resulting .trt file is
generally only loadable on a matching GPU architecture + TensorRT major
version -- it cannot be produced (or meaningfully faked) anywhere but
your own CMP 50HX with TensorRT installed. There is no TensorRT SDK
anywhere in this environment (confirmed: no pip package, no SDK
install, no NvInfer.h/trtexec on this whole machine), so nothing about
an actual engine build has been run here. What HAS been verified is
this script's own plumbing (path handling, --minShapes/--optShapes/
--maxShapes construction, precision-flag detection from the ONNX file's
own dtype) against a stub trtexec that asserts its arguments look
right -- the same technique setup.py's TensorRT comparison path was
verified with, since real trtexec isn't available here either.

Usage:
    python tools/build_trt_engine.py --onnx src/onnx/fp16_conv3d.onnx

    # Override the shape profile (default is looked up by op name, parsed
    # from the filename, against KNOWN_INPUT_SHAPES below -- the same
    # bench.py-derived shapes tools/export_all_onnx.py used to export):
    python tools/build_trt_engine.py --onnx src/onnx/fp16_conv3d.onnx \\
        --shape input:1x512x8x32x32

Static-shape ONNX files (e.g. fp32_conv3d.onnx, exported with no
dynamic_axes) don't need --shape at all -- trtexec infers the profile
directly from the model.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent
OUT_DIR = REPO_DIR / 'src' / 'trt'

# Mirrors export_all_onnx.py's OPS[*]['build'] full_shape values -- kept as
# a small, explicit table here rather than importing that module, since
# these are physical dimensions (what a "realistic" shape means for this
# op), not export-time trace-shrinking details that module owns.
KNOWN_INPUT_SHAPES = {
    "matmul": {"input": [64, 4096]},          # M dim was left dynamic; 64 is just A valid M
    "conv": {"input": [64, 128, 128, 128]},
    "conv3d": {"input": [1, 512, 8, 32, 32]},
    "ConvTranspose2d": {"input": [64, 64, 64, 64]},
    "emb": {"input": [512, 1024]},
    "groupnorm": {"input": [64, 512, 128, 128]},
    "layernorm": {"input": [64, 128, 1024]},
    "rmsnorm": {"input": [64, 128, 1024]},
    "attention": {"q": [8, 8, 1024, 128], "k": [8, 8, 1024, 128], "v": [8, 8, 1024, 128]},
}


def _find_trtexec():
    """Same discovery logic as setup.py's _find_trtexec() -- duplicated
    rather than imported since setup.py isn't meant to be imported as a
    module (it runs the real build as a side effect at import time)."""
    import os
    env_home = os.environ.get('TENSORRT_HOME')
    if env_home:
        candidate = Path(env_home) / 'bin' / 'trtexec.exe'
        if candidate.is_file():
            return str(candidate)
    on_path = shutil.which('trtexec')
    if on_path:
        return on_path
    for root in (Path(r'C:\Program Files\NVIDIA'), Path(r'C:\TensorRT')):
        if not root.is_dir():
            continue
        for base in list(root.glob('TensorRT-*')) + list(root.glob('*')):
            candidate = base / 'bin' / 'trtexec.exe'
            if candidate.is_file():
                return str(candidate)
    return None


def _op_name_from_filename(onnx_path: Path) -> str:
    stem = onnx_path.stem  # e.g. "fp16_conv3d"
    for prefix in ("fp16_", "fp32_", "bf16_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def _inspect_onnx(onnx_path: Path):
    """Returns (input_names, dynamic_dim_names, is_fp16) by reading the
    ONNX file directly -- no assumptions beyond what's actually in it.
    """
    import onnx
    model = onnx.load(str(onnx_path))
    input_names = [i.name for i in model.graph.input]
    dynamic = set()
    is_fp16 = False
    for inp in model.graph.input:
        if inp.type.tensor_type.elem_type == 10:  # TensorProto.FLOAT16
            is_fp16 = True
        for d in inp.type.tensor_type.shape.dim:
            if d.dim_param:
                dynamic.add(inp.name)
    return input_names, dynamic, is_fp16


def build_shape_flags(onnx_path: Path, explicit_shapes: list[str] | None):
    """Returns the --minShapes/--optShapes/--maxShapes args trtexec needs,
    or [] if the model has no dynamic dims. explicit_shapes, if given, is a
    list of "name:dims" strings (repeatable --shape flags) that override
    the KNOWN_INPUT_SHAPES lookup.
    """
    input_names, dynamic_names, is_fp16 = _inspect_onnx(onnx_path)
    if not dynamic_names:
        return [], is_fp16

    overrides = {}
    for spec in explicit_shapes or []:
        name, dims = spec.split(':', 1)
        overrides[name] = dims

    op_name = _op_name_from_filename(onnx_path)
    known = KNOWN_INPUT_SHAPES.get(op_name, {})

    shape_parts = []
    for name in input_names:
        if name in overrides:
            dims = overrides[name]
        elif name in known:
            dims = 'x'.join(str(d) for d in known[name])
        else:
            raise ValueError(
                f"input '{name}' has dynamic dims but no known shape -- pass "
                f"--shape {name}:D1xD2x... explicitly (KNOWN_INPUT_SHAPES has "
                f"no entry for op '{op_name}')"
            )
        shape_parts.append(f"{name}:{dims}")

    joined = ','.join(shape_parts)
    # min=opt=max: a shape-specialized engine at exactly the production
    # shape gives TensorRT the most room to specialize, at the cost of not
    # accepting other shapes at runtime -- the right tradeoff for a
    # benchmark/deployment engine, not a general-purpose one.
    return [f'--minShapes={joined}', f'--optShapes={joined}', f'--maxShapes={joined}'], is_fp16


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", required=True, help="Path to the source .onnx file")
    parser.add_argument("--out", default=None, help="Output .trt path (default: src/trt/<same stem>.trt)")
    parser.add_argument("--shape", action="append", default=None,
                         help="Override a dynamic input's shape, name:D1xD2x... (repeatable)")
    args = parser.parse_args()

    onnx_path = Path(args.onnx).resolve()
    if not onnx_path.is_file():
        print(f"ONNX file not found: {onnx_path}")
        return 1

    trtexec = _find_trtexec()
    if trtexec is None:
        print("trtexec not found (checked TENSORRT_HOME, PATH, and common install paths) -- "
              "can't build an engine without a TensorRT SDK install. This is expected in an "
              "environment without TensorRT; run this on your CMP 50HX instead.")
        return 1

    try:
        shape_flags, is_fp16 = build_shape_flags(onnx_path, args.shape)
    except ImportError:
        print("The 'onnx' package is required to inspect the model's dynamic dims "
              "(pip install onnx) -- it's not installed here.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gi = OUT_DIR / '.gitignore'
    if not gi.exists():
        gi.write_text(
            "# Generated by tools/build_trt_engine.py -- NEVER commit these.\n"
            "# TensorRT engines are compiled against the exact GPU + TensorRT version\n"
            "# that built them; they are not portable and are meaningless on any other\n"
            "# machine (including CI). Rebuild locally instead.\n"
            "*.trt\n*.engine\n"
        )

    out_path = Path(args.out).resolve() if args.out else OUT_DIR / (onnx_path.stem + '.trt')

    cmd = [trtexec, f'--onnx={onnx_path}', f'--saveEngine={out_path}']
    if is_fp16:
        cmd.append('--fp16')
    cmd += shape_flags

    print("Running:", ' '.join(cmd))
    # TensorRT's tactic search can take well over a minute even for a
    # single layer -- no artificial timeout here, this is meant to be run
    # interactively on real hardware, not from an automated harness.
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"trtexec failed with exit code {result.returncode}")
        return result.returncode

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
