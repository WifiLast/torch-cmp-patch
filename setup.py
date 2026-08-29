import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# pip install -e . --no-build-isolation

# 获取CPU核心数
max_workers = multiprocessing.cpu_count()

THIS_DIR = Path(__file__).resolve().parent

# CMP 50HX = TU116, compute capability 7.5. Shared by the real build and the
# autotuner below so a compiled candidate always targets the same arch as
# the extension it might get folded into.
CUDA_GENCODE = '-gencode=arch=compute_75,code=sm_75'

# ---------------------------------------------------------------------------
# Autotuner for the tunable kernels' compile-time tile/tile-width parameters
# (see each .cu file's header comment and the register/occupancy sweeps they
# were seeded from). ON BY DEFAULT -- it compiles and benchmarks several
# standalone kernel variants per target below the real build, which costs
# real wall-clock time (~1-5 min total on a fresh install) but only once per
# machine: winning configs are cached (CACHE_FILE below), so every install
# after the first is a cache hit and doesn't re-run the sweep. Opt out with
# CMPEXT3_AUTOTUNE=0 if you want the plain hardcoded defaults instead (e.g.
# reproducible/CI builds). Skipped per-target (with a one-line notice),
# falling back to that target's default, whenever autotuning is disabled, no
# GPU/nvcc is visible, or literally any step fails for that target --
# autotuning must never be able to break an otherwise-working
# `pip install -e .`, and a failure on one target must never block the rest.
#
# Two kinds of target:
#  - GEMM-tiled (fp16 conv2d/conv3d): 4 params (BM, BN, BK, STAGES). First
#    candidate in each list is the shipped default -- also this project's
#    baseline in the comparison. Pre-filtered to fit Turing's
#    1024-threads/block and 64KB-shared-memory limits -- see
#    fp16_conv3d.cu's tile-size sweep for how these lists were chosen.
#  - Register-blocked (fp32 conv2d/conv3d/ConvTranspose2d, fp16
#    ConvTranspose2d): 1 param (CTILE, output channels per thread). Higher
#    CTILE trades more register pressure for more weight reuse -- a real
#    tradeoff, unlike --maxrregcount (see the earlier register sweep, which
#    found tuning that dead for these kernels' fixed 256-thread blocks).
# ---------------------------------------------------------------------------

AUTOTUNE_TARGETS = [
    {
        'name': 'conv3d_fp16',
        'source': 'src/cuda/fp16_conv3d.cu',
        'defines': ['CONV3D_BM', 'CONV3D_BN', 'CONV3D_BK', 'CONV3D_STAGES'],
        'candidates': [
            (256, 128, 32, 2),   # current default
            (128, 256, 16, 2),
            (128, 256, 32, 2),
            (256, 128, 16, 2),
            (512, 64, 16, 2),
            (256, 128, 32, 3),
        ],
        'half_precision': True,
    },
    {
        'name': 'conv2d_fp16',
        'source': 'src/cuda/fp16_conv.cu',
        'defines': ['CONV2D_BM', 'CONV2D_BN', 'CONV2D_BK', 'CONV2D_STAGES'],
        'candidates': [
            (256, 128, 32, 2),   # current default
            (128, 256, 16, 2),
            (128, 256, 32, 2),
            (256, 128, 16, 2),
            (512, 64, 16, 2),
            (256, 128, 32, 3),
        ],
        'half_precision': True,
    },
    {
        'name': 'conv2d_fp32',
        'source': 'src/cuda/fp32_conv.cu',
        'defines': ['CONV2D_FP32_CTILE'],
        'candidates': [(8,), (4,), (16,), (32,)],   # current default first
        'half_precision': False,
    },
    {
        'name': 'conv3d_fp32',
        'source': 'src/cuda/fp32_conv3d.cu',
        'defines': ['CONV3D_FP32_CTILE'],
        'candidates': [(8,), (4,), (16,), (32,)],
        'half_precision': False,
        # Optional side-comparison against a single-layer TensorRT engine
        # for the identical shape/precision, via tools/export_conv3d_onnx.py
        # + trtexec (see _run_trt_candidate() below) rather than hand-written
        # TensorRT C++ API code -- trtexec ships with every TensorRT SDK
        # install and needs no compiling against NvInfer.h at all, which
        # matters here because this project has no TensorRT SDK to test
        # against. See export_conv3d_onnx.py's header for why this might
        # lose (no --fmad=false equivalent in TensorRT) and
        # _run_trt_candidate() for why a TensorRT win is reported but not
        # "applied": there is no runtime path yet that would let the real
        # extension actually use a TensorRT engine instead of this kernel.
        'trt_onnx': True,
    },
    {
        'name': 'conv_transpose2d_fp32',
        'source': 'src/cuda/fp32_ConvTranspose2d.cu',
        'defines': ['CONVT2D_FP32_CTILE'],
        'candidates': [(8,), (4,), (16,), (32,)],
        'half_precision': False,
    },
    {
        'name': 'conv_transpose2d_fp16',
        'source': 'src/cuda/fp16_ConvTranspose2d.cu',
        'defines': ['CONVT2D_FP16_CTILE'],
        'candidates': [(8,), (4,), (16,), (32,)],
        'half_precision': True,
    },
]

CACHE_FILE = THIS_DIR / '.cmpext3_autotune_cache.json'


def _autotune_enabled():
    return os.environ.get('CMPEXT3_AUTOTUNE', '1').lower() not in ('0', 'false', 'no')


def _format_config(defines, config):
    return ' '.join(f'{name}={value}' for name, value in zip(defines, config))


def _load_cache():
    if os.environ.get('CMPEXT3_AUTOTUNE_FORCE', '0').lower() in ('1', 'true', 'yes'):
        return {}
    if not CACHE_FILE.exists():
        return {}
    try:
        return json.loads(CACHE_FILE.read_text())
    except Exception:
        return {}


def _save_cache(cache):
    try:
        CACHE_FILE.write_text(json.dumps(cache, indent=2))
    except OSError:
        pass


def _find_msvc_bindir():
    """Best-effort discovery of MSVC's cl.exe directory (Windows only). Bare
    `nvcc` can only find cl.exe via its own PATH-based search, which comes
    up empty outside a VS Developer shell (e.g. a plain terminal) even when
    a normal `pip install -e .` -- via torch's own, more thorough MSVC
    discovery -- would have worked fine. Returns None if nothing is found;
    callers should just omit --compiler-bindir in that case (a no-op change
    from not calling this at all).
    """
    roots = [
        Path(os.environ.get('ProgramFiles', r'C:\Program Files')) / 'Microsoft Visual Studio',
        Path(os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')) / 'Microsoft Visual Studio',
    ]
    candidates = []
    for root in roots:
        if root.is_dir():
            candidates.extend(root.glob('*/*/VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe'))
    if not candidates:
        return None
    # Prefer the highest MSVC toolset version -- lexicographic sort works
    # for the standard X.YY.ZZZZZ toolset-version directory naming.
    return str(sorted(candidates, key=lambda p: p.parts)[-1].parent)


def _find_trtexec():
    """Best-effort discovery of trtexec, the command-line benchmarking tool
    that ships with every TensorRT SDK install. Using it (via
    tools/export_conv3d_onnx.py + trtexec, see _run_trt_candidate()) instead
    of hand-written TensorRT C++ API code means no compiling against
    NvInfer.h is needed at all -- this project has no TensorRT SDK to test
    such code against. Note the `pip install tensorrt` wheel this repo's
    Stable-Diffusion-WebUI-TensorRT extension uses does NOT include
    trtexec; it needs the separate full SDK zip/installer distribution.
    Returns the executable path, or None if not found -- callers should
    just skip the TensorRT candidate in that case.

    Checked in order: TENSORRT_HOME env var, trtexec already on PATH, then
    common Windows install locations for the TensorRT SDK distribution.
    """
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


def _find_tensorrt_sdk():
    """Best-effort discovery of a full TensorRT SDK install providing
    NvInfer.h + libnvinfer -- needed to compile src/trt/*.cpp (the
    optional CMPEXT3_USE_TENSORRT runtime path; see trt_common.h's design
    rationale and main.cpp's cmpext3_trt_enabled()). NOT the same thing as
    `pip install tensorrt` (that wheel is Python-bindings-only, no C++
    headers -- see _find_trtexec()'s docstring above for why this
    project's autotuner needs the separate full SDK too): this needs the
    actual C/C++ SDK, either the manually-extracted tarball/zip
    distribution (set TENSORRT_HOME to its root, same env var
    _find_trtexec() checks) or a system package install (e.g. Debian/
    Ubuntu's libnvinfer-dev, which spreads headers/libs across the normal
    multiarch system include/lib directories instead of one self-contained
    tree).

    Returns (include_dir, lib_dir) as strings, or None if no NvInfer.h +
    matching libnvinfer could be found anywhere checked. src/trt/*.cpp and
    -DCMPEXT3_WITH_TENSORRT are only added to the real build (see setup()
    below) when this succeeds -- absence here just means
    CMPEXT3_USE_TENSORRT stays a no-op at runtime (main.cpp's existing
    hand-tuned-kernel dispatch is unaffected), same graceful-skip
    philosophy as every other optional tool in this file (trtexec, MSVC).
    """
    def _pair(include_dir, lib_dir):
        include_dir, lib_dir = Path(include_dir), Path(lib_dir)
        if not (include_dir / 'NvInfer.h').is_file():
            return None
        lib_names = ['nvinfer.lib'] if os.name == 'nt' else ['libnvinfer.so']
        if not any((lib_dir / name).is_file() for name in lib_names):
            return None
        return (str(include_dir), str(lib_dir))

    env_home = os.environ.get('TENSORRT_HOME')
    if env_home:
        root = Path(env_home)
        found = _pair(root / 'include', root / 'lib')
        if found:
            return found

    if os.name == 'nt':
        for root in (Path(r'C:\Program Files\NVIDIA'), Path(r'C:\TensorRT')):
            if not root.is_dir():
                continue
            for base in list(root.glob('TensorRT-*')) + list(root.glob('*')):
                found = _pair(base / 'include', base / 'lib')
                if found:
                    return found
        return None

    # Linux system package install (e.g. `apt install libnvinfer-dev`):
    # headers land in /usr/include (or the multiarch include dir), libs in
    # the matching multiarch lib dir -- neither is one self-contained tree
    # the way the tarball distribution above is.
    include_candidates = [Path('/usr/include')]
    lib_candidates = [Path('/usr/lib'), Path('/usr/lib64')]
    try:
        multiarch = subprocess.run(
            ['gcc', '-print-multiarch'], capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        multiarch = ''
    if multiarch:
        include_candidates.append(Path(f'/usr/include/{multiarch}'))
        lib_candidates.append(Path(f'/usr/lib/{multiarch}'))
    for include_dir in include_candidates:
        for lib_dir in lib_candidates:
            found = _pair(include_dir, lib_dir)
            if found:
                return found

    return None


def _run_one_candidate(nvcc, tmpdir, target, config, msvc_bindir):
    source = THIS_DIR / target['source']
    tag = '_'.join(str(v) for v in config)
    exe = os.path.join(tmpdir, f"tune_{target['name']}_{tag}.exe")
    compile_cmd = [
        nvcc, str(source),
        '-o', exe,
        CUDA_GENCODE, '-O3', '--use_fast_math', '--fmad=false',
        '-DCMPEXT3_AUTOTUNE_HARNESS',
        '-cudart', 'static',
    ]
    if target.get('half_precision'):
        compile_cmd += [
            '-U__CUDA_NO_HALF_OPERATORS__',
            '-U__CUDA_NO_HALF_CONVERSIONS__',
            '-U__CUDA_NO_HALF2_OPERATORS__',
        ]
    compile_cmd += [f'-D{name}={value}' for name, value in zip(target['defines'], config)]
    if msvc_bindir:
        compile_cmd += ['--compiler-bindir', msvc_bindir]
    # Linux equivalent of msvc_bindir above: mirrors BuildExtension's own
    # `-ccbin $CC` behavior (see torch/utils/cpp_extension.py) so the
    # autotune probe and the real extension build agree on a host compiler.
    # Needed on any machine where nvcc's default host g++ is newer than
    # this CUDA release supports -- e.g. CUDA 12.8 + GCC 15 fails every
    # single .cu file with "type name is not allowed" in <type_traits>
    # (GCC 15's libstdc++ leans on compiler builtins like __is_pointer()
    # that cudafe++ doesn't know). Without this, every autotune candidate
    # on such a machine would silently "FAIL" and fall back to the
    # hardcoded default, defeating the sweep entirely.
    ccbin = os.environ.get('CC')
    if ccbin:
        compile_cmd += ['-ccbin', ccbin]
    subprocess.run(compile_cmd, check=True, capture_output=True, text=True, timeout=120)
    result = subprocess.run([exe], check=True, capture_output=True, text=True, timeout=30)
    for line in result.stdout.splitlines():
        if line.startswith('RESULT_MS='):
            return float(line.split('=', 1)[1])
    raise RuntimeError(f"harness produced no RESULT_MS line (stdout: {result.stdout!r})")


def _run_trt_candidate(trtexec, tmpdir, target):
    """Exports fp32_conv3d.cu's representative shape to ONNX (tools/
    export_conv3d_onnx.py) and benchmarks it with trtexec -- no compiling
    against TensorRT headers needed at all, unlike a hand-written C++ API
    harness this project has no TensorRT SDK to test such code against.
    Parses trtexec's own "GPU Compute Time" mean: the closest
    apples-to-apples match to the pure kernel-execution timing (no
    host<->device transfer) every other harness in this project measures
    via CUDA events.
    """
    onnx_path = os.path.join(tmpdir, 'fp32_conv3d.onnx')
    export_script = THIS_DIR / 'tools' / 'export_conv3d_onnx.py'
    subprocess.run(
        [sys.executable, str(export_script), '--out', onnx_path],
        check=True, capture_output=True, text=True, timeout=60,
    )

    # TensorRT's own tactic search happens inside trtexec's engine build,
    # which can take well over the few seconds a plain kernel launch needs
    # -- generous timeout.
    result = subprocess.run(
        [trtexec, f'--onnx={onnx_path}', '--fp32', '--iterations=20', '--avgRuns=20'],
        capture_output=True, text=True, timeout=180,
    )
    output = result.stdout + result.stderr
    match = re.search(r'GPU Compute Time:.*?mean\s*=\s*([\d.]+)\s*ms', output)
    if not match:
        raise RuntimeError(f"trtexec produced no parseable GPU Compute Time (output tail: {output[-500:]!r})")
    return float(match.group(1))


def _print_trt_comparison(name, kernel_ms, trt_ms):
    if trt_ms < kernel_ms:
        print(f"[cmpext3] {name}: TensorRT engine measured FASTER ({trt_ms:.3f} ms) than "
              f"the hand-tuned kernel ({kernel_ms:.3f} ms) -- still using the hand-tuned "
              f"kernel for the real build, there is no runtime TensorRT dispatch path yet. "
              f"Reported for your information only; see tools/export_conv3d_onnx.py.")
    else:
        print(f"[cmpext3] {name}: TensorRT engine measured {trt_ms:.3f} ms vs. the "
              f"hand-tuned kernel's {kernel_ms:.3f} ms -- hand-tuned kernel wins.")


def autotune_kernel_tile(target, nvcc, msvc_bindir, trtexec, cache):
    """Returns the winning parameter tuple for one AUTOTUNE_TARGETS entry --
    from cache, a fresh benchmark sweep, or (by default, or on any failure)
    that target's hardcoded default, always candidates[0]. A failure here
    only affects this one target; callers loop over all targets independently.
    If the target declares 'trt_onnx' and trtexec was found, also runs that
    as an informational side-comparison (see _print_trt_comparison) -- it
    can never become the returned config, since there is no runtime path
    yet that would let the real build actually use a TensorRT engine
    instead of this kernel.
    """
    default = target['candidates'][0]
    name, defines = target['name'], target['defines']

    cached = cache.get(name)
    if cached is not None:
        config = tuple(cached['config'])
        print(f"[cmpext3] {name}: using cached autotune result "
              f"({_format_config(defines, config)}, {cached['ms']:.3f} ms/iter). "
              f"Set CMPEXT3_AUTOTUNE_FORCE=1 to re-tune.")
        if 'trt_ms' in cached:
            _print_trt_comparison(name, cached['ms'], cached['trt_ms'])
        return config

    print(f"[cmpext3] Autotuning {name} ({target['source']}, "
          f"{len(target['candidates'])} candidates)...")
    results = []
    with tempfile.TemporaryDirectory(prefix=f'cmpext3_autotune_{name}_') as tmpdir:
        for config in target['candidates']:
            try:
                ms = _run_one_candidate(nvcc, tmpdir, target, config, msvc_bindir)
                print(f"    {_format_config(defines, config):40s}  {ms:9.3f} ms/iter")
                results.append((config, ms))
            except Exception as exc:
                detail = getattr(exc, 'stderr', None) or str(exc)
                print(f"    {_format_config(defines, config):40s}  FAILED ({detail!s:.200})")

    if not results:
        print(f"[cmpext3] {name}: every autotune candidate failed -- using the default.")
        return default

    best_config, best_ms = min(results, key=lambda r: r[1])
    print(f"[cmpext3] {name} winner: {_format_config(defines, best_config)}  ({best_ms:.3f} ms/iter)")
    cache_entry = {'config': list(best_config), 'ms': best_ms}

    if target.get('trt_onnx') and trtexec:
        with tempfile.TemporaryDirectory(prefix=f'cmpext3_autotune_{name}_trt_') as tmpdir:
            try:
                trt_ms = _run_trt_candidate(trtexec, tmpdir, target)
                cache_entry['trt_ms'] = trt_ms
                _print_trt_comparison(name, best_ms, trt_ms)
            except Exception as exc:
                detail = getattr(exc, 'stderr', None) or str(exc)
                print(f"[cmpext3] {name}: TensorRT comparison build/run failed "
                      f"({detail!s:.300}) -- continuing with the hand-tuned kernel only.")

    cache[name] = cache_entry
    return best_config


def autotune_all():
    """Returns {target_name: config_tuple} for every AUTOTUNE_TARGETS entry,
    each independently resolved from cache/sweep/default -- see
    autotune_kernel_tile(). One shared GPU/nvcc/MSVC/trtexec probe up
    front so a missing prerequisite is reported once, not once per target.
    """
    winners = {t['name']: t['candidates'][0] for t in AUTOTUNE_TARGETS}
    if not _autotune_enabled():
        return winners

    try:
        import torch
        if not torch.cuda.is_available():
            print("[cmpext3] Autotune: no CUDA device is visible -- using default kernel "
                  "tile configs. (Set CMPEXT3_AUTOTUNE=0 to silence this notice on "
                  "GPU-less/CI builds.)")
            return winners
    except ImportError:
        return winners

    nvcc = shutil.which('nvcc')
    if nvcc is None:
        print("[cmpext3] Autotune: nvcc isn't on PATH -- using default kernel tile configs.")
        return winners

    msvc_bindir = _find_msvc_bindir()

    trtexec = None
    if any(t.get('trt_onnx') for t in AUTOTUNE_TARGETS):
        trtexec = _find_trtexec()
        if trtexec is None:
            print("[cmpext3] Autotune: trtexec not found (checked TENSORRT_HOME, PATH, and "
                  "common install paths) -- skipping the TensorRT side-comparison, hand-tuned "
                  "kernel candidates still run normally. Note: the `pip install tensorrt` "
                  "wheel alone doesn't include trtexec; it needs the full SDK distribution.")

    cache = _load_cache()
    for target in AUTOTUNE_TARGETS:
        winners[target['name']] = autotune_kernel_tile(target, nvcc, msvc_bindir, trtexec, cache)
    _save_cache(cache)
    return winners


_autotune_winners = autotune_all()

nvcc_flags = [
    '-O3',
    #'-std=c++14',
    '--use_fast_math',
    '--ptxas-options=-v',
    '--fmad=false',  # <--- 关键：全局禁止生成 FMA 指令
    # Explicit Turing (CMP 50HX = TU116) target: makes the
    # build reproducible on machines without the physical
    # GPU attached (CI/containers), where CUDAExtension's
    # normal auto-detection of the building machine's GPU
    # can't run. Redundant but harmless when building on the
    # card itself.
    CUDA_GENCODE,
    # 必须显式 "Undefine" PyTorch 自动添加的禁用 Half 的宏
    '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__',
    '-U__CUDA_NO_HALF2_OPERATORS__',
]

# Only append overrides that differ from a target's own default -- keeps the
# compile command clean in the (overwhelmingly common) case where autotuning
# is off or picked the default anyway. Safe to apply extension-wide even
# though extra_compile_args['nvcc'] applies to every .cu file in this
# extension: each target's #define names are prefixed uniquely (CONV3D_,
# CONV2D_, CONV2D_FP32_, CONV3D_FP32_, CONVT2D_FP32_, CONVT2D_FP16_) so a
# file that doesn't declare a given macro simply ignores an override meant
# for a different file -- see fp16_conv3d.cu's header comment for why.
for _target in AUTOTUNE_TARGETS:
    _winner = _autotune_winners[_target['name']]
    if _winner != _target['candidates'][0]:
        nvcc_flags += [f'-D{name}={value}' for name, value in zip(_target['defines'], _winner)]

cxx_flags = ['-O3']

# Optional runtime TensorRT execution path (CMPEXT3_USE_TENSORRT=1 opt-in
# at import time -- see main.cpp's cmpext3_trt_enabled() and
# trt_common.h's design rationale). src/trt/*.cpp already exists and
# main.cpp already #includes + calls it, but BOTH are inert without this:
# neither the sources below nor CMPEXT3_WITH_TENSORRT were ever wired up
# here before, so every build up to now silently compiled the whole
# TensorRT path out. Skipped gracefully (extension still builds fine,
# CMPEXT3_USE_TENSORRT just stays a no-op) when no TensorRT SDK is found --
# same philosophy as every other optional tool in this file.
_tensorrt_sdk = _find_tensorrt_sdk()
extra_ext_kwargs = {}
if _tensorrt_sdk:
    _trt_include_dir, _trt_lib_dir = _tensorrt_sdk
    print(f"[cmpext3] TensorRT SDK found ({_trt_include_dir}) -- building the optional "
          f"CMPEXT3_USE_TENSORRT runtime path (src/trt/).")
    cxx_flags += ['-DCMPEXT3_WITH_TENSORRT']
    extra_ext_kwargs['include_dirs'] = [_trt_include_dir]
    extra_ext_kwargs['library_dirs'] = [_trt_lib_dir]
    extra_ext_kwargs['libraries'] = ['nvinfer']
else:
    print("[cmpext3] TensorRT SDK not found (checked TENSORRT_HOME and common install paths) -- "
          "building without the optional TensorRT runtime path. The hand-tuned kernels are "
          "unaffected; CMPEXT3_USE_TENSORRT will just stay a no-op. Set TENSORRT_HOME to the "
          "SDK root (containing include/NvInfer.h) to enable it.")

sources = [
    'src/cuda/fp16_matmul.cu',
    'src/cuda/fp32_matmul.cu',
    'src/cuda/fp16_conv.cu',
    'src/cuda/fp32_conv.cu',
    'src/cuda/fp16_conv3d.cu',
    'src/cuda/fp32_conv3d.cu',
    # F(2x2x2,3x3x3) Winograd alternative to fp32_conv3d.cu, used only
    # for the narrow slice of shapes it applies to AND only when
    # measured faster there -- see custom_conv3d_forward in main.cpp.
    'src/cuda/fp32_conv3d_winograd.cu',
    'src/cuda/fp16_emb.cu',
    'src/cuda/fp32_emb.cu',
    'src/cuda/fp16_ConvTranspose2d.cu',
    'src/cuda/fp32_ConvTranspose2d.cu',
    'src/cuda/fp16_groupnorm.cu',
    'src/cuda/fp32_groupnorm.cu',
    'src/cuda/fp16_layernorm.cu',
    'src/cuda/fp32_layernorm.cu',
    'src/cuda/fp16_rmsnorm.cu',
    'src/cuda/fp32_rmsnorm.cu',
    'src/cuda/fp16_attention.cu',
    'src/cuda/fp32_attention.cu',
    #'src/cuda/fp16_upsample.cu',
    #'src/cuda/fp32_upsample.cu',
    #'src/cuda/bf16_upsample.cu',
    'src/cuda/fp16_gelu.cu',
    'src/cuda/fp32_gelu.cu',
    'src/cuda/fp16_silu.cu',
    'src/cuda/fp32_silu.cu',
    'src/cuda/fp16_swish.cu',
    'src/cuda/fp32_swish.cu',
    'src/cuda/fp16_mish.cu',
    'src/cuda/fp32_mish.cu',
    'src/cuda/fp16_softmax.cu',
    'src/cuda/fp32_softmax.cu',
    'src/cuda/fp16_softplus.cu',
    'src/cuda/fp32_softplus.cu',
    'src/cuda/fp16_softsign.cu',
    'src/cuda/fp32_softsign.cu',
    'src/cuda/fp16_softshrink.cu',
    'src/cuda/fp32_softshrink.cu',
    'src/cuda-base/fp16_base_tanh.cu',
    'src/cuda-base/fp32_base_tanh.cu',
    'src/cuda-base/fp16_base_erf.cu',
    'src/cuda-base/fp32_base_erf.cu',
    'src/main.cpp',
]
if _tensorrt_sdk:
    sources += [
        'src/trt/trt_common.cpp',
        'src/trt/conv3d_trt_runtime.cpp',
        'src/trt/conv2d_trt_runtime.cpp',
        'src/trt/matmul_trt_runtime.cpp',
        'src/trt/attention_trt_runtime.cpp',
    ]

setup(
    name='cmpext3',
    version='0.0.1',                       # 版本号
    description='A Pytorch Extension for CMP 170HX/50HX that monkeypatches torch/F ops with FMA-throttle-bypassing kernels.',
    author='eastmoe',
    url='https://github.com/eastmoe/cmp_ext',
    packages=['cmpext3'],
    ext_modules=[
        CUDAExtension(
            name='cmpext3._native',
            sources=sources,
            extra_compile_args={
                'cxx': cxx_flags,
                'nvcc': nvcc_flags
            },
            **extra_ext_kwargs
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False, max_workers=max_workers)
    }
)
