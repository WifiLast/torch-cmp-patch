
# cmpext3 - PyTorch/ComfyUI CUDA Extension

This is a PyTorch/ComfUI extension to bypass FFMA/Tensor cores throttling on CMP mining cards based on Turing chips (TU10X).
Forked from [eastmoe/cmp_ext](https://github.com/eastmoe/cmp_ext) - the original code targeted CMP 170HX (Ampere GPU).
Supports FP16, FP32, and BF16 (Turing has no native bf16 arithmetic, so bf16
tensors are converted at the kernel boundary and run through an existing
FMA-free kernel -- earlier BF16 paths that tried native bf16 arithmetic were
removed since Turing hardware can't run it). Every op converts bf16->fp32
except attention, which converts bf16->fp16: the fp32 attention kernel is a
naive one-thread-per-row implementation with no warp-level cooperation and
is catastrophically slow at real sequence lengths (~21s for one S=17402
call), while the fp16 attention kernel is properly warp-cooperative
(~6.5ms for the same shape) -- it's the same kernel design later forked
into the sageattention/xformers CMP-Turing ports. Conv2d and Linear/matmul
are tiled, GEMM-style kernels. Every op is patched, and which
implementation actually runs -- the hand-tuned kernel or stock
cuDNN/cuBLAS -- is measured per shape at runtime rather than decided in
advance; see "Automatic kernel selection" below.

For additional info - go to the original repository.

# Warning

Shamelessly vibecoded (using Claude.ai) to make it run on Turing cards. Can contain errors. Can break the output.
Tested on CMP 50HX card and text-to-image models (SDXL & Anima), maxing out the power usage on this card.
Speed is at least tripled in comparison to the normal FP16 workloads.

# Compilation

For additional info - go to the original repository.
Release page contains a pre-compiled wheel for CUDA 12.8 + Python 3.12

## Automatic kernel selection (on by default)

A hand-tuned FMA-free kernel beating stock cuDNN/cuBLAS is a bet, and on
this hardware it is a bet that pays off spectacularly for some ops and
loses badly for others -- often depending on the *shape*, not just the op.
So cmpext3 doesn't decide up front. The first call for a given op and
shape runs both implementations, checks their outputs agree, times them,
and keeps the winner for every later call with that shape
(`cmpext3/autoselect.py`). The loser never runs again.

Measured on the CMP 50HX by exactly that mechanism:

| op | kept | vs stock |
|---|---|---|
| linear fp32          | cmpext3 | 13.56x |
| conv3d fp32          | cmpext3 | 12.29x |
| silu fp16            | cmpext3 |  5.62x |
| conv2d fp32          | cmpext3 |  4.56x |
| group_norm fp16      | cmpext3 |  2.45x |
| conv_transpose2d fp32| cmpext3 |  1.58x |
| attention fp32       | **stock** | 0.50x — the kernel is 2x slower |
| conv_transpose2d fp16| **stock** | 0.23x |
| interpolate fp32     | **stock** | 0.17x |
| interpolate fp16     | **stock** | 0.11x |

That table is the argument for the feature. conv3d, conv_transpose2d and
interpolate used to be switched off wholesale behind
`CMPEXT3_ENABLE_UNVERIFIED_KERNELS=1` because nobody had benchmarked them
-- which was leaving a 12x conv3d win and a 1.58x conv_transpose2d win on
the floor. fp32 attention was switched *on* at half of stock's speed for
the same reason in reverse. Both are now decided by the same evidence, and
neither needs a human to keep a table of verdicts up to date.

Correctness is checked before speed, and costs nothing extra: both outputs
already exist at probe time, so a kernel whose result doesn't match stock
is rejected no matter how fast it is. That is what makes it safe to stop
gating "unverified" kernels behind an environment variable. It is not
hypothetical -- it already rejects one real kernel: `fp16_conv3d.cu`
accumulates in half precision (`half2` + `__hfma2`, which is how it dodges
the FFMA throttle), so its error grows with channel count -- ~3.1 at 64
channels but ~25.5 at 512, against cuDNN's flat ~0.4. At 512 channels
that's 4.3% of the output's peak, so it's dropped there despite being 2.3x
faster, while still being used at 64 channels where it's accurate enough.

Cost: one probe per (op, shape), a few extra executions of an op you were
about to run anyway; and ~0.5us per call afterwards for the cached lookup
(under 1% for any op taking more than 50us, ~5% for the very smallest).
Decisions live for the life of the process.

- `CMPEXT3_AUTOSELECT=0` — never measure; always use the kernel where it's
  eligible (the behavior before this existed).
- `CMPEXT3_DISABLE_OPS=conv3d,interpolate` — pin named ops to stock.
- `CMPEXT3_AUTOSELECT_TOLERANCE=0.1` — how far a kernel may sit from stock
  and still be eligible. Raising it to 0.1 is what re-enables the fp16
  conv3d kernel above, at 2.3x and 4.3% error. Look at the output before
  you decide that's a good trade.
- `CMPEXT3_VERBOSE=1` — print each decision as it's made.
- `cmpext3.autoselect_report()` — dump every decision made so far. Also the
  quickest answer to "is cmpext3 doing anything for my workload?"

## conv3d fp32: Winograd, chosen per shape at runtime

fp32 conv3d has two kernels behind it, and picks between them per shape
while the model runs (`custom_conv3d_forward` in `src/main.cpp`):

- `src/cuda/fp32_conv3d.cu` -- the direct, register-blocked kernel. Handles
  everything.
- `src/cuda/fp32_conv3d_winograd.cu` -- F(2x2x2, 3x3x3) Winograd. ~5.6x
  fewer multiplies, but it only handles 3x3x3 / stride 1 / padding 1 /
  dilation 1 / batch 1 with even spatial dims, and it pays for two extra
  transform passes plus a U/V workspace that can run to hundreds of MB.

Which one is faster genuinely depends on the shape, so it is measured, not
assumed: the first call for a given shape runs both a few times, keeps the
winner for the rest of the process, and prints one line to stderr saying
what it picked. On the CMP 50HX the crossover sits around 32-64 channels at
8x32x32 -- below it the transform overhead outweighs the multiply saving:

    [cmpext3] conv3d fp32 Winograd [B1 C32 8x32x32 -> C32 ...]:  0.383 ms vs.  0.293 ms for the default kernel -- using the default kernel
    [cmpext3] conv3d fp32 Winograd [B1 C512 8x32x32 -> C512 ...]: 19.66 ms vs. 31.73 ms for the default kernel -- using it

At the HunyuanVideo-style VAE-decode shape (512ch, 8x32x32) that is 19.7ms
vs. the direct kernel's 31.7ms, and vs. 282ms for stock cuDNN on the same
call. Shapes Winograd can't handle cost one guard check on their first call
and a dictionary lookup after that.

Set `CMPEXT3_DISABLE_WINOGRAD=1` to pin fp32 conv3d to the direct kernel --
useful for bisecting a suspected Winograd bug, or for reproducing output
from a build that predates it. It is *not* an accuracy fallback. Measured
against a float64 reference at 32/64/128/256 channels, Winograd is the more
accurate of the two at every shape tried, and beats stock cuDNN as well:

| shape (8x32x32)  | Winograd | direct kernel | cuDNN |
|------------------|----------|---------------|-------|
| 64 ch            | 8.8e-05  | 2.9e-04       | 1.2e-04 |
| 128 ch           | 1.5e-04  | 7.1e-04       | 3.1e-04 |
| 256 ch           | 3.3e-04  | 1.4e-03       | 6.0e-04 |

(max abs error vs. float64; all of them are float32 rounding noise, ~1e-6
to 1e-5 relative to the output scale.) The direct kernel drives 27*C_in
products through one fp32 accumulator; Winograd keeps 64 accumulators of
C_in terms each, so it does ~27x fewer sequential additions per accumulator
and rounds off less. There is no "force Winograd on" knob: the benchmark
already refuses to use it where it doesn't win.

Verified by `tests_hardware/test_conv3d_perf.py` (Check 5), which asserts
both that Winograd is actually selected on a shape it should win and that
its output matches an unpatched stock cuDNN reference.

## Kernel tile autotuning (on by default)

`setup.py` autotunes six kernels' compile-time tile parameters before
building, instead of just using their hardcoded defaults:

    pip install -e . --no-build-isolation

Two families: the GEMM-tiled fp16 kernels (`fp16_conv3d.cu`, `fp16_conv.cu`
-- `BM`/`BN`/`BK`/`STAGES`) and the register-blocked kernels
(`fp32_conv.cu`, `fp32_conv3d.cu`, `fp32_ConvTranspose2d.cu`,
`fp16_ConvTranspose2d.cu` -- `CTILE`, output channels per thread). Each
target's candidates are compiled and benchmarked independently as standalone
executables against your actual GPU (~1-5 minutes total on a *first*
install), the fastest per target is kept, and the real extension is built
with those configs. Needs a real CUDA-capable GPU and `nvcc`; falls back
silently, per target, to that kernel's default (with a one-line notice) if
either is missing or if anything in that target's sweep fails -- a failure
on one kernel never blocks the others, and autotuning overall can never turn
a working build into a
broken one. The winning config is cached in `.cmpext3_autotune_cache.json`
(gitignored), so every install after the first is a cache hit and skips
straight to the real build.

- Skip autotuning and use the hardcoded default (e.g. for reproducible/CI
  builds): `CMPEXT3_AUTOTUNE=0 pip install -e . --no-build-isolation`
- Force a fresh sweep, ignoring any cached result: `CMPEXT3_AUTOTUNE_FORCE=1 pip install -e . --no-build-isolation`
  (or just delete `.cmpext3_autotune_cache.json`)

## Causal attention and RMSNorm (LLM-backbone support)

Two gaps that only mattered for transformer/LLM-style models (as opposed to
the conv-heavy diffusion UNets this project was originally built for) are
now closed:

- **`F.scaled_dot_product_attention(..., is_causal=True)`** used to be an
  unconditional bail-to-stock in `_patched_sdpa` -- `fp16_attention.cu` had
  no masking logic, only the plain (non-causal) case. Since an LLM-style
  decoder (e.g. a Qwen3-backbone TTS model) calls SDPA with `is_causal=True`
  on essentially every forward pass, cmpext3 never engaged for that
  attention at all. Causal masking is now built into the kernel's tile loop
  (per-block `causal_limit` early-exit + per-row/per-column masking in the
  inner accumulation loop), ported from `sageattention`'s own
  `turing_fma_free` fork of this exact kernel
  (`other/amd_tools/source/sageattention-1.0.6/sageattention/csrc/
  turing_fma_free/attention_fp16_turing.cu` -- a single-buffered version of
  `fp16_attention.cu` that independently added causal masking for the same
  reason: SageAttention's `sageattn()` exposes `is_causal`). Only the fp16,
  head_dim=128 path supports it; the naive fp32 kernel and the optional
  TensorRT attention engine still don't (`custom_attention_forward` raises /
  skips them for a causal call rather than silently dropping the mask), so
  a causal call on those falls back to stock SDPA exactly like an
  unsupported shape does everywhere else in this file.
- **`F.rms_norm`** is now patched (`_patched_rms_norm`), reusing the
  `rmsnorm` native kernel that already existed in `src/main.cpp`/`ops.py`
  for manual use but was never wired to the torch dispatch. RMSNorm (not
  LayerNorm) is what Qwen3/Llama-family transformer blocks actually use, so
  without this every norm in an LLM backbone was running on throttled stock
  regardless of what else cmpext3 patched. `weight=None` falls back to stock
  since the native kernel has no bias-free path (unlike `layer_norm`/
  `group_norm`); every real transformer block supplies a weight anyway.

Neither change touches `flash_attn`, `torch.nn.attention.flex_attention`, or
a `flashinfer`-based attention/RMSNorm path -- those call their own CUDA
extensions directly, bypassing `torch.nn.functional` entirely, so there is
nothing for a monkeypatch to intercept. A model using those (e.g.
`other/OmniVoice`'s `omnivoice_flashinfer.py` variant) only picks up the
`F.linear`/`matmul`/`F.silu`/`F.embedding` speedups from this project, not
attention or norm, regardless of the causal/RMSNorm work above -- covering
that would mean patching or replacing those libraries' own kernels, not
`torch.nn.functional`.

## `cmpext3.rope_ops` / `cmpext3.fused_norm_ops` (opt-in Triton, LLM backbones)

Two more manually-callable modules, for the same "no F.* stock op to
monkeypatch" reason `cmpext3.ops.group_norm_silu` exists:

- **`cmpext3.rope_ops.rotary_embedding(...)`** -- fused, in-place NeoX-style
  RoPE application for query/key (any NeoX-style-RoPE model: Llama, Qwen,
  Mistral, ...). Without this, a Qwen3-backbone model's own rotate-half RoPE
  runs as several separate unfused elementwise ops (multiply, `cat`,
  multiply, add) every token, every layer.
- **`cmpext3.fused_norm_ops.fused_add_rms_norm(...)`** -- fuses the
  `residual = residual + x; normed = rmsnorm(residual)` pre-norm epilogue
  every decoder layer runs twice per layer into one kernel, on top of what
  the plain `F.rms_norm` patch above already covers.

Both are ports of `other/amd_tools/source/cmp_ext_turing/amd_tuned_torch/
rope_ops.py` and `fused_norm_ops.py` -- written for RDNA3/ROCm, but the
kernels themselves are plain `@triton.jit` Triton, no HIP/CK/CUDA-C, and
Triton's CUDA backend is in fact the *original* one (ROCm support came
later), so the port is close to line-for-line. Both carry a real backward
pass (ported unchanged from upstream), so -- unlike every hand-written CUDA
kernel in this project, which is forward-only and grad-gated off -- these
work inside a training loop too.

**Turing + bf16**: Triton's bf16 codegen needs ptxas sm_80+ (Ampere);
below that (Turing/sm_75, i.e. every CMP card) it fails to compile at all
-- the same wall documented in `other/amd_tools/source/sageattention-1.0.6/
sageattention/_turing_compat.py` for SageAttention's own Triton kernels on
this hardware. Both modules detect a sub-sm_80 device and work around it:
`fused_add_rms_norm` (returns fresh tensors) simply upcasts bf16 to fp32
for the kernel call and casts the outputs back; `rotary_embedding` (mutates
query/key in place) upcasts, runs, and copies the fp32 result back into the
original bf16 tensors -- and refuses that copy-back (raises rather than
silently dropping gradients) when autograd is live, since the copy isn't
gradient-tracked.

Neither is imported by plain `import cmpext3` (Triton is not a hard
dependency of this package) -- opt in explicitly with
`import cmpext3.rope_ops` / `import cmpext3.fused_norm_ops` and check
`.available()` first. **Unvalidated on real hardware** -- ported and
syntax-checked but never run on an actual GPU (this environment has
neither CUDA nor Triton installed); see
`tests_hardware/test_triton_llm_ops.py` for the correctness/gradient
checks to run before trusting either one.

## `other/amd_tools`

A separate project (ROCm kernel tuning for a Radeon RX 7900 XTX / RDNA3,
`gfx1100`-only -- explicitly "not portable to CDNA or NVIDIA" per its own
README) vendored under `other/amd_tools/source/`. It doesn't run on a CMP
Turing card directly, but two of its vendored subtrees were directly useful
as source material for the causal-attention work above:

- `other/amd_tools/source/sageattention-1.0.6/` -- a fork of SageAttention
  that already carries a **CMP-Turing-specific `turing_fma_free` CUDA
  extension** (`sageattention/csrc/turing_fma_free/attention_fp16_turing.cu`
  + `pybind.cpp`), auto-detected via `sageattention/_turing_compat.py`'s
  `_is_turing_cmp_device()` (compute capability 7.5 + "CMP" in the device
  name) and dispatched from `sageattention/cmp_patch.py`. Its kernel is a
  single-buffered fork of this project's own `fp16_attention.cu` with
  causal masking added -- exactly the missing piece ported into
  `fp16_attention.cu` above. `_turing_compat.py` also documents a real
  Triton `num_stages` pitfall on Turing worth knowing about independently:
  Turing's 64KB/block shared-memory cap means SageAttention's hardcoded
  `num_stages=4` for head_dim=128 overflows and fails to compile there;
  `num_stages=2` is the confirmed-working value.
- `other/amd_tools/source/xformers-0.0.27.post2/` -- referenced by this
  project's own docstrings as another lineage that forked the same
  FMA-free Turing kernel design; not separately mined for this change since
  sageattention's fork already had the causal masking needed, but worth
  checking for anything else (e.g. a memory-efficient-attention path) if
  extending this further.
- Everything else under `amd_tools/` -- `amd_tuned_torch/` itself, the
  LoRA/LyCORIS training scripts, and `source/cmp_ext_turing/` +
  `source/cmp_ext_turing_old/` (despite the name, a *different*, ROCm/HIP/
  Composable-Kernel-based codebase -- `ck_conv_torch.cpp`,
  `ck_gemm_torch.cpp`, etc. -- not a copy of this project) -- is
  RDNA3/ROCm-specific and wasn't relevant to this NVIDIA-Turing change.

## `cmpext3.fftconv_ops` (FFT convolution, large kernels)

A third module ported from `other/amd_tools`, this time relevant to the
project's *original* conv2d/conv3d use case (SDXL/video-diffusion), not
the LLM-backbone work above: FFT-based convolution, for kernels too large
for the hand-tuned direct/Winograd CUDA kernels in `src/cuda/` to be
efficient at. Direct/im2col convolution is O(N*K) per spatial dim; FFT-conv
is O(N log N) -- past some kernel width the algorithmic win outweighs the
FFT/complex-multiply/inverse-FFT overhead.

Ported from `other/amd_tools/source/cmp_ext_turing/amd_tuned_torch/
fftconv_ops.py` (RDNA3/ROCm), but -- like `rope_ops`/`fused_norm_ops` above
-- the actual algorithm is pure PyTorch (`torch.fft.rfftn`/`irfftn`,
`F.pad`, `torch.kron`, `einsum`), so it isn't ROCm-specific: CUDA's
`torch.fft` already goes through cuFFT, same as ROCm's goes through
hipFFT. What was **not** ported: the ROCm-specific `rocfft_ops`/`vkfft_ops`
routing (no CUDA equivalent need -- torch.fft on CUDA already calls cuFFT
about as directly as a raw binding could), and the `kernel_select`-based
"pad the FFT to a smoother length" contest (real optimization, but built
entirely on amd_tuned_torch's own N-way measurement infrastructure, which
doesn't exist in cmpext3 -- `cmpext3/autoselect.py` is a strict 2-way
native-vs-stock chooser). `fft_conv` is still fully correct without it,
just without that extra speedup -- see `cmpext3/fftconv_ops.py`'s own
docstring for the full comparison.

**Two ways to use it:**
- **Directly**: `cmpext3.fftconv_ops.fft_conv1d/2d/3d/convnd(...)` --
  `F.convNd`-shaped functions for a caller building a long-kernel layer
  (Hyena-style global convolution, a long-kernel 1D audio/vocoder layer, a
  large 3D kernel in a video-diffusion VAE) to call explicitly.
- **Auto-patched**: `_patched_conv2d`/`_patched_conv3d` in
  `cmpext3/__init__.py` route to `fftconv_ops.fft_conv{2,3}d` instead of
  the native CUDA kernel once the kernel's widest spatial axis crosses
  `CMPEXT3_FFTCONV2D_MIN_KERNEL` (default 32) / `CMPEXT3_FFTCONV3D_MIN_KERNEL`
  (default 7) -- still measured against stock via the normal autoselect
  contract, under a separate op name (`"conv2d_fft"`/`"conv3d_fft"`) so a
  shape too small to win doesn't poison the small-kernel decision or vice
  versa. `CMPEXT3_FFTCONV2D=0` / `CMPEXT3_FFTCONV3D=0` disables the
  auto-patch path entirely (the direct functions are unaffected).

Both threshold defaults are amd_tuned_torch's own **RDNA3** measurements,
not a measurement on a CMP/Turing card -- treat them as starting points,
not calibrated values, until re-measured on real hardware (there is no
`fftconv_calibration.py`-equivalent persistence layer ported here either,
for the same "no kernel_select to build on" reason as the smooth-length
contest above).

**UNVALIDATED ON A REAL CMP/TURING CARD**, same as every other port in this
conversation: the algorithm itself is unchanged from upstream's own
(independently tested) `fft-conv-pytorch` lineage, but the mixed-precision
boundary and grouped/depthwise paths haven't been re-verified numerically
or for speed here. See `tests_hardware/test_fftconv_ops.py` for the
correctness checks (including the auto-patch actually engaging end to
end) to run before trusting this for a real large-kernel workload.

# License

MIT
