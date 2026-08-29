
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

# License

MIT
