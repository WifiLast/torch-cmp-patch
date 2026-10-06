"""Correctness + speed of the HFMA2 conv2d vs stock and the old cmpext3 kernel at SD1.5 shapes."""
import os, glob, torch, torch.nn.functional as F
from torch.utils.cpp_extension import load
here = os.path.dirname(os.path.abspath(__file__))
cu13 = glob.glob(os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "cu13", "include"))
ext = load("dev_conv2d_hfma2", [os.path.join(here, "dev_conv2d_ext.cpp"),
           os.path.join(here, "..", "src", "cuda", "fp16_conv2d_hfma2.cu")],
           extra_cuda_cflags=["-O3", "--use_fast_math", "--fmad=false", "-gencode=arch=compute_75,code=sm_75",
                              "-Xptxas=-v"],
           extra_include_paths=cu13, build_directory=os.path.join(here, "..", "bench_logs", "dev_conv"), verbose=False)
import cmpext3
old = cmpext3._native
def t(fn, n=5):
    for _ in range(2): fn()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / n
torch.manual_seed(0)
# (N, Ci, H, W, Co, k, stride, pad) -- SD1.5 at 512x1024, CFG batch 2
cases = [(2, 320, 128, 64, 320, 3, 1, 1), (2, 960, 128, 64, 320, 3, 1, 1), (2, 640, 64, 32, 640, 3, 1, 1),
         (2, 1280, 32, 16, 1280, 3, 1, 1), (2, 2560, 32, 16, 1280, 3, 1, 1), (2, 1280, 16, 8, 1280, 3, 1, 1),
         (2, 320, 128, 64, 320, 3, 2, 1), (2, 960, 128, 64, 320, 1, 1, 0), (2, 4, 128, 64, 320, 3, 1, 1),
         (2, 320, 128, 64, 4, 3, 1, 1)]
for N, Ci, H, W, Co, k, s, p in cases:
    x = torch.randn(N, Ci, H, W, device="cuda", dtype=torch.float16)
    w = torch.randn(Co, Ci, k, k, device="cuda", dtype=torch.float16) / (Ci * k * k) ** 0.5
    b = torch.randn(Co, device="cuda", dtype=torch.float16)
    ref = F.conv2d(x.float(), w.float(), b.float(), s, p)
    out = ext.conv(x, w, b, s, p).float()
    rel = ((out - ref).abs().max() / ref.abs().max()).item()
    ts = t(lambda: F.conv2d(x, w, b, s, p)); tn = t(lambda: ext.conv(x, w, b, s, p))
    try:
        to = t(lambda: old.conv2d(x, w, b, [s, s], [p, p], [1, 1], 1)); old_s = f"{to:7.2f}"
    except Exception as ex:
        old_s = "   n/a"
    Ho, Wo = (H + 2 * p - k) // s + 1, (W + 2 * p - k) // s + 1
    gf = 2 * N * Ho * Wo * Co * Ci * k * k / 1e9
    print(f"{Ci:4d}->{Co:4d} k{k} s{s} {H:3d}x{W:3d}: stock {ts:7.2f}  old {old_s}  new {tn:7.2f} ms "
          f"({gf/tn:5.2f} TFLOPS)  rel.err {rel:.1e}")
