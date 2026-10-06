"""Correctness + speed of the HFMA2 linear vs stock and the old cmpext3 kernel at SD1.5 shapes."""
import os, glob, torch, torch.nn.functional as F
from torch.utils.cpp_extension import load
here = os.path.dirname(os.path.abspath(__file__))
cu13 = glob.glob(os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "cu13", "include"))
ext = load("dev_linear_hfma2", [os.path.join(here, "dev_linear_ext.cpp"),
           os.path.join(here, "..", "src", "cuda", "fp16_linear_hfma2.cu")],
           extra_cuda_cflags=["-O3", "--use_fast_math", "--fmad=false", "-gencode=arch=compute_75,code=sm_75"],
           extra_include_paths=cu13, build_directory=os.path.join(here, "..", "bench_logs", "dev_lin"), verbose=False)
import cmpext3
old = cmpext3._native
def t(fn, n=10):
    for _ in range(3): fn()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / n
torch.manual_seed(0)
cases = [((2, 8192, 320), 2560), ((2, 8192, 320), 320), ((2, 8192, 1280), 320), ((2, 2048, 640), 5120),
         ((2, 2048, 640), 640), ((2, 512, 1280), 10240), ((2, 512, 1280), 1280), ((2, 308, 768), 320),
         ((2, 308, 768), 1280), ((2, 1280), 1280)]
for xs, M in cases:
    x = torch.randn(*xs, device="cuda", dtype=torch.float16)
    w = torch.randn(M, xs[-1], device="cuda", dtype=torch.float16) / xs[-1] ** 0.5
    b = torch.randn(M, device="cuda", dtype=torch.float16)
    ref = F.linear(x.float(), w.float(), b.float())
    out = ext.lin(x, w, b).float()
    rel = ((out - ref).abs().max() / ref.abs().max()).item()
    ts, tn = t(lambda: F.linear(x, w, b)), t(lambda: ext.lin(x, w, b))
    to = t(lambda: old.linear(x, w, b))
    n = x.numel() // xs[-1]; gf = 2 * n * M * xs[-1] / 1e9
    print(f"{str(xs):18s} -> {M:5d}: stock {ts:6.2f}  old {to:6.2f}  new {tn:6.2f} ms ({gf/tn:5.2f} TFLOPS)  rel.err {rel:.1e}")
