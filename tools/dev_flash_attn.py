"""Correctness + speed of the HFMA2 flash attention vs stock SDPA at SD1.5 shapes."""
import os, sys, glob, math, torch, torch.nn.functional as F
from torch.utils.cpp_extension import load
here = os.path.dirname(os.path.abspath(__file__))
cu13 = glob.glob(os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "cu13", "include"))
ext = load("dev_flash_attn_hfma2", [os.path.join(here, "dev_flash_attn_ext.cpp"),
           os.path.join(here, "..", "src", "cuda", "fp16_flash_attention_hfma2.cu")],
           extra_cuda_cflags=["-O3", "--use_fast_math", "--fmad=false", "-gencode=arch=compute_75,code=sm_75"],
           extra_include_paths=cu13, build_directory=os.path.join(here, "..", "bench_logs", "dev_ext"), verbose=False)
def t(fn, n=5):
    for _ in range(2): fn()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / n
torch.manual_seed(0)
cases = [(2, 8, 8192, 8192, 40), (2, 8, 8192, 308, 40), (2, 8, 2048, 2048, 80), (2, 8, 2048, 308, 80),
         (2, 8, 512, 512, 160), (2, 8, 512, 308, 160), (1, 3, 1000, 77, 64), (2, 4, 777, 1003, 128)]
for B, H, Sq, Sk, D in cases:
    # webui/diffusers layout: [B, S, H, D] memory viewed as [B, H, S, D]
    q = torch.randn(B, Sq, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)
    k = torch.randn(B, Sk, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)
    v = torch.randn(B, Sk, H, D, device="cuda", dtype=torch.float16).transpose(1, 2)
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    out = ext.attn(q, k, v, 1 / math.sqrt(D)).float()
    stock16 = F.scaled_dot_product_attention(q, k, v).float()
    err = (out - ref).abs().max().item(); err16 = (stock16 - ref).abs().max().item()
    ts, tn = t(lambda: F.scaled_dot_product_attention(q, k, v)), t(lambda: ext.attn(q, k, v, 1 / math.sqrt(D)))
    gf = 4 * B * H * Sq * Sk * D / 1e9
    print(f"B{B} H{H} Sq{Sq:5d} Sk{Sk:5d} D{D:3d}: stock {ts:7.2f} ms  new {tn:7.2f} ms ({gf/tn:5.2f} TFLOPS) "
          f"x{ts/tn:4.2f}  maxerr new {err:.4f} / stock-fp16 {err16:.4f}")
