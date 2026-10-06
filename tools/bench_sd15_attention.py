"""Stock SDPA cost at SD1.5's attention shapes, 512x1024, CFG batch 2, ctx 308.
Per UNet call: 5 transformer blocks at 64x128 and 32x64, 6 at 16x32 (incl. mid);
each block = 1 self-attention + 1 cross-attention. 8 heads."""
import os, sys, torch, torch.nn.functional as F
B, H, CTX = 2, 8, 308
levels = [(8192, 40, 5), (2048, 80, 5), (512, 160, 6)]   # (tokens, head_dim, blocks)
backend = sys.argv[1] if len(sys.argv) > 1 else "default"
from torch.nn.attention import sdpa_kernel, SDPBackend
ctxmgr = {"default": None, "math": SDPBackend.MATH, "efficient": SDPBackend.EFFICIENT_ATTENTION,
          "flash": SDPBackend.FLASH_ATTENTION}[backend]
def t(fn, n=5):
    for _ in range(2): fn()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / n
total = 0
for n, hd, blocks in levels:
    q = torch.randn(B, H, n, hd, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, n, hd, device="cuda", dtype=torch.float16)
    kc = torch.randn(B, H, CTX, hd, device="cuda", dtype=torch.float16)
    def run(kk):
        if ctxmgr is None: return F.scaled_dot_product_attention(q, kk, kk)
        with sdpa_kernel(ctxmgr): return F.scaled_dot_product_attention(q, kk, kk)
    try:
        s_ms, c_ms = t(lambda: run(k)), t(lambda: run(kc))
    except RuntimeError as ex:
        print(f"{n} tok hd{hd}: {backend} unavailable: {str(ex)[:80]}"); continue
    total += blocks * (s_ms + c_ms)
    print(f"{n:5d} tok hd{hd:3d}: self {s_ms:7.2f} ms  cross {c_ms:6.2f} ms  x{blocks} blocks = {blocks*(s_ms+c_ms):7.1f} ms")
print(f"{backend}: attention per UNet call = {total:.1f} ms")
