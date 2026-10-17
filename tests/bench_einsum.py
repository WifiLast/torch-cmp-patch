"""Stock torch.einsum vs cmpext3's einsum (planned onto the patched bmm) on the installed GPU."""
import time

import torch
import cmpext3

STOCK = cmpext3._ORIGINALS[(torch, "einsum")]
CASES = [
    ("ij,mjc->mic", [(24, 24), (100000, 24, 2)], "FEM element forces"),
    ("mij,mjc->mic", [(20000, 24, 24), (20000, 24, 3)], "per-element stiffness"),
    ("bhqd,bhkd->bhqk", [(2, 16, 1024, 64), (2, 16, 1024, 64)], "attention scores"),
    ("bhqk,bhkd->bhqd", [(2, 16, 1024, 1024), (2, 16, 1024, 64)], "attention values"),
    ("bnd,dk->bnk", [(8, 4096, 512), (512, 512)], "token projection"),
    ("ij,jk->ik", [(2048, 2048), (2048, 2048)], "square matmul"),
]


def timed(fn, n=20):
    """GPU time per call from CUDA events (host timers include wake-up and sync noise)."""
    fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n


torch.randn(1, device="cuda").sum().item()  # wake the GPU before the first autoselect probe
print(torch.cuda.get_device_name(0))
for dt in (torch.float32, torch.float16):
    for eq, shapes, what in CASES:
        ops = [torch.randn(*s, device="cuda", dtype=dt) for s in shapes]
        torch.einsum(eq, *ops)  # first call: autoselect probes and keeps the faster one
        t_stock = timed(lambda: STOCK(eq, *ops))
        t_new = timed(lambda: torch.einsum(eq, *ops))
        print(f"{str(dt)[6:]:<8} {eq:<18} {what:<22} stock {t_stock:8.2f} ms  cmpext3 {t_new:8.2f} ms  "
              f"x{t_stock / t_new:5.2f}")
