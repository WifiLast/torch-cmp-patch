"""einsum_ops (einsum -> permute/reshape + bmm) against stock torch.einsum.

Run: python tests/test_einsum.py  (CPU always; CUDA fp32/fp16 + the installed patch when a GPU is present)
"""
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("einsum_ops_src", os.path.join(HERE, "..", "cmpext3", "einsum_ops.py"))
eo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eo)

# the stock einsum, even when cmpext3 has already patched torch.einsum
try:
    import cmpext3
    STOCK = cmpext3._ORIGINALS.get((torch, "einsum"), torch.einsum)
except Exception:  # noqa: BLE001
    cmpext3 = None
    STOCK = torch.einsum

CASES = [
    ("ij,jk->ik", [(5, 7), (7, 3)]),                       # plain matmul
    ("ij,jk", [(5, 7), (7, 3)]),                           # implicit output
    ("bij,bjk->bik", [(4, 5, 7), (4, 7, 3)]),              # batched
    ("ij,kj->ik", [(5, 7), (3, 7)]),                       # A @ B^T
    ("ji,jk->ik", [(7, 5), (7, 3)]),                       # A^T @ B
    ("ij,mjc->mic", [(24, 24), (50, 24, 2)]),              # FEM element forces (KE @ u_e)
    ("mij,mjc->mic", [(50, 24, 24), (50, 24, 3)]),         # per-element stiffness
    ("mic,ij,mjc->m", [(50, 24, 2), (24, 24), (50, 24, 2)]),  # element energy, 3 operands
    ("bhqd,bhkd->bhqk", [(2, 3, 9, 8), (2, 3, 11, 8)]),    # attention scores
    ("bhqk,bhkd->bhqd", [(2, 3, 9, 11), (2, 3, 11, 8)]),   # attention values
    ("i,i->", [(13,), (13,)]),                             # dot product
    ("i,j->ij", [(4,), (6,)]),                             # outer product
    ("ij,j->i", [(5, 7), (7,)]),                           # matrix-vector
    ("ijk,ik->j", [(3, 4, 5), (3, 5)]),                    # contraction over two indices
    ("ab,bc,cd->ad", [(3, 4), (4, 5), (5, 6)]),            # chain
    ("abc,c->ab", [(2, 1, 4), (4,)]),                      # broadcasting size-1 dim
    ("ij,kl->ik", [(3, 4), (5, 6)]),                       # lone indices summed out
    ("ij->ji", [(3, 4)]),                                  # single operand transpose
    ("ijk->k", [(3, 4, 5)]),                               # single operand reduction
]
UNSUPPORTED = [("ii->i", [(4, 4)]), ("...ij,...jk->...ik", [(2, 3, 4), (2, 4, 5)])]


def run(device, dtype, tol):
    g = torch.Generator(device="cpu").manual_seed(0)
    bad = 0
    for eq, shapes in CASES:
        ops = [torch.randn(*s, generator=g, dtype=torch.float64).to(device=device, dtype=dtype) for s in shapes]
        ref = STOCK(eq, *ops)
        got = eo.einsum(eq, ops)
        err = (got.double() - ref.double()).abs().max().item() / max(ref.double().abs().max().item(), 1e-12)
        ok = got.shape == ref.shape and err <= tol
        bad += not ok
        print(f"  {'ok ' if ok else 'BAD'} {eq:<22} {str(tuple(got.shape)):<16} rel err {err:.2e}")
    for eq, shapes in UNSUPPORTED:
        ops = [torch.randn(*s, dtype=torch.float32).to(device) for s in shapes]
        r = eo.supported(eq, ops)
        print(f"  {'ok ' if r else 'BAD'} {eq:<22} falls back to stock ({r})")
        bad += r is None
    return bad


bad = 0
print("CPU float64")
bad += run("cpu", torch.float64, 1e-12)
if torch.cuda.is_available():
    for dt, tol in ((torch.float32, 1e-5), (torch.float16, 5e-3)):
        print(f"CUDA {dt}")
        bad += run("cuda", dt, tol)
    if cmpext3 is not None and getattr(torch.einsum, "__name__", "") == "_patched_einsum":
        a, b = torch.randn(64, 24, 24, device="cuda"), torch.randn(64, 24, 3, device="cuda")
        err = (torch.einsum("mij,mjc->mic", a, b) - STOCK("mij,mjc->mic", a, b)).abs().max().item()
        print(f"patched torch.einsum through cmpext3 dispatch: max abs err {err:.2e}")
        bad += err > 1e-3
print("FAILED" if bad else "all passed")
sys.exit(1 if bad else 0)
