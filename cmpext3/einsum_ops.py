"""torch.einsum as batched matrix products, so the FMA-free bmm kernels apply to it.

Stock einsum lowers to cuBLAS bmm internally (C++), which the Python-level torch.bmm patch never sees, so
every einsum on a CMP card ran on the throttled path. Here an equation is planned as permute/reshape ->
(B, M, K) @ (B, K, N) -> reshape/permute, with the bmm going through torch.bmm (patched by cmpext3, so its
own autoselect picks the faster kernel per shape). The __init__ wrapper measures the whole rewrite against
stock einsum once per signature (autoselect), so a rewrite that loses (e.g. because of the permute copies)
is never used again.

Supported: any number of operands with explicit or implicit output, letters a-z/A-Z, broadcasting of
size-1 dims. Not supported (-> stock einsum): ellipsis "...", an index repeated inside one operand
(diagonals/traces), an output index that appears in no operand.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch


class Unsupported(Exception):
    pass


def parse(equation: str, n_operands: int) -> Tuple[List[str], str]:
    eq = equation.replace(" ", "")
    if "." in eq:
        raise Unsupported("ellipsis")
    if "->" in eq:
        lhs, out = eq.split("->")
    else:  # implicit output: indices appearing exactly once, alphabetical
        lhs = eq
        counts = {}
        for ch in lhs.replace(",", ""):
            counts[ch] = counts.get(ch, 0) + 1
        out = "".join(sorted(ch for ch, c in counts.items() if c == 1))
    terms = lhs.split(",")
    if len(terms) != n_operands:
        raise Unsupported("operand count")
    for t in terms:
        if len(set(t)) != len(t):
            raise Unsupported("repeated index in one operand")
    if len(set(out)) != len(out) or any(ch not in lhs for ch in out):
        raise Unsupported("bad output")
    return terms, out


def _sizes(terms: Sequence[str], ops: Sequence[torch.Tensor]) -> dict:
    size = {}
    for t, x in zip(terms, ops):
        if x.dim() != len(t):
            raise Unsupported("rank mismatch")
        for ch, n in zip(t, x.shape):
            n = int(n)
            if ch in size and size[ch] != n and 1 not in (size[ch], n):
                raise Unsupported("size mismatch")
            size[ch] = max(size.get(ch, 1), n)
    return size


def _drop_lone(x: torch.Tensor, t: str, other: str, out: str) -> Tuple[torch.Tensor, str]:
    lone = [ch for ch in t if ch not in other and ch not in out]
    if lone:
        x = x.sum(dim=[t.index(ch) for ch in lone])
        t = "".join(ch for ch in t if ch not in lone)
    return x, t


def _pair(a: torch.Tensor, ta: str, b: torch.Tensor, tb: str, out: str, size: dict, bmm) -> torch.Tensor:
    """Two-operand contraction a[ta] * b[tb] -> [out] as one batched matmul."""
    # indices only in one operand and not wanted later: sum them out first
    a, ta = _drop_lone(a, ta, tb, out)
    b, tb = _drop_lone(b, tb, ta, out)
    batch = [ch for ch in out if ch in ta and ch in tb]
    contr = [ch for ch in ta if ch in tb and ch not in out]
    fa = [ch for ch in out if ch in ta and ch not in tb]
    fb = [ch for ch in out if ch in tb and ch not in ta]
    # broadcast size-1 dims to the common size (expand is free; reshape copies only when needed)
    a = a.expand(tuple(size[ch] for ch in ta))
    b = b.expand(tuple(size[ch] for ch in tb))
    pa = a.permute(tuple(ta.index(ch) for ch in batch + fa + contr))
    pb = b.permute(tuple(tb.index(ch) for ch in batch + contr + fb))
    nb = 1
    for ch in batch:
        nb *= size[ch]
    m = 1
    for ch in fa:
        m *= size[ch]
    k = 1
    for ch in contr:
        k *= size[ch]
    n = 1
    for ch in fb:
        n *= size[ch]
    r = bmm(pa.reshape(nb, m, k), pb.reshape(nb, k, n))
    r = r.reshape(tuple(size[ch] for ch in batch + fa + fb))
    have = batch + fa + fb
    return r.permute(tuple(have.index(ch) for ch in out)) if have != list(out) else r


def einsum(equation: str, operands: Sequence[torch.Tensor], bmm=None) -> torch.Tensor:
    """einsum via batched matmuls; raises Unsupported for equations it does not plan."""
    bmm = bmm or torch.bmm
    terms, out = parse(equation, len(operands))
    size = _sizes(terms, operands)
    ops = list(operands)
    if len(ops) == 1:
        x, t = ops[0], terms[0]
        lone = [ch for ch in t if ch not in out]
        if lone:
            x = x.sum(dim=[t.index(ch) for ch in lone])
            t = "".join(ch for ch in t if ch not in lone)
        return x.permute(tuple(t.index(ch) for ch in out))
    # left to right: keep only the indices still needed by the remaining operands or the output
    x, tx = ops[0], terms[0]
    for i in range(1, len(ops)):
        rest = set(out).union(*[set(t) for t in terms[i + 1:]]) if i + 1 < len(ops) else set(out)
        keep = "".join(ch for ch in dict.fromkeys(tx + terms[i]) if ch in rest)
        x = _pair(x, tx, ops[i], terms[i], keep if i + 1 < len(ops) else out, size, bmm)
        tx = keep if i + 1 < len(ops) else out
    return x


def supported(equation: str, operands: Sequence[torch.Tensor]) -> Optional[str]:
    """None if einsum() can plan it, else the reason."""
    try:
        terms, _ = parse(equation, len(operands))
        _sizes(terms, operands)
        return None
    except Unsupported as e:
        return str(e)
