"""Fused rotary positional embedding (RoPE), NeoX-style, applied in-place to
query and key -- ported from other/amd_tools/source/cmp_ext_turing/
amd_tuned_torch/rope_ops.py (itself vendored from Conch; see that file's
NOTICE.md for upstream provenance), which wrote it for RDNA3/ROCm. The
kernel is plain Triton -- no HIP/CK/CUDA-C -- so it is not actually
ROCm-specific: Triton's CUDA backend is the original one (ROCm support was
added later), so this ports here basically unchanged. See the Turing/bf16
caveat below for the one thing that IS backend-specific.

WHY THIS EXISTS HERE. Unlike cmpext3's other ops (raw CUDA kernels wired
into src/main.cpp, patched onto stock torch/F.* calls), there is no
F.rotary_embedding to monkeypatch -- every model applies RoPE its own way
(rotate-half via cat/mul/add in plain PyTorch, usually). A Qwen3-backbone
model (e.g. other/OmniVoice's non-flashinfer path) runs that unfused,
several-elementwise-ops version twice per decoder layer, every token. This
is a manually-callable replacement for that -- call it directly from a
model's attention layer in place of its own rotate-half application,
analogous to how cmpext3.ops.group_norm_silu is called directly rather than
patched onto anything (see cmpext3/__init__.py's module docstring).

NeoX-style, full-head rotation ONLY (rotary_dim == head_size) -- see
rotary_embedding()'s assertion. Qwen3/Llama/Mistral-family models use this
convention; a model with partial rotary_pct < 1.0 (some GPT-NeoX/Phi
configs) cannot use this as-is.

TURING + BF16: Triton's bf16 codegen requires ptxas sm_80+ (Ampere) --
ptxas rejects `.bf16` PTX below that, a hardware ISA limit, not a
throttle (see other/amd_tools/source/sageattention-1.0.6/sageattention/
_turing_compat.py's bf16_triton_unsupported(), which documents the exact
same wall for SageAttention's Triton kernels on this same hardware).
rotary_embedding() below detects a Turing/pre-Ampere device and, for bf16
inputs, upcasts to fp32 for the kernel call and copies the result back
into the original bf16 tensors -- preserving the "mutates query/key in
place" contract without ever asking Triton to compile bf16 code on sm_75.
fp16/fp32 need no such workaround and run the kernel directly on the
original tensors.

HAS A REAL BACKWARD PASS (ported unchanged from upstream): a rotation is
orthogonal, so the gradient is the same kernel called with sin negated
(INVERSE=True). This is usable in a training loop, unlike cmpext3's own
CUDA kernels (forward-only, grad-gated off in cmpext3/__init__.py).

UNVALIDATED ON A REAL CMP/TURING CARD: ported without hardware access to
test on. Before relying on this: compare its output against a manual
rotate-half RoPE application (mathematically identical for the full-head
case) numerically, for your actual model's head_size/num_heads/dtype, and
benchmark against whatever unfused RoPE your model currently runs.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def available() -> bool:
    return _TRITON_AVAILABLE


def bf16_triton_unsupported(device) -> bool:
    """True if this device's compute capability can't JIT-compile Triton's
    bf16 codegen at all (sm_75/Turing and older). See module docstring."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major < 8


if _TRITON_AVAILABLE:

    @triton.jit
    def _rotary_embedding_kernel(
        positions_ptr, query_ptr, key_ptr, cos_sin_cache_ptr,
        rot_dim, query_stride, key_stride,
        num_heads, num_kv_heads, head_size,
        cxpr_block_size: tl.constexpr,
        INVERSE: tl.constexpr = False,
    ):
        # One program per token. cos_sin_cache_ptr is [max_position,
        # rot_dim] with rot_dim split into [cos (rot_dim/2) | sin
        # (rot_dim/2)] -- see compute_cos_sin_cache below for the layout
        # this expects.
        token_idx = tl.program_id(0)
        pos = tl.load(positions_ptr + token_idx)
        rot_cache_ptr = cos_sin_cache_ptr + pos * rot_dim

        embed_dim = rot_dim // 2
        cos_ptr = rot_cache_ptr
        sin_ptr = rot_cache_ptr + embed_dim

        nq = num_heads * embed_dim
        query_token_offset = token_idx.to(tl.int64) * query_stride
        i = tl.arange(0, cxpr_block_size)
        for _ in tl.range(0, nq, cxpr_block_size):
            head_idx = i // embed_dim
            token_head = query_token_offset + head_idx * head_size
            rot_offset = i % embed_dim

            x_index = rot_offset
            y_index = rot_offset + embed_dim
            query_offset = query_ptr + token_head
            mask = i < nq
            x = tl.load(query_offset + x_index, mask=mask)
            y = tl.load(query_offset + y_index, mask=mask)
            cos = tl.load(cos_ptr + x_index, mask=mask)
            sin = tl.load(sin_ptr + x_index, mask=mask)
            if INVERSE:
                sin = -sin
            x_rot = x * cos - y * sin
            y_rot = y * cos + x * sin
            tl.store(query_offset + x_index, x_rot, mask=mask)
            tl.store(query_offset + y_index, y_rot, mask=mask)
            i += cxpr_block_size

        nk = num_kv_heads * embed_dim
        key_token_offset = token_idx.to(tl.int64) * key_stride
        j = tl.arange(0, cxpr_block_size)
        for _ in tl.range(0, nk, cxpr_block_size):
            head_idx = j // embed_dim
            token_head = key_token_offset + head_idx * head_size
            rot_offset = j % embed_dim

            x_index = rot_offset
            y_index = rot_offset + embed_dim
            key_offset = key_ptr + token_head
            mask = j < nk
            x = tl.load(key_offset + x_index, mask=mask)
            y = tl.load(key_offset + y_index, mask=mask)
            cos = tl.load(cos_ptr + x_index, mask=mask)
            sin = tl.load(sin_ptr + x_index, mask=mask)
            if INVERSE:
                sin = -sin
            x_rot = x * cos - y * sin
            y_rot = y * cos + x * sin
            tl.store(key_offset + x_index, x_rot, mask=mask)
            tl.store(key_offset + y_index, y_rot, mask=mask)
            j += cxpr_block_size


def compute_cos_sin_cache(base: float, rotary_dim: int, max_position_embeddings: int) -> torch.Tensor:
    """[max_position_embeddings, rotary_dim] cache, laid out as
    cat([cos, sin], dim=-1) -- the layout rotary_embedding()'s kernel
    expects. Build once (e.g. at model init) and reuse across every
    decode/prefill step; recomputing per call defeats the point of a fused
    kernel."""
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    t = torch.arange(max_position_embeddings, dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


if _TRITON_AVAILABLE:

    class _RotaryEmbeddingFunction(torch.autograd.Function):
        """Real backward pass: RoPE's forward is `y = R(theta) @ x` for each
        rotated pair, and a rotation matrix is orthogonal
        (R(theta)^-1 == R(theta)^T == R(-theta)), so the gradient w.r.t. x
        is exactly `R(-theta) @ grad_y` -- the SAME kernel, called again
        with INVERSE=True (sin negated)."""

        @staticmethod
        def forward(ctx, positions, query, key, head_size, cos_sin_cache):
            rot_dim = cos_sin_cache.shape[-1]
            num_heads = query.shape[-1] // head_size
            num_kv_heads = key.shape[-1] // head_size
            query_stride = query.stride(-2)
            key_stride = key.stride(-2)
            block = triton.next_power_of_2(head_size)
            _rotary_embedding_kernel[(query.shape[0],)](
                positions, query, key, cos_sin_cache,
                rot_dim, query_stride, key_stride,
                num_heads, num_kv_heads, head_size,
                block, INVERSE=False,
            )
            ctx.mark_dirty(query, key)
            ctx.save_for_backward(positions, cos_sin_cache)
            ctx.head_size = head_size
            ctx.rot_dim = rot_dim
            ctx.num_heads = num_heads
            ctx.num_kv_heads = num_kv_heads
            ctx.block = block
            return query, key

        @staticmethod
        def backward(ctx, grad_query, grad_key):
            positions, cos_sin_cache = ctx.saved_tensors
            _rotary_embedding_kernel[(grad_query.shape[0],)](
                positions, grad_query, grad_key, cos_sin_cache,
                ctx.rot_dim, grad_query.stride(-2), grad_key.stride(-2),
                ctx.num_heads, ctx.num_kv_heads, ctx.head_size,
                ctx.block, INVERSE=True,
            )
            return None, grad_query, grad_key, None, None


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply NeoX-style rotary embedding to query and key IN PLACE (matches
    upstream vLLM/Conch semantics -- the same tensors passed in are mutated
    and also returned, for call-site convenience).

    positions: [num_tokens] token position indices into cos_sin_cache.
    query: [num_tokens, num_heads * head_size].
    key: [num_tokens, num_kv_heads * head_size] -- num_kv_heads may differ
        from num_heads (GQA/MQA), derived from key.shape[-1] // head_size.
    head_size: must equal cos_sin_cache.shape[-1] (full-head rotation only).
    cos_sin_cache: [max_position, head_size] from compute_cos_sin_cache().

    Both query and key must be 2D (unbatched: batch and sequence already
    flattened into num_tokens) -- reshape a (batch, seq, num_heads,
    head_size) tensor to (batch*seq, num_heads*head_size) before calling.

    bf16 on a Turing/pre-Ampere GPU (compute capability < 8.0) is handled
    by upcasting to fp32 for the kernel call and copying the result back
    into the original bf16 tensors -- see the TURING + BF16 note in the
    module docstring for why Triton itself can't compile bf16 code there."""
    if not available():
        raise RuntimeError("rope_ops.rotary_embedding: triton is not importable")
    assert positions.dim() == 1, "positions must be 1D [num_tokens]"
    assert query.dim() == 2 and key.dim() == 2, "query/key must be 2D [num_tokens, num_heads*head_size]"
    assert query.shape[0] == key.shape[0] == positions.shape[0], "num_tokens must match across positions/query/key"
    rot_dim = cos_sin_cache.shape[-1]
    assert rot_dim == head_size, (
        "rope_ops only supports full-head rotation (rotary_dim == head_size); "
        f"got cos_sin_cache rot_dim={rot_dim}, head_size={head_size}"
    )
    assert query.shape[-1] % head_size == 0 and key.shape[-1] % head_size == 0, \
        "query/key last dim must be an exact multiple of head_size"

    if query.dtype == torch.bfloat16 and query.is_cuda and bf16_triton_unsupported(query.device):
        # The upcast-copy-downcast workaround below uses two in-place
        # `.copy_()` calls that are NOT autograd-tracked back through the
        # fp32 copies -- fine for inference (the common case for this
        # project), wrong under autograd (gradients would silently stop at
        # the copy). Only take it when nothing here is actually
        # differentiating; otherwise fail loudly instead of silently
        # dropping gradients.
        grad_live = torch.is_grad_enabled() and any(
            isinstance(t, torch.Tensor) and t.requires_grad for t in (query, key, cos_sin_cache)
        )
        if grad_live:
            raise RuntimeError(
                "rope_ops.rotary_embedding: bf16 on this GPU (compute capability "
                f"{torch.cuda.get_device_capability(query.device)}) needs the fp32 upcast "
                "workaround for Triton's sm_80+ bf16 codegen requirement, but that workaround "
                "isn't autograd-safe. Call this under torch.no_grad()/inference_mode(), or use "
                "fp16/fp32 query/key instead."
            )
        q32 = query.float()
        k32 = key.float()
        cache32 = cos_sin_cache.float() if cos_sin_cache.dtype != torch.float32 else cos_sin_cache
        _RotaryEmbeddingFunction.apply(positions, q32, k32, head_size, cache32)
        query.copy_(q32)
        key.copy_(k32)
        return query, key

    return _RotaryEmbeddingFunction.apply(positions, query, key, head_size, cos_sin_cache)
