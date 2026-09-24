"""Fused residual-add + RMSNorm -- the standard pre-norm transformer
epilogue (`hidden = residual + sublayer_out; normed = rmsnorm(hidden)`,
with `hidden` becoming the next layer's residual) done as a single kernel
instead of two full read-modify-write passes over (batch*seq, hidden_dim)
tensors. Every decoder layer of every transformer model runs this once per
sublayer (attention and FFN).

Ported from other/amd_tools/source/cmp_ext_turing/amd_tuned_torch/
fused_norm_ops.py, which wrote it for RDNA3/ROCm. The kernel is plain
Triton -- no HIP/CK/CUDA-C at all -- so despite being written for a Radeon
card, it isn't actually ROCm-specific; Triton's CUDA backend is the
original one, so it ports here basically unchanged. See the Turing/bf16
caveat below for the one thing that IS backend-specific.

WHY THIS EXISTS HERE. cmpext3's own `_patched_rms_norm` (cmpext3/__init__.py)
accelerates F.rms_norm itself but has no way to fuse in the preceding
residual add -- F.rms_norm has no such argument to key off of, same reason
cmpext3.ops.group_norm_silu (fused GroupNorm+SiLU) exists as a manually-
callable helper rather than a monkeypatch target (see that function's
docstring in cmpext3/__init__.py). This is that same pattern applied to the
`residual + x` -> RMSNorm epilogue every Qwen3/Llama-family decoder layer
runs twice per layer -- call it directly from a model's decoder-layer
forward in place of the unfused add + F.rms_norm, exactly as
group_norm_silu replaces F.group_norm + F.silu in a diffusion UNet block.

TURING + BF16: Triton's bf16 codegen requires ptxas sm_80+ (Ampere) --
see other/amd_tools/source/sageattention-1.0.6/sageattention/
_turing_compat.py's bf16_triton_unsupported() for the same wall hit by
SageAttention's Triton kernels on this hardware, and rope_ops.py in this
package for the same note. Unlike rope_ops.rotary_embedding (in-place
mutation, so a bf16 workaround needs an explicit copy-back),
fused_add_rms_norm already returns brand-new tensors -- so on a Turing/
pre-Ampere GPU, bf16 inputs are simply upcast to fp32 for the kernel call
and the two outputs cast back to bf16, the same boundary-conversion
convention cmpext3's own CUDA kernels use throughout src/main.cpp. This
path IS autograd-safe (float()/`.to()` are ordinary differentiable ops,
unlike rope_ops's in-place `.copy_()` workaround), so no grad-liveness
guard is needed here.

HAS A REAL BACKWARD PASS (ported unchanged from upstream, itself adapted
from LinkedIn Liger Kernel's RMSNorm backward -- see the amd_tools file's
own docstring for the full derivation and provenance). Usable in a
training loop, unlike cmpext3's own CUDA kernels (forward-only, grad-gated
off in cmpext3/__init__.py).

UNVALIDATED ON A REAL CMP/TURING CARD: ported without hardware access to
test on. Before relying on this: compare its output (and, if training,
its gradients) numerically against `residual + x` followed by stock
F.rms_norm/manual RMSNorm + autograd, for your actual shapes/dtypes, and
benchmark against the unfused version.
"""
from __future__ import annotations

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


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
    def _fused_add_rmsnorm_fwd_kernel(
        x_ptr, residual_ptr, weight_ptr, out_ptr, new_residual_ptr, rstd_ptr,
        n_cols,
        x_row_stride, residual_row_stride, out_row_stride, new_residual_row_stride,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        # One program per row (i.e. per token): load that row's x and
        # residual, add in fp32 (more accurate than adding in fp16/bf16 and
        # then computing variance on the rounded result), write the new
        # residual back out in the working dtype, then normalize. rstd is
        # cached per row for backward.
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + row * residual_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        hidden = x + residual

        out_dtype = x_ptr.dtype.element_ty
        tl.store(new_residual_ptr + row * new_residual_row_stride + cols, hidden.to(out_dtype), mask=mask)

        variance = tl.sum(hidden * hidden, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(variance + eps)
        tl.store(rstd_ptr + row, rstd)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = hidden * rstd * weight

        tl.store(out_ptr + row * out_row_stride + cols, out.to(out_dtype), mask=mask)

    @triton.jit
    def _fused_add_rmsnorm_bwd_kernel(
        grad_out_ptr, grad_new_residual_ptr, new_residual_ptr, weight_ptr, rstd_ptr,
        grad_x_ptr, grad_residual_ptr, grad_weight_ptr,
        n_cols,
        grad_out_row_stride, grad_new_residual_row_stride, new_residual_row_stride,
        grad_x_row_stride, grad_residual_row_stride,
        HAS_RESIDUAL_GRAD: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        # dhidden = rstd*(dout*w) - (1/N)*rstd^3*sum((dout*w)*hidden)*hidden
        #         + dnew_residual   (identity gradient through hidden = x + residual)
        # dx = dresidual = dhidden  (both x and residual feed hidden identically)
        # dweight = sum over rows of (dout * hidden * rstd)
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        grad_out = tl.load(grad_out_ptr + row * grad_out_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        hidden = tl.load(new_residual_ptr + row * new_residual_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + row)

        m = grad_out * weight
        dot = tl.sum(m * hidden, axis=0)
        dhidden = rstd * m - (1.0 / n_cols) * rstd * rstd * rstd * dot * hidden

        if HAS_RESIDUAL_GRAD:
            grad_new_residual = tl.load(
                grad_new_residual_ptr + row * grad_new_residual_row_stride + cols, mask=mask, other=0.0
            ).to(tl.float32)
            dhidden += grad_new_residual

        out_dtype = grad_x_ptr.dtype.element_ty
        dhidden_out = dhidden.to(out_dtype)
        tl.store(grad_x_ptr + row * grad_x_row_stride + cols, dhidden_out, mask=mask)
        tl.store(grad_residual_ptr + row * grad_residual_row_stride + cols, dhidden_out, mask=mask)

        dweight = grad_out * hidden * rstd
        tl.atomic_add(grad_weight_ptr + cols, dweight, mask=mask)


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


_MAX_BLOCK_SIZE = 65536


def _num_warps(block_size: int) -> int:
    num_warps = 4
    if block_size >= 2048:
        num_warps = 8
    if block_size >= 8192:
        num_warps = 16
    return num_warps


if _TRITON_AVAILABLE:

    class _FusedAddRMSNormFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x2d, residual2d, weight, eps, block_size, n_rows):
            out = torch.empty_like(x2d)
            new_residual = torch.empty_like(x2d)
            rstd = torch.empty(n_rows, dtype=torch.float32, device=x2d.device)
            num_warps = _num_warps(block_size)

            _fused_add_rmsnorm_fwd_kernel[(n_rows,)](
                x2d, residual2d, weight, out, new_residual, rstd,
                x2d.shape[-1],
                x2d.stride(0), residual2d.stride(0), out.stride(0), new_residual.stride(0),
                eps,
                BLOCK_SIZE=block_size, num_warps=num_warps,
            )
            ctx.save_for_backward(new_residual, weight, rstd)
            ctx.block_size = block_size
            ctx.num_warps = num_warps
            return out, new_residual

        @staticmethod
        def backward(ctx, grad_out, grad_new_residual):
            new_residual, weight, rstd = ctx.saved_tensors
            n_rows, n_cols = new_residual.shape
            if grad_out is None:
                grad_out = torch.zeros_like(new_residual)

            grad_x = torch.empty_like(new_residual)
            grad_residual = torch.empty_like(new_residual)
            grad_weight_fp32 = torch.zeros(n_cols, dtype=torch.float32, device=weight.device)
            has_residual_grad = grad_new_residual is not None

            _fused_add_rmsnorm_bwd_kernel[(n_rows,)](
                grad_out, grad_new_residual, new_residual, weight, rstd,
                grad_x, grad_residual, grad_weight_fp32,
                n_cols,
                grad_out.stride(0),
                grad_new_residual.stride(0) if has_residual_grad else 0,
                new_residual.stride(0),
                grad_x.stride(0), grad_residual.stride(0),
                HAS_RESIDUAL_GRAD=has_residual_grad,
                BLOCK_SIZE=ctx.block_size, num_warps=ctx.num_warps,
            )
            grad_weight = grad_weight_fp32.to(weight.dtype)
            return grad_x, grad_residual, grad_weight, None, None, None


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """residual = residual + x; return (rms_norm(residual) * weight, residual)
    -- the returned second element is the new residual to carry into the
    next sublayer, matching the standard pre-norm decoder-layer pattern:

        attn_out = self_attn(x)
        x, residual = fused_add_rms_norm(attn_out, residual, ln1_weight, eps)
        ffn_out = ffn(x)
        x, residual = fused_add_rms_norm(ffn_out, residual, ln2_weight, eps)

    x/residual: any shape ending in hidden_dim (flattened internally to
    (rows, hidden_dim) and reshaped back). weight: (hidden_dim,).

    Has a real backward pass -- gradients flow correctly through x,
    residual, and weight, including when the returned `residual` is itself
    used downstream (e.g. by the next layer's own fused_add_rms_norm call).

    bf16 on a Turing/pre-Ampere GPU (compute capability < 8.0) is handled
    by running the kernel in fp32 and casting both outputs back to bf16 --
    see the TURING + BF16 note in the module docstring."""
    if not available():
        raise RuntimeError("fused_norm_ops.fused_add_rms_norm: triton is not importable")
    assert x.shape == residual.shape, "x and residual must have the same shape"
    assert x.shape[-1] == weight.shape[0], "last dim of x must match weight's length"
    assert x.dtype in _DTYPES, f"unsupported dtype {x.dtype}"
    assert residual.dtype == x.dtype, "x and residual must share a dtype"
    assert x.is_cuda and residual.is_cuda and weight.is_cuda, "fused_add_rms_norm requires CUDA/ROCm tensors"

    orig_dtype = x.dtype
    if orig_dtype == torch.bfloat16 and bf16_triton_unsupported(x.device):
        out, new_residual = fused_add_rms_norm(x.float(), residual.float(), weight.float(), eps)
        return out.to(torch.bfloat16), new_residual.to(torch.bfloat16)

    orig_shape = x.shape
    hidden_dim = orig_shape[-1]
    x2d = x.reshape(-1, hidden_dim).contiguous()
    residual2d = residual.reshape(-1, hidden_dim).contiguous()
    weight = weight.contiguous()
    n_rows = x2d.shape[0]

    block_size = _next_pow2(hidden_dim)
    if block_size > _MAX_BLOCK_SIZE:
        raise RuntimeError(
            f"fused_add_rms_norm: hidden_dim={hidden_dim} needs a {block_size}-wide "
            f"block, over this single-block implementation's {_MAX_BLOCK_SIZE} limit"
        )

    out, new_residual = _FusedAddRMSNormFunction.apply(x2d, residual2d, weight, eps, block_size, n_rows)
    return out.view(orig_shape), new_residual.view(orig_shape)
