"""Python entrypoint for Kernel 1 (Fused RMSNorm + Residual Addition).

Thin wrapper over `custom_cuda._native.rmsnorm_residual_fwd`: allocates the
output tensors and calls the fused CUDA kernel. See
`baselines/rmsnorm_residual.py::eager_rmsnorm_residual` for the reference
semantics this must match.
"""

from __future__ import annotations

import torch
from custom_cuda import _native

__all__ = ["rmsnorm_residual"]


def rmsnorm_residual(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """`y, residual_out = RMSNorm(x + residual) * weight, x + residual`.

    x, residual: `[..., hidden_dim]` contiguous CUDA tensors, same dtype.
    weight: `[hidden_dim]` contiguous CUDA tensor, same dtype as x.
    """
    y = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    _native.rmsnorm_residual_fwd(x, residual, weight, y, residual_out, eps)
    return y, residual_out
