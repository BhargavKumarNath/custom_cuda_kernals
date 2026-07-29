"""Python entrypoint for Kernel 5 (Fused MatMul + Add Bias).

Thin wrapper over `custom_cuda._native.matmul_add_bias_fwd`. See
`baselines/matmul_add_bias.py::eager_matmul_add_bias_unfused` for the
reference semantics this must match.
"""

from __future__ import annotations

import torch
from custom_cuda import _native

__all__ = ["matmul_add_bias"]


def matmul_add_bias(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """`y = x @ weight.T + bias`. x: `[M, K]`, weight: `[N, K]`, bias:
    `[N]` or `None`, contiguous CUDA tensors sharing one dtype.
    """
    m = x.shape[0]
    n = weight.shape[0]
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    _native.matmul_add_bias_fwd(x, weight, bias, y)
    return y
