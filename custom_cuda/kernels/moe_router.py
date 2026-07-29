"""Python entrypoint for Kernel 6 (MoE Top-K Router).

Thin wrapper over `custom_cuda._native.moe_router_fwd`. See
`baselines/moe_router.py::eager_moe_router` for the reference semantics
this must match (softmax gating only — see that module's docstring for
the documented scope).
"""

from __future__ import annotations

import torch
from custom_cuda import _native

__all__ = ["moe_router"]


def moe_router(
    logits: torch.Tensor, k: int, renormalize: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """`topk_weights, topk_indices = topk(softmax(logits, dim=-1), k)`,
    optionally renormalized to sum to 1.

    logits: `[T, E]` contiguous CUDA tensor, `E <= 256`. Returns
    `topk_weights: [T, k]` (float32) and `topk_indices: [T, k]` (int64).
    """
    n_tokens = logits.shape[0]
    topk_weights = torch.empty((n_tokens, k), dtype=torch.float32, device=logits.device)
    topk_indices = torch.empty((n_tokens, k), dtype=torch.long, device=logits.device)
    _native.moe_router_fwd(logits, topk_weights, topk_indices, renormalize)
    return topk_weights, topk_indices
