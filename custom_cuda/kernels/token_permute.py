"""Python entrypoint for Kernel 7 (Token Scatter/Gather, Permute-Unpermute).

Thin wrappers over `custom_cuda._native.token_gather_fwd` /
`token_combine_fwd`. See `baselines/token_permute.py` for the reference
semantics and the index-computation-vs-kernel architecture split
(`compute_permutation` there computes the index arrays these functions
consume).
"""

from __future__ import annotations

import torch
from custom_cuda import _native

__all__ = ["token_gather", "token_combine"]


def token_gather(src: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """`dst[i] = src[indices[i]]`. src: `[S, H]` contiguous CUDA tensor,
    indices: `[N]` int64. Returns `dst: [N, H]`.
    """
    n = indices.shape[0]
    h = src.shape[1]
    dst = torch.empty((n, h), dtype=src.dtype, device=src.device)
    _native.token_gather_fwd(src, indices, dst)
    return dst


def token_combine(
    expert_output: torch.Tensor, unpermute_index: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """`combined[t] = sum_j weights[t,j] * expert_output[unpermute_index[t,j]]`.

    expert_output: `[N, H]` contiguous CUDA tensor. unpermute_index:
    `[T, k]` int64. weights: `[T, k]` float32. Returns
    `combined: [T, H]`.
    """
    t, k = unpermute_index.shape
    h = expert_output.shape[1]
    combined = torch.empty((t, h), dtype=expert_output.dtype, device=expert_output.device)
    _native.token_combine_fwd(expert_output, unpermute_index, weights, combined)
    return combined
