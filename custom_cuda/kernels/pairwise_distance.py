"""Python entrypoint for Kernel 9 (Block Pairwise Distance Matrix).

Precomputes the squared row norms of `a` and `b` in PyTorch (cheap,
O(M*Dim + N*Dim) relative to the kernel's O(M*N*Dim) tiled term — see
csrc/includes/pairwise_distance.h's docstring for why this isn't fused
into the tiled loop) and hands them to
`custom_cuda._native.pairwise_distance_fwd` alongside the caller's `a`/`b`.
See `baselines/pairwise_distance.py::eager_pairwise_distance_sq` for the
reference semantics this must match.
"""

from __future__ import annotations

import torch

from custom_cuda import _native

__all__ = ["pairwise_distance_sq"]


def pairwise_distance_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`dist_sq[i, j] = max(||a[i]||^2 + ||b[j]||^2 - 2*dot(a[i], b[j]), 0)`.

    a: `[M, Dim]`, b: `[N, Dim]`, contiguous CUDA tensors sharing one
    dtype. Returns `dist_sq: [M, N]`, always float32.
    """
    m = a.shape[0]
    n = b.shape[0]
    device = a.device

    a_norm_sq = (a.float() ** 2).sum(dim=-1).contiguous()
    b_norm_sq = (b.float() ** 2).sum(dim=-1).contiguous()
    dist_sq = torch.empty((m, n), dtype=torch.float32, device=device)

    _native.pairwise_distance_fwd(a, b, a_norm_sq, b_norm_sq, dist_sq)
    return dist_sq
