"""Python entrypoint for Kernel 10 (Spatiotemporal Graph Message Passing).

Converts each COO `(src, dst, weight)` edge set into CSR indexed by
destination node (sorting ~E edges is cheap bookkeeping relative to the
O(E * feature_dim) aggregation itself — same delegation pattern as
Kernel 4/7/8/9) and hands the result to
`custom_cuda._native.graph_message_passing_fwd`. See
`baselines/graph_message_passing.py::eager_spatiotemporal_mp` for the
reference semantics this must match.
"""

from __future__ import annotations

import torch

from custom_cuda import _native

__all__ = ["spatiotemporal_message_passing"]


def _to_csr(
    src: torch.Tensor, dst: torch.Tensor, weight: torch.Tensor, num_nodes: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort `(src, dst, weight)` COO edges by `dst` and build CSR
    `indptr`/`col`/`weight` arrays indexed by destination node.
    """
    device = weight.device
    if dst.numel() == 0:
        indptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
        return indptr, src, weight

    order = torch.argsort(dst)
    col = src.index_select(0, order).contiguous()
    weight_sorted = weight.index_select(0, order).contiguous()
    counts = torch.bincount(dst, minlength=num_nodes)
    indptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
    torch.cumsum(counts, dim=0, out=indptr[1:])
    return indptr, col, weight_sorted


def spatiotemporal_message_passing(
    x_curr: torch.Tensor,
    x_prev: torch.Tensor,
    spatial_src: torch.Tensor,
    spatial_dst: torch.Tensor,
    spatial_weight: torch.Tensor,
    temporal_src: torch.Tensor,
    temporal_dst: torch.Tensor,
    temporal_weight: torch.Tensor,
) -> torch.Tensor:
    """`out[dst] = sum_(src,dst) in spatial: spatial_weight * x_curr[src]
    + sum_(src,dst) in temporal: temporal_weight * x_prev[src]`.

    `x_curr`/`x_prev`: `[N, D]`, contiguous CUDA tensors sharing one
    dtype. Edge tensors are COO `(src, dst, weight)` triples, `src`/`dst`
    int64 in `[0, N)`, `weight` float32. Returns `out: [N, D]`, same
    dtype as `x_curr`.
    """
    num_nodes, _ = x_curr.shape

    spatial_indptr, spatial_col, spatial_weight_csr = _to_csr(spatial_src, spatial_dst, spatial_weight, num_nodes)
    temporal_indptr, temporal_col, temporal_weight_csr = _to_csr(
        temporal_src, temporal_dst, temporal_weight, num_nodes
    )

    out = torch.empty_like(x_curr)
    _native.graph_message_passing_fwd(
        x_curr, x_prev,
        spatial_indptr, spatial_col, spatial_weight_csr,
        temporal_indptr, temporal_col, temporal_weight_csr,
        out,
    )
    return out
