"""Python entrypoint for Kernel 8 (Fused Cosine Similarity + Top-K).

Orchestrates the two-kernel partition + merge design (see
csrc/includes/cosine_topk.h's docstring for why a single warp-per-query
kernel catastrophically underutilizes the GPU when the query count is
small — the realistic RAG case): this module chooses `num_partitions`
from `num_candidates`/`num_queries`, allocates the intermediate
`[Q, num_partitions, k]` buffers, and calls
`custom_cuda._native.cosine_topk_partial_fwd` then `..._merge_fwd`. See
`baselines/cosine_topk.py::eager_cosine_topk` for the reference semantics
this must match.
"""

from __future__ import annotations

import torch

from custom_cuda import _native

__all__ = ["cosine_topk"]

DEFAULT_EPS = 1e-8

# Tuned so that num_queries * num_partitions comfortably exceeds this
# GPU's SM count (36 on the RTX 4070 Laptop) with several warps to spare
# per SM for latency hiding, while each partition still scans enough
# candidates to amortize its own launch/reduction overhead.
_TARGET_TOTAL_WARPS = 512
_MIN_CANDIDATES_PER_PARTITION = 256
_MAX_PARTITIONS = 128


def _choose_num_partitions(num_queries: int, num_candidates: int) -> int:
    if num_queries <= 0 or num_candidates <= 0:
        return 1
    max_by_candidates = max(1, num_candidates // _MIN_CANDIDATES_PER_PARTITION)
    target = max(1, _TARGET_TOTAL_WARPS // num_queries)
    return max(1, min(target, max_by_candidates, _MAX_PARTITIONS))


def cosine_topk(
    queries: torch.Tensor, candidates: torch.Tensor, k: int, eps: float = DEFAULT_EPS
) -> tuple[torch.Tensor, torch.Tensor]:
    """`topk_scores, topk_indices = topk(cosine_similarity(queries, candidates), k)`,
    without ever materializing the full `[Q, N]` similarity matrix.

    queries: `[Q, D]`, candidates: `[N, D]`, contiguous CUDA tensors
    sharing one dtype. `k` must be `<= 32` and `<= N`. Returns
    `topk_scores: [Q, k]` (float32) and `topk_indices: [Q, k]` (int64).
    """
    n_queries = queries.shape[0]
    n_candidates = candidates.shape[0]
    device = queries.device

    num_partitions = _choose_num_partitions(n_queries, n_candidates)

    partial_scores = torch.empty((n_queries, num_partitions, k), dtype=torch.float32, device=device)
    partial_indices = torch.empty((n_queries, num_partitions, k), dtype=torch.long, device=device)
    _native.cosine_topk_partial_fwd(queries, candidates, partial_scores, partial_indices, eps)

    topk_scores = torch.empty((n_queries, k), dtype=torch.float32, device=device)
    topk_indices = torch.empty((n_queries, k), dtype=torch.long, device=device)
    _native.cosine_topk_merge_fwd(partial_scores, partial_indices, topk_scores, topk_indices)

    return topk_scores, topk_indices
