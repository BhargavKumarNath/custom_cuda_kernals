"""Baseline references for Fused Cosine Similarity and Top-K Selection
(Kernel 8).

Semantics every implementation must reproduce exactly (matching
`torch.nn.functional.cosine_similarity`'s clamp convention, not an
additive epsilon):

    cos_sim[q, n] = dot(query_q, candidate_n) / max(||query_q|| * ||candidate_n||, eps)
    topk_scores, topk_indices = topk(cos_sim, k, dim=-1)

`queries: [Q, D]`, `candidates: [N, D]`. The eager reference materializes
the full `[Q, N]` similarity matrix (that's the memory cost the fused
kernel exists to avoid — see project_plan.md Section 3.8); dot
products/norms are computed in fp32 regardless of storage dtype (same
convention as every other kernel here), and `topk_scores` is returned as
fp32 regardless of input dtype.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import torch
import torch._dynamo

torch._dynamo.config.recompile_limit = 64

__all__ = [
    "CosineTopKCase",
    "STANDARD_CASES",
    "EDGE_CASES",
    "ALL_CASES",
    "make_inputs",
    "eager_cosine_topk",
    "compiled_cosine_topk",
    "reference_fp64",
]

DEFAULT_EPS = 1e-8


@dataclasses.dataclass(frozen=True)
class CosineTopKCase:
    """One (shape, dtype) configuration shared by tests and benchmarks."""

    name: str
    n_queries: int
    n_candidates: int
    dim: int
    k: int
    dtype: torch.dtype

    @property
    def queries_shape(self) -> tuple[int, int]:
        return (self.n_queries, self.dim)

    @property
    def candidates_shape(self) -> tuple[int, int]:
        return (self.n_candidates, self.dim)


_DTYPES: tuple[torch.dtype, ...] = (torch.float32, torch.float16, torch.bfloat16)


def _cases_for(name: str, n_queries: int, n_candidates: int, dim: int, k: int) -> list[CosineTopKCase]:
    return [
        CosineTopKCase(f"{name}_{dt}".replace("torch.", ""), n_queries, n_candidates, dim, k, dt)
        for dt in _DTYPES
    ]


# Representative RAG dense-retrieval shapes: small batch of queries
# against a moderate-to-large candidate pool, common embedding dims.
STANDARD_CASES: list[CosineTopKCase] = [
    *_cases_for("small_pool", n_queries=8, n_candidates=2000, dim=384, k=10),
    *_cases_for("large_pool", n_queries=8, n_candidates=50000, dim=768, k=10),
    # k sweep at a fixed moderate pool
    *_cases_for("ksweep_k1", n_queries=8, n_candidates=4000, dim=384, k=1),
    *_cases_for("ksweep_k2", n_queries=8, n_candidates=4000, dim=384, k=2),
    *_cases_for("ksweep_k4", n_queries=8, n_candidates=4000, dim=384, k=4),
    *_cases_for("ksweep_k8", n_queries=8, n_candidates=4000, dim=384, k=8),
    *_cases_for("ksweep_k16", n_queries=8, n_candidates=4000, dim=384, k=16),
    *_cases_for("ksweep_k32", n_queries=8, n_candidates=4000, dim=384, k=32),
]

# Section 4.3 edge-case battery, plus cases specific to this kernel:
# non-power-of-two dim, single query, single candidate, k == n_candidates,
# tiny candidate pool.
EDGE_CASES: list[CosineTopKCase] = [
    *_cases_for("npot_dim", n_queries=4, n_candidates=500, dim=100, k=8),
    *_cases_for("npot_candidates", n_queries=4, n_candidates=1001, dim=128, k=8),
    *_cases_for("single_query", n_queries=1, n_candidates=1000, dim=256, k=8),
    *_cases_for("single_candidate", n_queries=4, n_candidates=1, dim=256, k=1),
    *_cases_for("k_eq_candidates", n_queries=4, n_candidates=16, dim=128, k=16),
    *_cases_for("tiny_pool", n_queries=2, n_candidates=8, dim=64, k=4),
    *_cases_for("empty_queries", n_queries=0, n_candidates=1000, dim=128, k=8),
]

ALL_CASES: list[CosineTopKCase] = [*STANDARD_CASES, *EDGE_CASES]


def make_inputs(
    case: CosineTopKCase, device: str = "cuda", seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (queries, candidates) for a case."""
    gen = torch.Generator(device=device).manual_seed(seed)
    queries = torch.randn(case.queries_shape, dtype=case.dtype, device=device, generator=gen)
    candidates = torch.randn(case.candidates_shape, dtype=case.dtype, device=device, generator=gen)
    return queries, candidates


def eager_cosine_topk(
    queries: torch.Tensor, candidates: torch.Tensor, k: int, eps: float = DEFAULT_EPS
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch eager reference: materializes the full `[Q, N]` cosine
    similarity matrix (the memory cost the fused kernel avoids).
    """
    q = queries.to(torch.float32)
    c = candidates.to(torch.float32)
    scores = q @ c.T
    q_norm = q.norm(dim=-1, keepdim=True)
    c_norm = c.norm(dim=-1, keepdim=True)
    denom = torch.clamp(q_norm * c_norm.T, min=eps)
    cos_sim = scores / denom
    return torch.topk(cos_sim, k, dim=-1)


_compiled_cache: dict[str, Callable] = {}


def compiled_cosine_topk(
    queries: torch.Tensor, candidates: torch.Tensor, k: int, eps: float = DEFAULT_EPS
) -> tuple[torch.Tensor, torch.Tensor]:
    key = str(k)
    fn = _compiled_cache.get(key)
    if fn is None:
        def _fn(queries, candidates):
            return eager_cosine_topk(queries, candidates, k, eps)

        fn = torch.compile(_fn, mode="max-autotune", fullgraph=True)
        _compiled_cache[key] = fn
    return fn(queries, candidates)


def reference_fp64(
    queries: torch.Tensor, candidates: torch.Tensor, k: int, eps: float = DEFAULT_EPS
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp64 ground truth for correctness tests only."""
    q = queries.to(torch.float64)
    c = candidates.to(torch.float64)
    scores = q @ c.T
    q_norm = q.norm(dim=-1, keepdim=True)
    c_norm = c.norm(dim=-1, keepdim=True)
    denom = torch.clamp(q_norm * c_norm.T, min=eps)
    cos_sim = scores / denom
    return torch.topk(cos_sim, k, dim=-1)
