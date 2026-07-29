"""Baseline references for Block Pairwise Distance Matrix Computation
(Kernel 9).

Scope: computes the **squared** Euclidean distance matrix (not the square
root) — many downstream uses (nearest-neighbor ranking, clustering
assignment) only need relative ordering, for which squared distance
suffices and avoids the extra `sqrt` cost per output element. Documented
here since project_plan.md Section 3.9's success criterion compares
against `torch.cdist` (which returns unsquared distance); this module
provides both `eager_pairwise_distance_sq` (our own formula-based
reference, matching exactly what the kernel computes) and
`cdist_distance_sq` (`torch.cdist(...) ** 2`, for a same-quantity
comparison against a different, independently-implemented PyTorch
primitive — used as both a cross-check and the benchmark's "beat cdist"
comparison point).

Semantics every implementation must reproduce exactly:

    dist_sq[i, j] = max(||A[i]||^2 + ||B[j]||^2 - 2 * dot(A[i], B[j]), 0)

`A: [M, Dim]`, `B: [N, Dim]`. The `max(..., 0)` clamp guards against
floating-point cancellation in the `||a||^2 + ||b||^2 - 2*a.b` expansion
producing small negative values for near-identical vectors (where the true
distance is ~0) — see the explicit `near_identical` edge case below, which
stress-tests exactly this. Computed in fp32 regardless of storage dtype
(same convention as every other kernel here).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import torch
import torch._dynamo

torch._dynamo.config.recompile_limit = 64

__all__ = [
    "PairwiseDistanceCase",
    "STANDARD_CASES",
    "EDGE_CASES",
    "ALL_CASES",
    "make_inputs",
    "eager_pairwise_distance_sq",
    "cdist_distance_sq",
    "compiled_pairwise_distance_sq",
    "reference_fp64",
]


@dataclasses.dataclass(frozen=True)
class PairwiseDistanceCase:
    """One (shape, dtype) configuration shared by tests and benchmarks."""

    name: str
    m: int
    n: int
    dim: int
    dtype: torch.dtype
    near_identical: bool = False

    @property
    def a_shape(self) -> tuple[int, int]:
        return (self.m, self.dim)

    @property
    def b_shape(self) -> tuple[int, int]:
        return (self.n, self.dim)


_DTYPES: tuple[torch.dtype, ...] = (torch.float32, torch.float16, torch.bfloat16)


def _cases_for(name: str, m: int, n: int, dim: int, **kwargs) -> list[PairwiseDistanceCase]:
    return [
        PairwiseDistanceCase(f"{name}_{dt}".replace("torch.", ""), m, n, dim, dt, **kwargs)
        for dt in _DTYPES
    ]


# Representative shapes: embedding clustering / dedup (moderate M, N,
# typical embedding dims), and a larger pool for the benchmark-style story.
STANDARD_CASES: list[PairwiseDistanceCase] = [
    *_cases_for("small", m=256, n=256, dim=128),
    *_cases_for("medium", m=1024, n=1024, dim=384),
    *_cases_for("large", m=4096, n=4096, dim=256),
]

# Section 4.3 edge-case battery, plus cases specific to this kernel:
# non-power-of-two M/N/dim, tall-skinny / short-wide, single row, and
# near-identical vectors (stress-tests the cancellation-guard clamp).
EDGE_CASES: list[PairwiseDistanceCase] = [
    *_cases_for("npot_dims", m=257, n=513, dim=100),
    *_cases_for("tall_skinny", m=4096, n=32, dim=64),
    *_cases_for("short_wide", m=32, n=4096, dim=64),
    *_cases_for("single_row", m=1, n=1, dim=128),
    *_cases_for("dim_eq_1", m=64, n=64, dim=1),
    *_cases_for("empty_m", m=0, n=64, dim=64),
    *_cases_for("near_identical", m=256, n=256, dim=128, near_identical=True),
]

ALL_CASES: list[PairwiseDistanceCase] = [*STANDARD_CASES, *EDGE_CASES]


def make_inputs(
    case: PairwiseDistanceCase, device: str = "cuda", seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (A, B) for a case. `near_identical=True` makes B a tiny
    perturbation of A, so the true squared distance is close to zero —
    exactly the regime where `||a||^2 + ||b||^2 - 2*a.b` is prone to
    floating-point cancellation producing small negative values.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn(case.a_shape, dtype=case.dtype, device=device, generator=gen)
    if case.near_identical:
        if case.m != case.n:
            raise ValueError("near_identical cases require m == n")
        noise = torch.randn(case.b_shape, dtype=torch.float32, device=device, generator=gen) * 1e-4
        b = (a.to(torch.float32) + noise).to(case.dtype)
    else:
        b = torch.randn(case.b_shape, dtype=case.dtype, device=device, generator=gen)
    return a, b


def eager_pairwise_distance_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch eager reference, matching the kernel's exact formula."""
    a32 = a.to(torch.float32)
    b32 = b.to(torch.float32)
    a_norm = (a32 * a32).sum(dim=-1, keepdim=True)
    b_norm = (b32 * b32).sum(dim=-1, keepdim=True)
    dot = a32 @ b32.T
    dist_sq = a_norm + b_norm.T - 2.0 * dot
    return torch.clamp(dist_sq, min=0.0)


def cdist_distance_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`torch.cdist(a, b) ** 2` — a different, independently-implemented
    PyTorch primitive, used for cross-checking and as the benchmark's
    "beat cdist" comparison point (Section 3.9's stated target).
    """
    return torch.cdist(a.to(torch.float32), b.to(torch.float32), p=2.0) ** 2


_compiled_cache: dict[str, Callable] = {}


def compiled_pairwise_distance_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    fn = _compiled_cache.get("fn")
    if fn is None:
        fn = torch.compile(eager_pairwise_distance_sq, mode="max-autotune", fullgraph=True)
        _compiled_cache["fn"] = fn
    return fn(a, b)


def reference_fp64(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """fp64 ground truth for correctness tests only."""
    a64 = a.to(torch.float64)
    b64 = b.to(torch.float64)
    a_norm = (a64 * a64).sum(dim=-1, keepdim=True)
    b_norm = (b64 * b64).sum(dim=-1, keepdim=True)
    dot = a64 @ b64.T
    dist_sq = a_norm + b_norm.T - 2.0 * dot
    return torch.clamp(dist_sq, min=0.0)
