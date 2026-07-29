"""Correctness tests for Kernel 8 (Fused Cosine Similarity + Top-K)
baselines. Mirrors tests/test_rmsnorm_residual.py's structure.

Compares `topk_indices` as an order-agnostic set per query rather than
requiring positional equality — Kernel 6 (MoE Router) hit exact
probability ties at fp16/bf16 precision that reorder same-valued entries
differently between implementations without changing which candidates are
selected; the same risk applies here given cosine similarity's bounded
[-1, 1] range and coarse fp16/bf16 quantization, so this test is written
defensively from the start rather than discovering the issue again.
"""

from __future__ import annotations

import pytest
import torch

from baselines.cosine_topk import (
    ALL_CASES,
    STANDARD_CASES,
    CosineTopKCase,
    compiled_cosine_topk,
    eager_cosine_topk,
    make_inputs,
    reference_fp64,
)
from tests.numerics import DEFAULT_BF16_OUTLIER_FRACTION, assert_close_with_outliers

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA device required for Cosine Top-K baseline tests", allow_module_level=True)

TOLERANCES: dict[torch.dtype, dict[str, float]] = {
    torch.float32: dict(rtol=1e-4, atol=1e-4),
    torch.float16: dict(rtol=1e-2, atol=1e-2),
    torch.bfloat16: dict(rtol=1e-2, atol=2e-2),
}
OUTLIER_FRACTION: dict[torch.dtype, float] = {
    torch.float32: 0.0,
    torch.float16: 0.0,
    torch.bfloat16: DEFAULT_BF16_OUTLIER_FRACTION,
}


def _case_id(case: CosineTopKCase) -> str:
    return case.name


def _assert_indices_equivalent(indices_a: torch.Tensor, indices_b: torch.Tensor) -> None:
    sorted_a, _ = torch.sort(indices_a, dim=-1)
    sorted_b, _ = torch.sort(indices_b, dim=-1)
    torch.testing.assert_close(sorted_a, sorted_b)


@pytest.mark.parametrize("case", ALL_CASES, ids=_case_id)
def test_eager_matches_fp64_reference(case: CosineTopKCase):
    queries, candidates = make_inputs(case)
    scores, indices = eager_cosine_topk(queries, candidates, case.k)

    assert scores.shape == (case.n_queries, case.k)
    assert indices.shape == (case.n_queries, case.k)
    assert scores.dtype == torch.float32

    if case.n_queries == 0:
        return

    scores_ref, indices_ref = reference_fp64(queries, candidates, case.k)
    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(scores, scores_ref.to(torch.float32), max_outlier_fraction=outliers, **tol)
    _assert_indices_equivalent(indices, indices_ref)


def test_eager_matches_reference_zero_queries_no_crash():
    case = CosineTopKCase("explicit_empty", n_queries=0, n_candidates=100, dim=32, k=4, dtype=torch.float32)
    queries, candidates = make_inputs(case)
    scores, indices = eager_cosine_topk(queries, candidates, case.k)
    assert scores.numel() == 0
    assert indices.numel() == 0


def test_eager_scores_within_valid_cosine_range():
    """Cosine similarity is mathematically bounded to [-1, 1] (up to
    floating-point slop); a bug that skips normalization would produce
    unbounded dot-product magnitudes instead.
    """
    case = CosineTopKCase("range_check", n_queries=16, n_candidates=500, dim=128, k=8, dtype=torch.float32)
    queries, candidates = make_inputs(case)
    scores, _ = eager_cosine_topk(queries, candidates, case.k)
    assert torch.all(scores <= 1.0 + 1e-4)
    assert torch.all(scores >= -1.0 - 1e-4)


def test_eager_rejects_k_greater_than_candidates():
    queries = torch.randn(4, 32, device="cuda")
    candidates = torch.randn(8, 32, device="cuda")
    with pytest.raises(RuntimeError):
        eager_cosine_topk(queries, candidates, k=16)


@pytest.mark.slow
@pytest.mark.parametrize("case", STANDARD_CASES, ids=_case_id)
def test_compiled_matches_eager(case: CosineTopKCase):
    queries, candidates = make_inputs(case)
    scores_eager, indices_eager = eager_cosine_topk(queries, candidates, case.k)
    scores_compiled, indices_compiled = compiled_cosine_topk(queries, candidates, case.k)

    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(scores_compiled, scores_eager, max_outlier_fraction=outliers, **tol)
    _assert_indices_equivalent(indices_compiled, indices_eager)
