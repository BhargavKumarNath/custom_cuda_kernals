"""Correctness tests for Kernel 9 (Block Pairwise Distance Matrix)
baselines. Mirrors tests/test_rmsnorm_residual.py's structure.
"""

from __future__ import annotations

import pytest
import torch

from baselines.pairwise_distance import (
    ALL_CASES,
    STANDARD_CASES,
    PairwiseDistanceCase,
    cdist_distance_sq,
    compiled_pairwise_distance_sq,
    eager_pairwise_distance_sq,
    make_inputs,
    reference_fp64,
)
from tests.numerics import DEFAULT_BF16_OUTLIER_FRACTION, assert_close_with_outliers

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA device required for Pairwise Distance baseline tests", allow_module_level=True)

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


def _case_id(case: PairwiseDistanceCase) -> str:
    return case.name


@pytest.mark.parametrize("case", ALL_CASES, ids=_case_id)
def test_eager_matches_fp64_reference(case: PairwiseDistanceCase):
    a, b = make_inputs(case)
    dist_sq = eager_pairwise_distance_sq(a, b)

    assert dist_sq.shape == (case.m, case.n)
    assert dist_sq.dtype == torch.float32

    if case.m == 0:
        return

    dist_sq_ref = reference_fp64(a, b)
    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(dist_sq, dist_sq_ref.to(torch.float32), max_outlier_fraction=outliers, **tol)


@pytest.mark.parametrize("case", STANDARD_CASES, ids=_case_id)
def test_eager_matches_cdist(case: PairwiseDistanceCase):
    """Cross-check against a different, independently-implemented PyTorch
    primitive (torch.cdist), not just our own formula.
    """
    a, b = make_inputs(case)
    dist_sq = eager_pairwise_distance_sq(a, b)
    dist_sq_cdist = cdist_distance_sq(a, b)

    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(dist_sq, dist_sq_cdist, max_outlier_fraction=outliers, **tol)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16], ids=lambda d: str(d).split(".")[-1]
)
def test_eager_no_negative_distances_for_near_identical_vectors(dtype: torch.dtype):
    """The exact scenario the max(..., 0) clamp exists for: when A and B
    are nearly identical, ||a||^2 + ||b||^2 - 2*a.b is prone to floating-
    point cancellation and can go slightly negative without the clamp.
    """
    case = PairwiseDistanceCase("clamp_check", m=256, n=256, dim=128, dtype=dtype, near_identical=True)
    a, b = make_inputs(case)
    dist_sq = eager_pairwise_distance_sq(a, b)
    assert torch.all(dist_sq >= 0.0)

    # Diagonal (A[i] vs its own near-identical perturbation) should be
    # very small, not just non-negative.
    diag = torch.diagonal(dist_sq)
    assert torch.all(diag < 1e-3)


def test_eager_matches_reference_zero_m_no_crash():
    case = PairwiseDistanceCase("explicit_empty", m=0, n=64, dim=64, dtype=torch.float32)
    a, b = make_inputs(case)
    dist_sq = eager_pairwise_distance_sq(a, b)
    assert dist_sq.numel() == 0


def test_eager_distance_to_self_is_zero():
    """dist_sq(a, a) must be exactly (up to fp roundoff) zero along the
    diagonal for fp32.
    """
    a = torch.randn(128, 64, device="cuda", dtype=torch.float32)
    dist_sq = eager_pairwise_distance_sq(a, a)
    diag = torch.diagonal(dist_sq)
    torch.testing.assert_close(diag, torch.zeros_like(diag), rtol=0, atol=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize("case", STANDARD_CASES, ids=_case_id)
def test_compiled_matches_eager(case: PairwiseDistanceCase):
    a, b = make_inputs(case)
    dist_sq_eager = eager_pairwise_distance_sq(a, b)
    dist_sq_compiled = compiled_pairwise_distance_sq(a, b)

    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(dist_sq_compiled, dist_sq_eager, max_outlier_fraction=outliers, **tol)
