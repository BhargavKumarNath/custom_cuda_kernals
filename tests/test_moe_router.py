"""Correctness tests for Kernel 6 (MoE Top-K Router) baselines. Mirrors
tests/test_rmsnorm_residual.py's structure.
"""

from __future__ import annotations

import pytest
import torch

from baselines.moe_router import (
    ALL_CASES,
    STANDARD_CASES,
    MoERouterCase,
    compiled_moe_router,
    eager_moe_router,
    make_inputs,
    reference_fp64,
)
from tests.numerics import DEFAULT_BF16_OUTLIER_FRACTION, assert_close_with_outliers

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA device required for MoE router baseline tests", allow_module_level=True)

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


def _case_id(case: MoERouterCase) -> str:
    return case.name


@pytest.mark.parametrize("case", ALL_CASES, ids=_case_id)
def test_eager_matches_fp64_reference(case: MoERouterCase):
    logits = make_inputs(case)
    weights, indices = eager_moe_router(logits, case.k, case.renormalize)

    assert weights.shape == (case.n_tokens, case.k)
    assert indices.shape == (case.n_tokens, case.k)
    assert weights.dtype == torch.float32

    if case.n_tokens == 0:
        return

    weights_ref, indices_ref = reference_fp64(logits, case.k, case.renormalize)
    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(weights, weights_ref.to(torch.float32), max_outlier_fraction=outliers, **tol)
    torch.testing.assert_close(indices, indices_ref)


def test_eager_matches_reference_zero_batch_no_crash():
    case = MoERouterCase("explicit_empty", n_tokens=0, n_experts=8, k=2, dtype=torch.float32)
    logits = make_inputs(case)
    weights, indices = eager_moe_router(logits, case.k, case.renormalize)
    assert weights.numel() == 0
    assert indices.numel() == 0


def test_eager_topk_weights_sum_to_one_when_renormalized():
    case = MoERouterCase("renorm_check", n_tokens=32, n_experts=16, k=4, dtype=torch.float32)
    logits = make_inputs(case)
    weights, _ = eager_moe_router(logits, case.k, renormalize=True)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(case.n_tokens, device=weights.device), rtol=1e-5, atol=1e-5)


def test_eager_rejects_k_greater_than_experts():
    logits = torch.randn(4, 8, device="cuda")
    with pytest.raises(RuntimeError):
        eager_moe_router(logits, k=16)


@pytest.mark.slow
@pytest.mark.parametrize("case", STANDARD_CASES, ids=_case_id)
def test_compiled_matches_eager(case: MoERouterCase):
    logits = make_inputs(case)
    weights_eager, indices_eager = eager_moe_router(logits, case.k, case.renormalize)
    weights_compiled, indices_compiled = compiled_moe_router(logits, case.k, case.renormalize)

    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(weights_compiled, weights_eager, max_outlier_fraction=outliers, **tol)
    torch.testing.assert_close(indices_compiled, indices_eager)
