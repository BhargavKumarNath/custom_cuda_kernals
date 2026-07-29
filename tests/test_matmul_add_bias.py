"""Correctness tests for Kernel 5 (Fused MatMul + Add Bias) baselines.
Mirrors tests/test_rmsnorm_residual.py's structure.
"""

from __future__ import annotations

import pytest
import torch

from baselines.matmul_add_bias import (
    ALL_CASES,
    STANDARD_CASES,
    MatmulBiasCase,
    compiled_matmul_add_bias,
    eager_matmul_add_bias_fused,
    eager_matmul_add_bias_unfused,
    make_inputs,
    reference_fp64,
)
from tests.numerics import DEFAULT_BF16_OUTLIER_FRACTION, assert_close_with_outliers

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA device required for MatMul+Bias baseline tests", allow_module_level=True)

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


def _case_id(case: MatmulBiasCase) -> str:
    return case.name


@pytest.mark.parametrize("case", ALL_CASES, ids=_case_id)
def test_unfused_matches_fp64_reference(case: MatmulBiasCase):
    x, weight, bias = make_inputs(case)
    y = eager_matmul_add_bias_unfused(x, weight, bias)

    assert y.shape == case.y_shape
    assert y.dtype == case.dtype

    if case.m == 0:
        return

    y_ref = reference_fp64(x, weight, bias)
    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(y, y_ref.to(case.dtype), max_outlier_fraction=outliers, **tol)


@pytest.mark.parametrize("case", ALL_CASES, ids=_case_id)
def test_fused_matches_unfused(case: MatmulBiasCase):
    """F.linear (cuBLAS-fused) and the naive two-op pattern must agree —
    both are eager references and must be mutually consistent before
    either is trusted (project_plan.md Section 2 Step 1).
    """
    x, weight, bias = make_inputs(case)
    y_unfused = eager_matmul_add_bias_unfused(x, weight, bias)
    y_fused = eager_matmul_add_bias_fused(x, weight, bias)

    if case.m == 0:
        assert y_fused.shape == y_unfused.shape
        return

    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(y_fused, y_unfused, max_outlier_fraction=outliers, **tol)


@pytest.mark.slow
@pytest.mark.parametrize("case", STANDARD_CASES, ids=_case_id)
def test_compiled_matches_eager(case: MatmulBiasCase):
    x, weight, bias = make_inputs(case)
    y_eager = eager_matmul_add_bias_unfused(x, weight, bias)
    y_compiled = compiled_matmul_add_bias(x, weight, bias)

    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(y_compiled, y_eager, max_outlier_fraction=outliers, **tol)
