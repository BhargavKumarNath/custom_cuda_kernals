"""Correctness tests for Kernel 1 (Fused RMSNorm + Residual Addition).

At this stage only the baselines in `baselines/rmsnorm_residual.py` exist —
these tests pin down that both reference implementations (eager,
`torch.compile`) are correct and mutually consistent *before* the CUDA
kernel is written against them. Once `custom_cuda.kernels.rmsnorm_residual`
exists, its output will be checked against `eager_rmsnorm_residual` using
the same tolerance table.

Tolerances and edge cases follow project_plan.md Section 4.
"""

from __future__ import annotations

import pytest
import torch

from baselines.rmsnorm_residual import (
    ALL_CASES,
    STANDARD_CASES,
    RMSNormResidualCase,
    compiled_rmsnorm_residual,
    eager_rmsnorm_residual,
    make_inputs,
    reference_fp64,
)
from tests.numerics import DEFAULT_BF16_OUTLIER_FRACTION, assert_close_with_outliers

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA device required for RMSNorm+residual baseline tests", allow_module_level=True)

# project_plan.md Section 4.1
TOLERANCES: dict[torch.dtype, dict[str, float]] = {
    torch.float32: dict(rtol=1e-4, atol=1e-4),
    torch.float16: dict(rtol=1e-2, atol=1e-3),
    torch.bfloat16: dict(rtol=1e-2, atol=1e-2),
}

# See tests/numerics.py: bf16's 8-bit precision produces rare, expected
# rounding-boundary mismatches at million-element scale. fp32/fp16 stay
# strict (0 outliers tolerated).
OUTLIER_FRACTION: dict[torch.dtype, float] = {
    torch.float32: 0.0,
    torch.float16: 0.0,
    torch.bfloat16: DEFAULT_BF16_OUTLIER_FRACTION,
}


def _case_id(case: RMSNormResidualCase) -> str:
    return case.name


@pytest.mark.parametrize("case", ALL_CASES, ids=_case_id)
def test_eager_matches_fp64_reference(case: RMSNormResidualCase):
    x, residual, weight = make_inputs(case)
    y, residual_out = eager_rmsnorm_residual(x, residual, weight, case.eps)

    assert y.shape == case.shape
    assert residual_out.shape == case.shape
    assert y.dtype == case.dtype
    assert residual_out.dtype == case.dtype

    if case.batch == 0:
        # Empty-batch: shape/dtype correctness is the whole test — there is
        # no numerical content to compare.
        return

    y_ref, residual_out_ref = reference_fp64(x, residual, weight, case.eps)
    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(y, y_ref.to(case.dtype), max_outlier_fraction=outliers, **tol)
    assert_close_with_outliers(
        residual_out, residual_out_ref.to(case.dtype), max_outlier_fraction=outliers, **tol
    )


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16], ids=lambda d: str(d).split(".")[-1]
)
def test_non_contiguous_inputs(dtype: torch.dtype):
    """Section 4.2: stride-2 (non-contiguous) last dimension must still
    produce correct results from the eager reference.
    """
    case = RMSNormResidualCase("non_contig", batch=2, seq_len=33, hidden_dim=256, dtype=dtype)
    x, residual, weight = make_inputs(case, contiguous=False)
    assert not x.is_contiguous()
    assert not residual.is_contiguous()

    y, residual_out = eager_rmsnorm_residual(x, residual, weight, case.eps)
    y_ref, residual_out_ref = reference_fp64(x, residual, weight, case.eps)

    tol = TOLERANCES[dtype]
    outliers = OUTLIER_FRACTION[dtype]
    assert_close_with_outliers(y, y_ref.to(dtype), max_outlier_fraction=outliers, **tol)
    assert_close_with_outliers(
        residual_out, residual_out_ref.to(dtype), max_outlier_fraction=outliers, **tol
    )


def test_eager_matches_reference_zero_batch_no_crash():
    """Explicit empty-input smoke test beyond the shape/dtype check above:
    must not launch a CUDA kernel over zero elements and must not raise.
    """
    case = RMSNormResidualCase("explicit_empty", batch=0, seq_len=8, hidden_dim=64, dtype=torch.float32)
    x, residual, weight = make_inputs(case)
    y, residual_out = eager_rmsnorm_residual(x, residual, weight, case.eps)
    assert y.numel() == 0
    assert residual_out.numel() == 0


@pytest.mark.slow
@pytest.mark.parametrize("case", STANDARD_CASES, ids=_case_id)
def test_compiled_matches_eager(case: RMSNormResidualCase):
    """Step 1 of the Section 2 lifecycle: eager and torch.compile references
    must agree before either is trusted as a benchmark baseline. Marked
    `slow` because `mode="max-autotune"` triggers a real autotuning search
    per unique shape on first call.
    """
    x, residual, weight = make_inputs(case)
    y_eager, residual_out_eager = eager_rmsnorm_residual(x, residual, weight, case.eps)
    y_compiled, residual_out_compiled = compiled_rmsnorm_residual(x, residual, weight, case.eps)

    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(y_compiled, y_eager, max_outlier_fraction=outliers, **tol)
    assert_close_with_outliers(
        residual_out_compiled, residual_out_eager, max_outlier_fraction=outliers, **tol
    )
