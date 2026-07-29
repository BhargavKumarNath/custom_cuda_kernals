"""Correctness tests for Kernel 1's CUDA implementation
(csrc/kernels/rmsnorm_residual.cu, via custom_cuda.kernels.rmsnorm_residual)
against the PyTorch eager baseline.

project_plan.md Section 2 Step 4: the kernel must match
`eager_rmsnorm_residual` within the Section 4.1 tolerance table across all
three dtypes and the full edge-case battery already exercised by
`baselines/rmsnorm_residual.py`.
"""

from __future__ import annotations

import pytest
import torch

from baselines.rmsnorm_residual import (
    ALL_CASES,
    RMSNormResidualCase,
    eager_rmsnorm_residual,
    make_inputs,
)
from tests.numerics import DEFAULT_BF16_OUTLIER_FRACTION, assert_close_with_outliers

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA device required for RMSNorm+residual kernel tests", allow_module_level=True)

pytest.importorskip(
    "custom_cuda._native", reason="Rust extension not built — run `maturin develop` first"
)
from custom_cuda.kernels.rmsnorm_residual import rmsnorm_residual  # noqa: E402

TOLERANCES: dict[torch.dtype, dict[str, float]] = {
    torch.float32: dict(rtol=1e-4, atol=1e-4),
    torch.float16: dict(rtol=1e-2, atol=1e-3),
    torch.bfloat16: dict(rtol=1e-2, atol=1e-2),
}
OUTLIER_FRACTION: dict[torch.dtype, float] = {
    torch.float32: 0.0,
    torch.float16: 0.0,
    torch.bfloat16: DEFAULT_BF16_OUTLIER_FRACTION,
}


def _case_id(case: RMSNormResidualCase) -> str:
    return case.name


@pytest.mark.parametrize("case", ALL_CASES, ids=_case_id)
def test_kernel_matches_eager(case: RMSNormResidualCase):
    x, residual, weight = make_inputs(case)
    y_eager, residual_out_eager = eager_rmsnorm_residual(x, residual, weight, case.eps)
    y_kernel, residual_out_kernel = rmsnorm_residual(x, residual, weight, case.eps)

    assert y_kernel.shape == case.shape
    assert residual_out_kernel.shape == case.shape
    assert y_kernel.dtype == case.dtype
    assert residual_out_kernel.dtype == case.dtype

    if case.batch == 0:
        return

    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(y_kernel, y_eager, max_outlier_fraction=outliers, **tol)
    assert_close_with_outliers(
        residual_out_kernel, residual_out_eager, max_outlier_fraction=outliers, **tol
    )


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16], ids=lambda d: str(d).split(".")[-1]
)
def test_kernel_rejects_non_contiguous(dtype: torch.dtype):
    """This first-pass kernel requires contiguous inputs (Section 4.2) and
    must raise a clear error rather than silently computing garbage.
    """
    case = RMSNormResidualCase("non_contig_reject", batch=2, seq_len=17, hidden_dim=64, dtype=dtype)
    x, residual, weight = make_inputs(case, contiguous=False)
    with pytest.raises(ValueError):
        rmsnorm_residual(x, residual, weight, case.eps)


def test_kernel_rejects_cpu_tensor():
    x = torch.randn(2, 4, 8)
    residual = torch.randn(2, 4, 8)
    weight = torch.randn(8)
    with pytest.raises(ValueError):
        rmsnorm_residual(x, residual, weight, 1e-6)


def test_kernel_rejects_dtype_mismatch():
    x = torch.randn(2, 4, 8, device="cuda", dtype=torch.float32)
    residual = torch.randn(2, 4, 8, device="cuda", dtype=torch.float16)
    weight = torch.randn(8, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError):
        rmsnorm_residual(x, residual, weight, 1e-6)
