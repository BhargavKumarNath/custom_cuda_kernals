"""Correctness tests for Kernel 5's CUDA implementation
(csrc/kernels/matmul_add_bias.cu, via custom_cuda.kernels.matmul_add_bias)
against the PyTorch eager (unfused) baseline. Mirrors
tests/test_rmsnorm_residual_kernel.py.
"""

from __future__ import annotations

import pytest
import torch
from baselines.matmul_add_bias import ALL_CASES, MatmulBiasCase, eager_matmul_add_bias_unfused, make_inputs
from tests.numerics import DEFAULT_BF16_OUTLIER_FRACTION, assert_close_with_outliers
pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA device required for MatMul+Bias kernel tests", allow_module_level=True)

pytest.importorskip(
    "custom_cuda._native", reason="Rust extension not built — run `maturin develop` first"
)
from custom_cuda.kernels.matmul_add_bias import matmul_add_bias  # noqa: E402

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
def test_kernel_matches_eager(case: MatmulBiasCase):
    x, weight, bias = make_inputs(case)
    y_eager = eager_matmul_add_bias_unfused(x, weight, bias)
    y_kernel = matmul_add_bias(x, weight, bias)

    assert y_kernel.shape == case.y_shape
    assert y_kernel.dtype == case.dtype

    if case.m == 0:
        return

    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(y_kernel, y_eager, max_outlier_fraction=outliers, **tol)


def test_kernel_rejects_cpu_tensor():
    x = torch.randn(4, 8)
    weight = torch.randn(6, 8)
    with pytest.raises(ValueError):
        matmul_add_bias(x, weight)


def test_kernel_rejects_k_mismatch():
    x = torch.randn(4, 8, device="cuda")
    weight = torch.randn(6, 16, device="cuda")
    with pytest.raises(ValueError):
        matmul_add_bias(x, weight)


def test_kernel_rejects_dtype_mismatch():
    x = torch.randn(4, 8, device="cuda", dtype=torch.float32)
    weight = torch.randn(6, 8, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError):
        matmul_add_bias(x, weight)


def test_kernel_none_bias_matches_zero_bias():
    case = MatmulBiasCase("none_vs_zero", m=32, k=16, n=24, dtype=torch.float32, has_bias=False)
    x, weight, _ = make_inputs(case)
    y_no_bias = matmul_add_bias(x, weight, None)
    zero_bias = torch.zeros(24, device="cuda", dtype=torch.float32)
    y_zero_bias = matmul_add_bias(x, weight, zero_bias)
    torch.testing.assert_close(y_no_bias, y_zero_bias, rtol=1e-6, atol=1e-6)
