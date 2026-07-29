"""Correctness tests for Kernel 2's CUDA implementation
(csrc/kernels/swiglu.cu, via custom_cuda.kernels.swiglu) against the
PyTorch eager baseline. Mirrors tests/test_rmsnorm_residual_kernel.py.
"""

from __future__ import annotations

import pytest
import torch

from baselines.swiglu import ALL_CASES, SwiGLUCase, eager_swiglu, make_inputs
from tests.numerics import DEFAULT_BF16_OUTLIER_FRACTION, assert_close_with_outliers

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA device required for SwiGLU kernel tests", allow_module_level=True)

pytest.importorskip(
    "custom_cuda._native", reason="Rust extension not built — run `maturin develop` first"
)
from custom_cuda.kernels.swiglu import swiglu  # noqa: E402

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


def _case_id(case: SwiGLUCase) -> str:
    return case.name


@pytest.mark.parametrize("case", ALL_CASES, ids=_case_id)
def test_kernel_matches_eager(case: SwiGLUCase):
    gate, up = make_inputs(case)
    y_eager = eager_swiglu(gate, up)
    y_kernel = swiglu(gate, up)

    assert y_kernel.shape == case.shape
    assert y_kernel.dtype == case.dtype

    if case.batch == 0:
        return

    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(y_kernel, y_eager, max_outlier_fraction=outliers, **tol)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16], ids=lambda d: str(d).split(".")[-1]
)
def test_kernel_rejects_non_contiguous(dtype: torch.dtype):
    case = SwiGLUCase("non_contig_reject", batch=2, seq_len=17, intermediate_dim=64, dtype=dtype)
    gate, up = make_inputs(case, contiguous=False)
    with pytest.raises(ValueError):
        swiglu(gate, up)


def test_kernel_rejects_cpu_tensor():
    gate = torch.randn(2, 4, 8)
    up = torch.randn(2, 4, 8)
    with pytest.raises(ValueError):
        swiglu(gate, up)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16], ids=lambda d: str(d).split(".")[-1]
)
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100, 101, 127, 128, 129])
def test_kernel_vectorization_tail_path(dtype: torch.dtype, n: int):
    """The kernel vectorizes with width 4 (fp32) / width 8 (fp16, bf16) and
    handles any remainder with a separate scalar-tail launch
    (csrc/kernels/swiglu.cu). None of the shape-based edge cases in
    baselines/swiglu.py happen to have a total element count that isn't
    exactly divisible by these widths, so this test exercises every
    remainder value directly to make sure the tail path itself is covered.
    """
    gate = torch.randn(n, device="cuda", dtype=dtype)
    up = torch.randn(n, device="cuda", dtype=dtype)
    y_kernel = swiglu(gate, up)
    y_eager = eager_swiglu(gate, up)

    tol = TOLERANCES[dtype]
    outliers = OUTLIER_FRACTION[dtype]
    assert_close_with_outliers(y_kernel, y_eager, max_outlier_fraction=outliers, **tol)


def test_kernel_rejects_dtype_mismatch():
    gate = torch.randn(2, 4, 8, device="cuda", dtype=torch.float32)
    up = torch.randn(2, 4, 8, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError):
        swiglu(gate, up)
