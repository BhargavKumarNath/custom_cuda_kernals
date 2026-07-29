"""Correctness tests for Kernel 2 (Fused SwiGLU Gated Activation) baselines.

Mirrors tests/test_rmsnorm_residual.py's structure — see that file for the
rationale behind the outlier-tolerant bf16 comparison and the
torch._dynamo.config.recompile_limit bump in baselines/swiglu.py.
"""

from __future__ import annotations

import pytest
import torch

from baselines.swiglu import (
    ALL_CASES,
    STANDARD_CASES,
    SwiGLUCase,
    compiled_swiglu,
    eager_swiglu,
    make_inputs,
    reference_fp64,
)
from tests.numerics import DEFAULT_BF16_OUTLIER_FRACTION, assert_close_with_outliers

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA device required for SwiGLU baseline tests", allow_module_level=True)

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
def test_eager_matches_fp64_reference(case: SwiGLUCase):
    gate, up = make_inputs(case)
    y = eager_swiglu(gate, up)

    assert y.shape == case.shape
    assert y.dtype == case.dtype

    if case.batch == 0:
        return

    y_ref = reference_fp64(gate, up)
    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(y, y_ref.to(case.dtype), max_outlier_fraction=outliers, **tol)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16], ids=lambda d: str(d).split(".")[-1]
)
def test_non_contiguous_inputs(dtype: torch.dtype):
    case = SwiGLUCase("non_contig", batch=2, seq_len=33, intermediate_dim=256, dtype=dtype)
    gate, up = make_inputs(case, contiguous=False)
    assert not gate.is_contiguous()
    assert not up.is_contiguous()

    y = eager_swiglu(gate, up)
    y_ref = reference_fp64(gate, up)

    tol = TOLERANCES[dtype]
    outliers = OUTLIER_FRACTION[dtype]
    assert_close_with_outliers(y, y_ref.to(dtype), max_outlier_fraction=outliers, **tol)


def test_eager_matches_reference_zero_batch_no_crash():
    case = SwiGLUCase("explicit_empty", batch=0, seq_len=8, intermediate_dim=64, dtype=torch.float32)
    gate, up = make_inputs(case)
    y = eager_swiglu(gate, up)
    assert y.numel() == 0


@pytest.mark.slow
@pytest.mark.parametrize("case", STANDARD_CASES, ids=_case_id)
def test_compiled_matches_eager(case: SwiGLUCase):
    gate, up = make_inputs(case)
    y_eager = eager_swiglu(gate, up)
    y_compiled = compiled_swiglu(gate, up)

    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(y_compiled, y_eager, max_outlier_fraction=outliers, **tol)
