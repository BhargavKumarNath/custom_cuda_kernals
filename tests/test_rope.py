"""Correctness tests for Kernel 3 (Fused RoPE) baselines. Mirrors
tests/test_rmsnorm_residual.py's structure.
"""

from __future__ import annotations

import pytest
import torch

from baselines.rope import (
    ALL_CASES,
    STANDARD_CASES,
    RopeCase,
    compiled_rope,
    eager_rope,
    make_inputs,
    reference_fp64,
)
from tests.numerics import DEFAULT_BF16_OUTLIER_FRACTION, assert_close_with_outliers

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA device required for RoPE baseline tests", allow_module_level=True)

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


def _case_id(case: RopeCase) -> str:
    return case.name


@pytest.mark.parametrize("case", ALL_CASES, ids=_case_id)
def test_eager_matches_fp64_reference(case: RopeCase):
    q, k, cos, sin = make_inputs(case)
    q_out, k_out = eager_rope(q, k, cos, sin)

    assert q_out.shape == case.q_shape
    assert k_out.shape == case.k_shape
    assert q_out.dtype == case.dtype

    if case.batch == 0:
        return

    q_ref, k_ref = reference_fp64(q, k, cos, sin)
    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(q_out, q_ref.to(case.dtype), max_outlier_fraction=outliers, **tol)
    assert_close_with_outliers(k_out, k_ref.to(case.dtype), max_outlier_fraction=outliers, **tol)


def test_eager_matches_reference_zero_batch_no_crash():
    case = RopeCase("explicit_empty", batch=0, seq_len=8, n_q_heads=2, n_kv_heads=2, head_dim=64, dtype=torch.float32)
    q, k, cos, sin = make_inputs(case)
    q_out, k_out = eager_rope(q, k, cos, sin)
    assert q_out.numel() == 0
    assert k_out.numel() == 0


def test_rotation_preserves_vector_norm():
    """Sanity check independent of the fp64 reference: RoPE is an
    orthogonal rotation, so it must preserve each (q/k) vector's L2 norm.
    """
    case = RopeCase("norm_check", batch=2, seq_len=16, n_q_heads=4, n_kv_heads=4, head_dim=64, dtype=torch.float32)
    q, k, cos, sin = make_inputs(case)
    q_out, k_out = eager_rope(q, k, cos, sin)

    torch.testing.assert_close(q_out.norm(dim=-1), q.norm(dim=-1), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(k_out.norm(dim=-1), k.norm(dim=-1), rtol=1e-4, atol=1e-4)


@pytest.mark.slow
@pytest.mark.parametrize("case", STANDARD_CASES, ids=_case_id)
def test_compiled_matches_eager(case: RopeCase):
    q, k, cos, sin = make_inputs(case)
    q_eager, k_eager = eager_rope(q, k, cos, sin)
    q_compiled, k_compiled = compiled_rope(q, k, cos, sin)

    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(q_compiled, q_eager, max_outlier_fraction=outliers, **tol)
    assert_close_with_outliers(k_compiled, k_eager, max_outlier_fraction=outliers, **tol)
