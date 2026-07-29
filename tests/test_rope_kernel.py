"""Correctness tests for Kernel 3's CUDA implementation (csrc/kernels/rope.cu,
via custom_cuda.kernels.rope) against the PyTorch eager baseline. Mirrors
tests/test_rmsnorm_residual_kernel.py.
"""

from __future__ import annotations

import pytest
import torch

from baselines.rope import ALL_CASES, RopeCase, eager_rope, make_inputs
from tests.numerics import DEFAULT_BF16_OUTLIER_FRACTION, assert_close_with_outliers

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA device required for RoPE kernel tests", allow_module_level=True)

pytest.importorskip(
    "custom_cuda._native", reason="Rust extension not built — run `maturin develop` first"
)
from custom_cuda.kernels.rope import rope  # noqa: E402

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
def test_kernel_matches_eager(case: RopeCase):
    q, k, cos, sin = make_inputs(case)
    q_eager, k_eager = eager_rope(q, k, cos, sin)
    q_kernel, k_kernel = rope(q, k, cos, sin)

    assert q_kernel.shape == case.q_shape
    assert k_kernel.shape == case.k_shape
    assert q_kernel.dtype == case.dtype

    if case.batch == 0:
        return

    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(q_kernel, q_eager, max_outlier_fraction=outliers, **tol)
    assert_close_with_outliers(k_kernel, k_eager, max_outlier_fraction=outliers, **tol)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16], ids=lambda d: str(d).split(".")[-1]
)
@pytest.mark.parametrize("head_dim", [6, 10, 14])
def test_kernel_scalar_fallback_path(dtype: torch.dtype, head_dim: int):
    """The kernel vectorizes with width 4 (fp32) / width 8 (fp16, bf16)
    along the half-head-dim and falls back to a full scalar kernel when
    `half_dim` isn't divisible by that width (csrc/kernels/rope.cu). None
    of baselines/rope.py's cases exercise this — real head_dims are always
    powers of two — so this test forces it directly with head_dim values
    whose half (3, 5, 7) is divisible by neither 4 nor 8.
    """
    from baselines.rope import compute_cos_sin, eager_rope

    seq_len = 5
    q = torch.randn(2, seq_len, 3, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(2, seq_len, 1, head_dim, device="cuda", dtype=dtype)
    cos, sin = compute_cos_sin(seq_len, head_dim, device="cuda")

    q_kernel, k_kernel = rope(q, k, cos, sin)
    q_eager, k_eager = eager_rope(q, k, cos, sin)

    tol = TOLERANCES[dtype]
    outliers = OUTLIER_FRACTION[dtype]
    assert_close_with_outliers(q_kernel, q_eager, max_outlier_fraction=outliers, **tol)
    assert_close_with_outliers(k_kernel, k_eager, max_outlier_fraction=outliers, **tol)


def test_kernel_rejects_cpu_tensor():
    case = RopeCase("cpu_reject", batch=1, seq_len=4, n_q_heads=2, n_kv_heads=2, head_dim=16, dtype=torch.float32)
    q = torch.randn(case.q_shape)
    k = torch.randn(case.k_shape)
    cos = torch.randn(4, 8)
    sin = torch.randn(4, 8)
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    with pytest.raises(ValueError):
        from custom_cuda import _native

        _native.rope_fwd(q, k, cos, sin, q_out, k_out)


def test_kernel_rejects_dtype_mismatch_qk():
    q = torch.randn(1, 4, 2, 16, device="cuda", dtype=torch.float32)
    k = torch.randn(1, 4, 2, 16, device="cuda", dtype=torch.float16)
    cos = torch.randn(4, 8, device="cuda", dtype=torch.float32)
    sin = torch.randn(4, 8, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError):
        rope(q, k, cos, sin)


def test_kernel_rejects_non_float32_cos_sin():
    q = torch.randn(1, 4, 2, 16, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 4, 2, 16, device="cuda", dtype=torch.float16)
    cos = torch.randn(4, 8, device="cuda", dtype=torch.float16)
    sin = torch.randn(4, 8, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError):
        rope(q, k, cos, sin)


def test_kernel_rejects_odd_head_dim():
    q = torch.randn(1, 4, 2, 15, device="cuda", dtype=torch.float32)
    k = torch.randn(1, 4, 2, 15, device="cuda", dtype=torch.float32)
    cos = torch.randn(4, 7, device="cuda", dtype=torch.float32)
    sin = torch.randn(4, 7, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError):
        rope(q, k, cos, sin)
