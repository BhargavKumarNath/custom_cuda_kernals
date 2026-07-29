"""Correctness tests for Kernel 7's CUDA implementation
(csrc/kernels/token_permute.cu, via custom_cuda.kernels.token_permute)
against the PyTorch eager baseline. Mirrors
tests/test_rmsnorm_residual_kernel.py.
"""

from __future__ import annotations

import pytest
import torch

from baselines.token_permute import (
    COMBINE_ALL_CASES,
    GATHER_ALL_CASES,
    TokenCombineCase,
    TokenGatherCase,
    compute_permutation,
    eager_token_combine,
    eager_token_gather,
    make_combine_inputs,
    make_gather_inputs,
)
from tests.numerics import DEFAULT_BF16_OUTLIER_FRACTION, assert_close_with_outliers

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA device required for token permute kernel tests", allow_module_level=True)

pytest.importorskip(
    "custom_cuda._native", reason="Rust extension not built — run `maturin develop` first"
)
from custom_cuda.kernels.token_permute import token_combine, token_gather  # noqa: E402

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


def _gather_id(case: TokenGatherCase) -> str:
    return case.name


def _combine_id(case: TokenCombineCase) -> str:
    return case.name


@pytest.mark.parametrize("case", GATHER_ALL_CASES, ids=_gather_id)
def test_gather_kernel_matches_eager(case: TokenGatherCase):
    src, indices = make_gather_inputs(case)
    dst_eager = eager_token_gather(src, indices)
    dst_kernel = token_gather(src, indices)

    assert dst_kernel.shape == case.dst_shape
    assert dst_kernel.dtype == case.dtype

    if case.n_dst_rows == 0:
        return

    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(dst_kernel, dst_eager, max_outlier_fraction=outliers, **tol)


@pytest.mark.parametrize("case", COMBINE_ALL_CASES, ids=_combine_id)
def test_combine_kernel_matches_eager(case: TokenCombineCase):
    expert_output, unpermute_index, weights = make_combine_inputs(case)
    combined_eager = eager_token_combine(expert_output, unpermute_index, weights)
    combined_kernel = token_combine(expert_output, unpermute_index, weights)

    assert combined_kernel.shape == case.combined_shape
    assert combined_kernel.dtype == case.dtype

    if case.n_tokens == 0:
        return

    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(combined_kernel, combined_eager, max_outlier_fraction=outliers, **tol)


def test_kernel_permutation_roundtrip():
    """End-to-end integration: gather via the kernel using
    compute_permutation's permute_index, then combine back via the kernel
    with weight=1 at k=1, must exactly recover the original rows.
    """
    torch.manual_seed(0)
    t, num_experts = 128, 8
    topk_indices = torch.randint(0, num_experts, (t, 1), device="cuda")
    permute_index, unpermute_index = compute_permutation(topk_indices)

    hidden = torch.randn(t, 64, device="cuda", dtype=torch.float32)
    permuted = token_gather(hidden, permute_index)
    weights = torch.ones(t, 1, device="cuda", dtype=torch.float32)
    recovered = token_combine(permuted, unpermute_index, weights)

    torch.testing.assert_close(recovered, hidden, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16], ids=lambda d: str(d).split(".")[-1]
)
@pytest.mark.parametrize("hidden_dim", [1, 3, 5, 7, 9, 15, 17, 31, 33])
def test_gather_vectorization_tail_and_fallback_path(dtype: torch.dtype, hidden_dim: int):
    """Sweeps hidden_dim values whose row-byte-count isn't a multiple of
    16 for at least one dtype, forcing the scalar-fallback dispatch path
    (csrc/kernels/token_permute.cu) — none of baselines/token_permute.py's
    cases happen to hit every such remainder.
    """
    src = torch.randn(8, hidden_dim, device="cuda", dtype=dtype)
    indices = torch.randint(0, 8, (16,), device="cuda", dtype=torch.long)
    dst_kernel = token_gather(src, indices)
    dst_eager = eager_token_gather(src, indices)
    torch.testing.assert_close(dst_kernel, dst_eager, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16], ids=lambda d: str(d).split(".")[-1]
)
@pytest.mark.parametrize("hidden_dim", [1, 3, 5, 7, 9, 15, 17, 31, 33])
def test_combine_vectorization_tail_and_fallback_path(dtype: torch.dtype, hidden_dim: int):
    expert_output = torch.randn(24, hidden_dim, device="cuda", dtype=dtype)
    unpermute_index = torch.randperm(24, device="cuda")[:16].reshape(8, 2)
    weights = torch.rand(8, 2, device="cuda", dtype=torch.float32)
    combined_kernel = token_combine(expert_output, unpermute_index, weights)
    combined_eager = eager_token_combine(expert_output, unpermute_index, weights)

    tol = TOLERANCES[dtype]
    outliers = OUTLIER_FRACTION[dtype]
    assert_close_with_outliers(combined_kernel, combined_eager, max_outlier_fraction=outliers, **tol)


def test_gather_rejects_cpu_tensor():
    src = torch.randn(4, 8)
    indices = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError):
        token_gather(src, indices)


def test_combine_rejects_dtype_mismatch():
    expert_output = torch.randn(4, 8, device="cuda", dtype=torch.float32)
    unpermute_index = torch.zeros(2, 2, dtype=torch.long, device="cuda")
    weights = torch.ones(2, 2, device="cuda", dtype=torch.float16)  # must be float32
    combined = torch.empty(2, 8, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError):
        from custom_cuda import _native

        _native.token_combine_fwd(expert_output, unpermute_index, weights, combined)
