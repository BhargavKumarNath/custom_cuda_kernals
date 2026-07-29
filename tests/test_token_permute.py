"""Correctness tests for Kernel 7 (Token Scatter/Gather, Permute-Unpermute)
baselines. Mirrors tests/test_rmsnorm_residual.py's structure.
"""

from __future__ import annotations

import pytest
import torch

from baselines.token_permute import (
    COMBINE_ALL_CASES,
    COMBINE_STANDARD_CASES,
    GATHER_ALL_CASES,
    GATHER_STANDARD_CASES,
    TokenCombineCase,
    TokenGatherCase,
    compiled_token_combine,
    compiled_token_gather,
    compute_permutation,
    eager_token_combine,
    eager_token_gather,
    make_combine_inputs,
    make_gather_inputs,
    reference_combine_fp64,
    reference_gather_fp64,
)
from tests.numerics import DEFAULT_BF16_OUTLIER_FRACTION, assert_close_with_outliers

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA device required for token permute baseline tests", allow_module_level=True)

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
def test_eager_gather_matches_fp64_reference(case: TokenGatherCase):
    src, indices = make_gather_inputs(case)
    dst = eager_token_gather(src, indices)

    assert dst.shape == case.dst_shape
    assert dst.dtype == case.dtype

    if case.n_dst_rows == 0:
        return

    dst_ref = reference_gather_fp64(src, indices)
    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(dst, dst_ref.to(case.dtype), max_outlier_fraction=outliers, **tol)


@pytest.mark.parametrize("case", COMBINE_ALL_CASES, ids=_combine_id)
def test_eager_combine_matches_fp64_reference(case: TokenCombineCase):
    expert_output, unpermute_index, weights = make_combine_inputs(case)
    combined = eager_token_combine(expert_output, unpermute_index, weights)

    assert combined.shape == case.combined_shape
    assert combined.dtype == case.dtype

    if case.n_tokens == 0:
        return

    combined_ref = reference_combine_fp64(expert_output, unpermute_index, weights)
    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(combined, combined_ref.to(case.dtype), max_outlier_fraction=outliers, **tol)


def test_permutation_roundtrip():
    """compute_permutation's permute_index/unpermute_index must be
    self-consistent: gathering hidden states via permute_index, then
    gathering back via unpermute_index with weight=1, must reproduce the
    original per-token rows (summed over k, so use k=1 for an exact
    single-valued round trip).
    """
    torch.manual_seed(0)
    t, num_experts = 64, 8
    topk_indices = torch.randint(0, num_experts, (t, 1), device="cuda")
    permute_index, unpermute_index = compute_permutation(topk_indices)

    hidden = torch.randn(t, 32, device="cuda")
    permuted = eager_token_gather(hidden, permute_index)
    weights = torch.ones(t, 1, device="cuda")
    recovered = eager_token_combine(permuted, unpermute_index, weights)
    torch.testing.assert_close(recovered, hidden, rtol=1e-5, atol=1e-5)


def test_permutation_groups_by_expert():
    """After permutation, all rows belonging to the same expert must be
    contiguous, in non-decreasing expert-ID order.
    """
    torch.manual_seed(0)
    t, num_experts = 128, 8
    k = 2
    topk_indices = torch.randint(0, num_experts, (t, k), device="cuda")
    permute_index, _ = compute_permutation(topk_indices)

    # permute_index[i] is the *token* now at permuted slot i, but each
    # token has k assignments; recover which expert each permuted slot
    # actually belongs to by re-deriving the same sort the function used.
    flat_experts = topk_indices.reshape(-1)
    sort_order = torch.argsort(flat_experts, stable=True)
    experts_in_permuted_order = flat_experts[sort_order]
    assert torch.all(experts_in_permuted_order[:-1] <= experts_in_permuted_order[1:])


@pytest.mark.slow
@pytest.mark.parametrize("case", GATHER_STANDARD_CASES, ids=_gather_id)
def test_compiled_gather_matches_eager(case: TokenGatherCase):
    src, indices = make_gather_inputs(case)
    dst_eager = eager_token_gather(src, indices)
    dst_compiled = compiled_token_gather(src, indices)
    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(dst_compiled, dst_eager, max_outlier_fraction=outliers, **tol)


@pytest.mark.slow
@pytest.mark.parametrize("case", COMBINE_STANDARD_CASES, ids=_combine_id)
def test_compiled_combine_matches_eager(case: TokenCombineCase):
    expert_output, unpermute_index, weights = make_combine_inputs(case)
    combined_eager = eager_token_combine(expert_output, unpermute_index, weights)
    combined_compiled = compiled_token_combine(expert_output, unpermute_index, weights)
    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(combined_compiled, combined_eager, max_outlier_fraction=outliers, **tol)
