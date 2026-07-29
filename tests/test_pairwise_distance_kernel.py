"""Correctness tests for Kernel 9's CUDA implementation
(csrc/kernels/pairwise_distance.cu, via
custom_cuda.kernels.pairwise_distance) against the PyTorch eager baseline.
Mirrors tests/test_matmul_add_bias_kernel.py.
"""

from __future__ import annotations

import pytest
import torch
from baselines.pairwise_distance import ALL_CASES, PairwiseDistanceCase, eager_pairwise_distance_sq, make_inputs
from tests.numerics import DEFAULT_BF16_OUTLIER_FRACTION, assert_close_with_outliers

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA device required for Pairwise Distance kernel tests", allow_module_level=True)

pytest.importorskip(
    "custom_cuda._native", reason="Rust extension not built — run `maturin develop` first"
)
from custom_cuda.kernels.pairwise_distance import pairwise_distance_sq  # noqa: E402

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

# near_identical cases drive ||a||^2 + ||b||^2 - 2*dot to near-total
# cancellation (true distance ~0 from a sum of ~256-magnitude terms).
# There the kernel's tiled fp32 dot-product accumulation order and eager's
# cuBLAS matmul accumulation order round differently at the ULP level, and
# cancellation amplifies that gap — observed up to ~1.5e-4 on the diagonal
# even at fp32, which the strict fp32 atol (1e-4) doesn't budget for. This
# is the same cancellation the max(.,0) clamp guards against, just below
# the clamp's threshold; it's bounded well within the diag<1e-3 check in
# test_kernel_no_negative_distances_for_near_identical_vectors below, not a
# kernel correctness bug.
NEAR_IDENTICAL_OUTLIER_FRACTION = 1e-4


def _case_id(case: PairwiseDistanceCase) -> str:
    return case.name


@pytest.mark.parametrize("case", ALL_CASES, ids=_case_id)
def test_kernel_matches_eager(case: PairwiseDistanceCase):
    a, b = make_inputs(case)
    dist_sq_eager = eager_pairwise_distance_sq(a, b)
    dist_sq_kernel = pairwise_distance_sq(a, b)

    assert dist_sq_kernel.shape == (case.m, case.n)
    assert dist_sq_kernel.dtype == torch.float32

    if case.m == 0:
        return

    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    if case.near_identical:
        outliers = max(outliers, NEAR_IDENTICAL_OUTLIER_FRACTION)
    assert_close_with_outliers(dist_sq_kernel, dist_sq_eager, max_outlier_fraction=outliers, **tol)


def test_kernel_no_negative_distances_for_near_identical_vectors():
    case = PairwiseDistanceCase("clamp_check", m=256, n=256, dim=128, dtype=torch.float16, near_identical=True)
    a, b = make_inputs(case)
    dist_sq = pairwise_distance_sq(a, b)
    assert torch.all(dist_sq >= 0.0)

    diag = torch.diagonal(dist_sq)
    assert torch.all(diag < 1e-3)


def test_kernel_distance_to_self_is_zero():
    a = torch.randn(128, 64, device="cuda", dtype=torch.float32)
    dist_sq = pairwise_distance_sq(a, a)
    diag = torch.diagonal(dist_sq)
    torch.testing.assert_close(diag, torch.zeros_like(diag), rtol=0, atol=1e-3)


def test_kernel_rejects_cpu_tensor():
    a = torch.randn(4, 8)
    b = torch.randn(6, 8)
    with pytest.raises(ValueError):
        pairwise_distance_sq(a, b)


def test_kernel_rejects_dim_mismatch():
    a = torch.randn(4, 8, device="cuda")
    b = torch.randn(6, 16, device="cuda")
    with pytest.raises(ValueError):
        pairwise_distance_sq(a, b)


def test_kernel_rejects_dtype_mismatch():
    a = torch.randn(4, 8, device="cuda", dtype=torch.float32)
    b = torch.randn(6, 8, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError):
        pairwise_distance_sq(a, b)
