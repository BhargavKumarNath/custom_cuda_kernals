"""Correctness tests for Kernel 8's CUDA implementation
(csrc/kernels/cosine_topk.cu, via custom_cuda.kernels.cosine_topk)
against the PyTorch eager baseline. Mirrors
tests/test_rmsnorm_residual_kernel.py.
"""

from __future__ import annotations

import pytest
import torch

from baselines.cosine_topk import ALL_CASES, CosineTopKCase, eager_cosine_topk, make_inputs
from tests.numerics import DEFAULT_BF16_OUTLIER_FRACTION, assert_close_with_outliers

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA device required for Cosine Top-K kernel tests", allow_module_level=True)

pytest.importorskip(
    "custom_cuda._native", reason="Rust extension not built — run `maturin develop` first"
)
from custom_cuda.kernels.cosine_topk import cosine_topk  # noqa: E402

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


def _case_id(case: CosineTopKCase) -> str:
    return case.name


def _assert_indices_equivalent(indices_a: torch.Tensor, indices_b: torch.Tensor) -> None:
    sorted_a, _ = torch.sort(indices_a, dim=-1)
    sorted_b, _ = torch.sort(indices_b, dim=-1)
    torch.testing.assert_close(sorted_a, sorted_b)


@pytest.mark.parametrize("case", ALL_CASES, ids=_case_id)
def test_kernel_matches_eager(case: CosineTopKCase):
    queries, candidates = make_inputs(case)
    scores_eager, indices_eager = eager_cosine_topk(queries, candidates, case.k)
    scores_kernel, indices_kernel = cosine_topk(queries, candidates, case.k)

    assert scores_kernel.shape == (case.n_queries, case.k)
    assert indices_kernel.shape == (case.n_queries, case.k)
    assert scores_kernel.dtype == torch.float32
    assert indices_kernel.dtype == torch.int64

    if case.n_queries == 0:
        return

    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(scores_kernel, scores_eager, max_outlier_fraction=outliers, **tol)
    _assert_indices_equivalent(indices_kernel, indices_eager)


def test_kernel_rejects_cpu_tensor():
    queries = torch.randn(4, 32)
    candidates = torch.randn(100, 32)
    with pytest.raises(ValueError):
        cosine_topk(queries, candidates, k=4)


def test_kernel_rejects_k_greater_than_candidates():
    queries = torch.randn(4, 32, device="cuda")
    candidates = torch.randn(8, 32, device="cuda")
    with pytest.raises(RuntimeError):
        cosine_topk(queries, candidates, k=16)


def test_kernel_rejects_k_greater_than_32():
    queries = torch.randn(4, 32, device="cuda")
    candidates = torch.randn(100, 32, device="cuda")
    with pytest.raises(RuntimeError):
        cosine_topk(queries, candidates, k=33)


def test_kernel_rejects_dim_mismatch():
    queries = torch.randn(4, 32, device="cuda")
    candidates = torch.randn(100, 64, device="cuda")
    with pytest.raises(ValueError):
        cosine_topk(queries, candidates, k=4)


def test_kernel_scores_within_valid_cosine_range():
    queries = torch.randn(16, 128, device="cuda")
    candidates = torch.randn(500, 128, device="cuda")
    scores, _ = cosine_topk(queries, candidates, k=8)
    assert torch.all(scores <= 1.0 + 1e-4)
    assert torch.all(scores >= -1.0 - 1e-4)
