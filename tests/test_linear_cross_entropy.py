"""Correctness tests for Kernel 4 (Fused Linear Cross Entropy Loss)
baselines. Mirrors tests/test_rmsnorm_residual.py's structure.

Note: PyTorch's `F.cross_entropy(reduction="mean")` returns NaN when every
target is `ignore_index` (0/0) — see `all_ignored` case below, which
therefore needs `equal_nan=True`.
"""

from __future__ import annotations

import pytest
import torch

from baselines.linear_cross_entropy import (
    ALL_CASES,
    STANDARD_CASES,
    LinearCECase,
    compiled_linear_cross_entropy,
    eager_linear_cross_entropy,
    make_inputs,
    reference_fp64,
)
from tests.numerics import DEFAULT_BF16_OUTLIER_FRACTION, assert_close_with_outliers

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA device required for Linear CE baseline tests", allow_module_level=True)

# Cross-entropy loss is a scalar (or small per-token vector) derived from a
# full [N, V] reduction — looser than the elementwise Section 4.1 table by
# design, since float32 accumulation over thousands of vocab entries still
# accumulates more rounding than a single elementwise op. Kept as one
# consistent, slightly wider table across dtypes for this kernel.
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


def _case_id(case: LinearCECase) -> str:
    return case.name


@pytest.mark.parametrize("case", ALL_CASES, ids=_case_id)
def test_eager_matches_fp64_reference(case: LinearCECase):
    hidden, weight, targets = make_inputs(case)

    if case.n_tokens == 0:
        loss = eager_linear_cross_entropy(hidden, weight, targets, reduction=case.reduction)
        if case.reduction == "none":
            assert loss.numel() == 0
        return

    loss = eager_linear_cross_entropy(hidden, weight, targets, reduction=case.reduction)
    loss_ref = reference_fp64(hidden, weight, targets, reduction=case.reduction)

    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    equal_nan = case.name.startswith("all_ignored")
    if equal_nan:
        assert torch.isnan(loss) and torch.isnan(loss_ref)
        return
    assert_close_with_outliers(
        loss, loss_ref.to(case.dtype), max_outlier_fraction=outliers, **tol
    )


def test_eager_all_ignored_mean_is_nan():
    """Documents the PyTorch behavior our kernel wrapper must replicate:
    reduction="mean" with every token ignored is 0/0 -> NaN, not 0.
    """
    case = LinearCECase("explicit_all_ignored", n_tokens=32, hidden_dim=64, vocab_size=100, dtype=torch.float32, ignore_fraction=1.0)
    hidden, weight, targets = make_inputs(case)
    assert torch.all(targets == -100)
    loss = eager_linear_cross_entropy(hidden, weight, targets, reduction="mean")
    assert torch.isnan(loss)


@pytest.mark.slow
@pytest.mark.parametrize("case", STANDARD_CASES, ids=_case_id)
def test_compiled_matches_eager(case: LinearCECase):
    hidden, weight, targets = make_inputs(case)
    loss_eager = eager_linear_cross_entropy(hidden, weight, targets, reduction=case.reduction)
    loss_compiled = compiled_linear_cross_entropy(hidden, weight, targets, reduction=case.reduction)

    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(loss_compiled, loss_eager, max_outlier_fraction=outliers, **tol)
