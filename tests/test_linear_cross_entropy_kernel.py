"""Correctness tests for Kernel 4's CUDA implementation
(csrc/kernels/linear_cross_entropy.cu, via
custom_cuda.kernels.linear_cross_entropy) against the PyTorch eager
baseline. Mirrors tests/test_rmsnorm_residual_kernel.py.
"""

from __future__ import annotations

import pytest
import torch

from baselines.linear_cross_entropy import ALL_CASES, LinearCECase, eager_linear_cross_entropy, make_inputs
from tests.numerics import DEFAULT_BF16_OUTLIER_FRACTION, assert_close_with_outliers

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA device required for Linear CE kernel tests", allow_module_level=True)

pytest.importorskip(
    "custom_cuda._native", reason="Rust extension not built — run `maturin develop` first"
)
from custom_cuda.kernels.linear_cross_entropy import linear_cross_entropy  # noqa: E402

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
def test_kernel_matches_eager(case: LinearCECase):
    hidden, weight, targets = make_inputs(case)
    loss_eager = eager_linear_cross_entropy(hidden, weight, targets, reduction=case.reduction)
    loss_kernel = linear_cross_entropy(hidden, weight, targets, reduction=case.reduction)

    if case.reduction == "none":
        assert loss_kernel.shape == (case.n_tokens,)
    else:
        assert loss_kernel.shape == ()

    # The kernel always accumulates and returns the loss in float32 (same
    # fp32-reduction convention used throughout this project — see
    # baselines/rmsnorm_residual.py etc.), regardless of hidden/weight's
    # dtype, whereas F.cross_entropy preserves the logits' input dtype.
    # float32 is the more correct choice for a loss value (training loops
    # generally want it at full precision for the optimizer/logging), so
    # this is a deliberate API difference, not a bug — compare against the
    # eager reference upcast to fp32.
    assert loss_kernel.dtype == torch.float32
    loss_eager = loss_eager.to(torch.float32)

    if case.n_tokens == 0:
        return

    equal_nan = case.name.startswith("all_ignored") and case.reduction == "mean"
    if equal_nan:
        assert torch.isnan(loss_kernel) and torch.isnan(loss_eager)
        return

    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(loss_kernel, loss_eager, max_outlier_fraction=outliers, **tol)


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 500, 1000, 10_000])
def test_kernel_matches_eager_across_chunk_sizes(chunk_size: int):
    """The chunking loop (custom_cuda/kernels/linear_cross_entropy.py) must
    produce identical results regardless of chunk_size — including sizes
    that don't evenly divide vocab_size, and sizes both much smaller and
    much larger than vocab_size (a single "chunk" covering everything).
    """
    case = LinearCECase("chunk_size_sweep", n_tokens=64, hidden_dim=64, vocab_size=1000, dtype=torch.float32)
    hidden, weight, targets = make_inputs(case)
    loss_eager = eager_linear_cross_entropy(hidden, weight, targets)
    loss_kernel = linear_cross_entropy(hidden, weight, targets, chunk_size=chunk_size)
    torch.testing.assert_close(loss_kernel, loss_eager, rtol=1e-4, atol=1e-4)


def test_kernel_rejects_cpu_tensor():
    hidden = torch.randn(4, 8)
    weight = torch.randn(10, 8)
    targets = torch.randint(0, 10, (4,), dtype=torch.long)
    with pytest.raises(ValueError):
        linear_cross_entropy(hidden, weight, targets)


def test_kernel_loss_has_no_grad_fn():
    """Documents a real limitation (see custom_cuda/kernels/
    linear_cross_entropy.py's docstring): `_native.linear_ce_chunk_update`
    is a raw PyO3 call, invisible to autograd, so the returned loss carries
    no gradient history back to hidden/weight — `.backward()` is not
    usable on it. This kernel is forward-only (inference/eval), not a
    training-loop drop-in.
    """
    hidden = torch.randn(8, 16, device="cuda", requires_grad=True)
    weight = torch.randn(20, 16, device="cuda", requires_grad=True)
    targets = torch.randint(0, 20, (8,), device="cuda", dtype=torch.long)

    loss = linear_cross_entropy(hidden, weight, targets)

    assert loss.grad_fn is None
    assert not loss.requires_grad


def test_kernel_rejects_invalid_reduction():
    hidden = torch.randn(4, 8, device="cuda")
    weight = torch.randn(10, 8, device="cuda")
    targets = torch.randint(0, 10, (4,), device="cuda", dtype=torch.long)
    with pytest.raises(ValueError):
        linear_cross_entropy(hidden, weight, targets, reduction="bogus")
