"""Correctness tests for Kernel 6's CUDA implementation
(csrc/kernels/moe_router.cu, via custom_cuda.kernels.moe_router) against
the PyTorch eager baseline. Mirrors tests/test_rmsnorm_residual_kernel.py.
"""

from __future__ import annotations

import pytest
import torch

from baselines.moe_router import ALL_CASES, MoERouterCase, eager_moe_router, make_inputs
from tests.numerics import DEFAULT_BF16_OUTLIER_FRACTION, assert_close_with_outliers

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA device required for MoE router kernel tests", allow_module_level=True)

pytest.importorskip(
    "custom_cuda._native", reason="Rust extension not built — run `maturin develop` first"
)
from custom_cuda.kernels.moe_router import moe_router  # noqa: E402

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


def _case_id(case: MoERouterCase) -> str:
    return case.name


@pytest.mark.parametrize("case", ALL_CASES, ids=_case_id)
def test_kernel_matches_eager(case: MoERouterCase):
    logits = make_inputs(case)
    weights_eager, indices_eager = eager_moe_router(logits, case.k, case.renormalize)
    weights_kernel, indices_kernel = moe_router(logits, case.k, case.renormalize)

    assert weights_kernel.shape == (case.n_tokens, case.k)
    assert indices_kernel.shape == (case.n_tokens, case.k)
    assert weights_kernel.dtype == torch.float32
    assert indices_kernel.dtype == torch.int64

    if case.n_tokens == 0:
        return

    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(weights_kernel, weights_eager, max_outlier_fraction=outliers, **tol)

    # Compare as an order-agnostic *set* of selected experts per token,
    # not positional equality: low-precision (fp16/bf16) logits routinely
    # produce exact probability ties (many distinct fp32 logits collapse
    # to the same quantized value), and our kernel's warp-shuffle argmax
    # breaks ties differently from torch.topk. Verified empirically that
    # every observed mismatch is an exact tie (probability difference
    # 0.0) with the same set of experts merely reordered — not a real
    # selection bug. float32 essentially never hits this (no widespread
    # exact ties), so positional order matches there anyway.
    sorted_kernel, _ = torch.sort(indices_kernel, dim=-1)
    sorted_eager, _ = torch.sort(indices_eager, dim=-1)
    torch.testing.assert_close(sorted_kernel, sorted_eager)


def test_kernel_rejects_cpu_tensor():
    logits = torch.randn(4, 8)
    with pytest.raises(ValueError):
        moe_router(logits, k=2)


def test_kernel_rejects_k_greater_than_experts():
    logits = torch.randn(4, 8, device="cuda")
    with pytest.raises(RuntimeError):
        moe_router(logits, k=16)


def test_kernel_rejects_too_many_experts():
    logits = torch.randn(4, 300, device="cuda")
    with pytest.raises(ValueError):
        moe_router(logits, k=2)


def test_kernel_no_renormalize_matches_eager():
    case = MoERouterCase("kernel_no_renorm", n_tokens=128, n_experts=32, k=4, dtype=torch.float32, renormalize=False)
    logits = make_inputs(case)
    weights_eager, indices_eager = eager_moe_router(logits, case.k, renormalize=False)
    weights_kernel, indices_kernel = moe_router(logits, case.k, renormalize=False)
    torch.testing.assert_close(weights_kernel, weights_eager, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(indices_kernel, indices_eager)
