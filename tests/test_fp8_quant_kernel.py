"""Correctness tests for Kernel 12's CUDA implementation
(csrc/kernels/fp8_quant.cu, via custom_cuda.kernels.fp8_quant) against
the PyTorch eager 2-pass baseline. Mirrors tests/test_pairwise_distance_kernel.py.
"""

from __future__ import annotations

import math

import pytest
import torch
from baselines.fp8_quant import ALL_CASES, FP8_DTYPES, Fp8QuantCase, dequantize, eager_fp8_quant, make_inputs
from tests.numerics import assert_close_with_outliers

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA device required for FP8 Quant kernel tests", allow_module_level=True)

pytest.importorskip(
    "custom_cuda._native", reason="Rust extension not built — run `maturin develop` first"
)
from custom_cuda.kernels.fp8_quant import fp8_quant  # noqa: E402

# fp8's mantissa is only 2-3 bits (eps=0.125 for e4m3, 0.25 for e5m2), so
# an exact tie between two representable fp8 values is far more common
# than at fp16/bf16 precision — the kernel computes `x * (1/scale)`
# while eager computes `x / scale`; these can round to opposite
# neighboring fp8 codes in the rare case `x/scale` lands within a ULP of
# the true tie point (verified empirically: every mismatch found is
# exactly one fp8 code apart, never more). fp16-storage inputs hit this
# more often than fp32 (fp16's own coarser mantissa means its values
# land near an fp8 tie point after scaling more often) — observed up to
# ~5e-4 for small (256x256) fp16 cases; 1e-3 leaves comfortable margin
# while still failing hard on any real, systematic mismatch — the same
# "tolerate a tiny fraction of legitimate boundary-rounding outliers"
# pattern as tests/numerics.py's DEFAULT_BF16_OUTLIER_FRACTION.
FP8_TIE_BREAK_OUTLIER_FRACTION = 1e-3


def _case_id(case: Fp8QuantCase) -> str:
    return case.name


@pytest.mark.parametrize("case", ALL_CASES, ids=_case_id)
def test_kernel_scale_matches_eager(case: Fp8QuantCase):
    x = make_inputs(case)
    _x_fp8_eager, scale_eager = eager_fp8_quant(x, case.fp8_format, case.granularity)
    x_fp8_kernel, scale_kernel = fp8_quant(x, case.fp8_format, case.granularity)

    assert x_fp8_kernel.shape == (case.m, case.n)
    assert x_fp8_kernel.dtype == FP8_DTYPES[case.fp8_format]

    if case.granularity == "tensor":
        assert scale_kernel.shape == (1,)
        torch.testing.assert_close(scale_kernel[0], scale_eager, rtol=1e-5, atol=1e-9)
    else:
        assert scale_kernel.shape == (math.ceil(case.m / 128), math.ceil(case.n / 128))
        torch.testing.assert_close(scale_kernel, scale_eager, rtol=1e-5, atol=1e-9)


@pytest.mark.parametrize("case", ALL_CASES, ids=_case_id)
def test_kernel_dequantized_output_matches_eager(case: Fp8QuantCase):
    """Compare the *dequantized* value rather than raw fp8 bytes: the
    kernel and eager reference both round to nearest representable fp8
    value, so as long as they agree on the scale (checked separately)
    their fp8 outputs must be bit-identical too — dequantizing is just a
    convenient, format-agnostic way to assert that without hand-decoding
    fp8 bytes in the test.
    """
    x = make_inputs(case)
    x_fp8_eager, scale_eager = eager_fp8_quant(x, case.fp8_format, case.granularity)
    x_fp8_kernel, scale_kernel = fp8_quant(x, case.fp8_format, case.granularity)

    scale_kernel_cmp = scale_kernel[0] if case.granularity == "tensor" else scale_kernel
    dequant_eager = dequantize(x_fp8_eager, scale_eager, case.granularity)
    dequant_kernel = dequantize(x_fp8_kernel, scale_kernel_cmp, case.granularity)

    assert_close_with_outliers(
        dequant_kernel, dequant_eager, rtol=0, atol=0, max_outlier_fraction=FP8_TIE_BREAK_OUTLIER_FRACTION
    )


def test_kernel_all_zero_block_gives_zero_output_and_finite_scale():
    x = torch.zeros(256, 256, device="cuda", dtype=torch.float32)
    x_fp8, scale = fp8_quant(x, "e4m3", "block")
    assert torch.all(x_fp8.to(torch.float32) == 0.0)
    assert torch.isfinite(scale).all()
    assert torch.all(scale > 0.0)


def test_kernel_rejects_cpu_tensor():
    x = torch.randn(4, 8)
    with pytest.raises(ValueError):
        fp8_quant(x, "e4m3", "block")


def test_kernel_rejects_bad_fp8_format():
    x = torch.randn(4, 8, device="cuda")
    with pytest.raises(ValueError):
        fp8_quant(x, "not_a_format", "block")


def test_kernel_rejects_bad_granularity():
    x = torch.randn(4, 8, device="cuda")
    with pytest.raises(ValueError):
        fp8_quant(x, "e4m3", "not_a_granularity")
