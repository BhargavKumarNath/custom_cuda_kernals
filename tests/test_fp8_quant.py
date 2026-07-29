"""Correctness tests for Kernel 12 (FP8 Dynamic Quantization & Casting)
baselines. Mirrors tests/test_pairwise_distance.py's structure.
"""

from __future__ import annotations

import pytest
import torch

from baselines.fp8_quant import (
    ALL_CASES,
    FP8_DTYPES,
    Fp8QuantCase,
    dequantize,
    eager_fp8_quant,
    expand_block_scale,
    make_inputs,
    reference_fp64,
)

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA device required for FP8 Quant baseline tests", allow_module_level=True)


def _case_id(case: Fp8QuantCase) -> str:
    return case.name


@pytest.mark.parametrize("case", ALL_CASES, ids=_case_id)
def test_eager_scale_matches_fp64_reference(case: Fp8QuantCase):
    x = make_inputs(case)
    _x_fp8, scale = eager_fp8_quant(x, case.fp8_format, case.granularity)
    scale_ref = reference_fp64(x, case.fp8_format, case.granularity)

    if case.granularity == "tensor":
        assert scale.shape == ()
    else:
        import math

        assert scale.shape == (math.ceil(case.m / 128), math.ceil(case.n / 128))

    torch.testing.assert_close(scale, scale_ref.to(torch.float32), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("case", ALL_CASES, ids=_case_id)
def test_eager_output_shape_and_dtype(case: Fp8QuantCase):
    x = make_inputs(case)
    x_fp8, _scale = eager_fp8_quant(x, case.fp8_format, case.granularity)
    assert x_fp8.shape == (case.m, case.n)
    assert x_fp8.dtype == FP8_DTYPES[case.fp8_format]
    assert torch.isfinite(x_fp8.to(torch.float32)).all()


@pytest.mark.parametrize("case", ALL_CASES, ids=_case_id)
def test_eager_roundtrip_error_bounded_by_quantization_step(case: Fp8QuantCase):
    """The round-trip (quantize -> dequantize) error, expressed in the
    *scaled* domain (i.e. relative to each element's own local scale —
    tensor-wide or per-block), must be bounded by the fp8 format's own
    rounding unit: `eps/2` in the format's normal range, plus a floor of
    `tiny` (the smallest normal magnitude) for the denormal range near
    zero. A 2x safety factor absorbs fp32 arithmetic rounding in the
    scale/divide/multiply chain on top of the fp8 cast's own rounding.
    """
    fp8_dtype = FP8_DTYPES[case.fp8_format]
    fp8_eps = torch.finfo(fp8_dtype).eps
    fp8_tiny = torch.finfo(fp8_dtype).tiny

    x = make_inputs(case)
    x32 = x.to(torch.float32)
    x_fp8, scale = eager_fp8_quant(x, case.fp8_format, case.granularity)
    dequant = dequantize(x_fp8, scale, case.granularity)

    if case.granularity == "tensor":
        scale_expanded = scale
    else:
        scale_expanded = expand_block_scale(scale, 128, case.m, case.n)

    scaled_x = x32 / scale_expanded
    scaled_dequant = dequant / scale_expanded

    bound = 2.0 * (0.5 * fp8_eps * scaled_x.abs() + fp8_tiny)
    err = (scaled_dequant - scaled_x).abs()
    assert torch.all(err <= bound), f"max violation: {(err - bound).max().item()}"


def test_eager_all_zero_block_gives_zero_output_and_finite_scale():
    case = Fp8QuantCase("zero_check", m=256, n=256, dtype=torch.float32, fp8_format="e4m3", granularity="block")
    x = torch.zeros(case.shape, device="cuda", dtype=case.dtype)
    x_fp8, scale = eager_fp8_quant(x, case.fp8_format, case.granularity)
    assert torch.all(x_fp8.to(torch.float32) == 0.0)
    assert torch.isfinite(scale).all()
    assert torch.all(scale > 0.0)


def test_eager_block_granularity_preserves_dynamic_range_better_than_tensor():
    """A matrix with one small-magnitude block and one large-magnitude
    outlier block: tensor-wide scaling must crush the small block's
    precision (its whole range maps to only a few near-zero fp8 codes),
    while block-wise scaling keeps each block independently
    well-resolved.
    """
    device = "cuda"
    m = n = 256  # exactly two 128x128 blocks per axis
    x = torch.zeros(m, n, device=device)
    x[:128, :128] = torch.randn(128, 128, device=device) * 0.01  # tiny-magnitude block
    x[128:, 128:] = torch.randn(128, 128, device=device) * 100.0  # large-magnitude block

    case_tensor = Fp8QuantCase("dyn_range_tensor", m, n, torch.float32, "e4m3", "tensor")
    case_block = Fp8QuantCase("dyn_range_block", m, n, torch.float32, "e4m3", "block")

    x_fp8_tensor, scale_tensor = eager_fp8_quant(x, case_tensor.fp8_format, case_tensor.granularity)
    x_fp8_block, scale_block = eager_fp8_quant(x, case_block.fp8_format, case_block.granularity)

    dequant_tensor = dequantize(x_fp8_tensor, scale_tensor, "tensor")
    dequant_block = dequantize(x_fp8_block, scale_block, "block")

    small_block_orig = x[:128, :128]
    err_tensor = (dequant_tensor[:128, :128] - small_block_orig).abs().mean()
    err_block = (dequant_block[:128, :128] - small_block_orig).abs().mean()

    assert err_block < err_tensor
