"""Baseline references for FP8 Dynamic Quantization & Casting (Kernel 12).

Scope: given an fp32/fp16/bf16 activation matrix `x: [M, N]`, compute a
*dynamic* scaling factor from `x`'s own amax (no calibration / running
statistics — recomputed fresh every call, matching DeepSeek-V3-style
per-forward-pass fp8 recipes) and cast to fp8 (e4m3fn or e5m2):

    scale = clamp(amax / FP8_MAX, min=EPS)
    x_fp8 = (x.float() / scale).to(fp8_dtype)

Two scale granularities (project_plan.md Section 3.12):
  - "tensor": one scalar `scale` for the whole matrix.
  - "block": one `scale` per `128x128` tile (`DeepSeek-V3`'s recipe —
    preserves dynamic range much better than a single tensor-wide scale
    when activation magnitude varies a lot across the matrix). Edge
    ("ragged") tiles at the matrix boundary when `M`/`N` aren't multiples
    of 128 are handled directly (amax over the partial tile), not by
    padding the tensor itself.

`FP8_MAX`/`EPS` come directly from `torch.finfo(fp8_dtype)` rather than
hardcoded constants, so this stays correct if a future torch version
changes the format's exact representable range. The "naive" 2-pass
reference this kernel must beat is exactly `eager_fp8_quant` below: a
first pass computing amax (`_block_amax` or `.abs().max()`), then a
second pass scaling and casting — two full read/write traversals of
`x`/`x_fp8` where the fused kernel does one.

Round-trip (quantize -> dequantize) error is bounded by the fp8 format's
own relative rounding unit (`torch.finfo(fp8_dtype).eps`) *of the scaled
value*, not of `x` directly — see `test_fp8_quant.py`'s
`test_eager_roundtrip_error_bounded_by_quantization_step` for the exact
bound and why.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F

__all__ = [
    "Fp8QuantCase",
    "STANDARD_CASES",
    "EDGE_CASES",
    "ALL_CASES",
    "FP8_DTYPES",
    "make_inputs",
    "eager_fp8_quant",
    "dequantize",
    "reference_fp64",
    "block_amax",
    "expand_block_scale",
]

FP8_DTYPES: dict[str, torch.dtype] = {
    "e4m3": torch.float8_e4m3fn,
    "e5m2": torch.float8_e5m2,
}

_BLOCK_SIZE = 128
_EPS = 1e-12


@dataclasses.dataclass(frozen=True)
class Fp8QuantCase:
    """One (shape, dtype, fp8 format, granularity) configuration shared
    by tests and benchmarks.
    """

    name: str
    m: int
    n: int
    dtype: torch.dtype
    fp8_format: str  # "e4m3" | "e5m2"
    granularity: str  # "tensor" | "block"

    @property
    def shape(self) -> tuple[int, int]:
        return (self.m, self.n)


_DTYPES: tuple[torch.dtype, ...] = (torch.float32, torch.float16, torch.bfloat16)
_FORMATS: tuple[str, ...] = ("e4m3", "e5m2")
_GRANULARITIES: tuple[str, ...] = ("tensor", "block")


def _cases_for(name: str, m: int, n: int) -> list[Fp8QuantCase]:
    return [
        Fp8QuantCase(f"{name}_{dt}".replace("torch.", "") + f"_{fmt}_{gran}", m, n, dt, fmt, gran)
        for dt in _DTYPES
        for fmt in _FORMATS
        for gran in _GRANULARITIES
    ]


# Representative activation-matrix shapes: exact multiples of the 128
# tile size, plus a larger "realistic" shape.
STANDARD_CASES: list[Fp8QuantCase] = [
    *_cases_for("small", m=256, n=256),
    *_cases_for("medium", m=1024, n=1024),
    *_cases_for("large", m=4096, n=4096),
]

# Section 4.3-style edge-case battery, plus this kernel's own: ragged
# (non-multiple-of-128) tile boundaries, a tile smaller than 128 in
# either dimension, single row/col, an all-zero block (amax=0), and a
# tensor with one extreme outlier value (stress-tests dynamic-range
# preservation — the whole reason block granularity exists).
EDGE_CASES: list[Fp8QuantCase] = [
    *_cases_for("npot_ragged", m=257, n=513),
    *_cases_for("smaller_than_tile", m=64, n=64),
    *_cases_for("single_row", m=1, n=256),
    *_cases_for("single_col", m=256, n=1),
    *_cases_for("tall_skinny", m=2048, n=128),
]

ALL_CASES: list[Fp8QuantCase] = [*STANDARD_CASES, *EDGE_CASES]


def make_inputs(case: Fp8QuantCase, device: str = "cuda", seed: int = 0) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(case.shape, dtype=case.dtype, device=device, generator=gen)


def _pad_to_multiple(x_abs: torch.Tensor, block_size: int) -> torch.Tensor:
    m, n = x_abs.shape
    pad_m = (-m) % block_size
    pad_n = (-n) % block_size
    if pad_m or pad_n:
        x_abs = F.pad(x_abs, (0, pad_n, 0, pad_m), value=0.0)
    return x_abs


def block_amax(x_abs: torch.Tensor, block_size: int = _BLOCK_SIZE) -> torch.Tensor:
    """Vectorized per-`block_size x block_size`-tile amax via reshape,
    zero-padding ragged edge tiles first (padding with 0 doesn't affect
    amax of `abs()` values). Returns `[num_row_blocks, num_col_blocks]`.
    """
    padded = _pad_to_multiple(x_abs, block_size)
    pm, pn = padded.shape
    nbr, nbc = pm // block_size, pn // block_size
    blocks = padded.view(nbr, block_size, nbc, block_size)
    return blocks.amax(dim=(1, 3))


def expand_block_scale(scale_blocks: torch.Tensor, block_size: int, m: int, n: int) -> torch.Tensor:
    expanded = scale_blocks.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    return expanded[:m, :n]


def eager_fp8_quant(
    x: torch.Tensor, fp8_format: str, granularity: str, block_size: int = _BLOCK_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch eager 2-pass reference: a separate amax-reduction pass
    followed by a separate scale-and-cast pass — the actual comparison
    point this kernel's fused single pass must beat.

    Returns `(x_fp8, scale)`: for `granularity="tensor"`, `scale` is a
    0-D tensor; for `"block"`, `scale: [ceil(M/128), ceil(N/128)]`.
    """
    fp8_dtype = FP8_DTYPES[fp8_format]
    fp8_max = torch.finfo(fp8_dtype).max
    x32 = x.to(torch.float32)
    x_abs = x32.abs()

    if granularity == "tensor":
        amax = x_abs.max()
        scale = torch.clamp(amax / fp8_max, min=_EPS)
        x_fp8 = (x32 / scale).to(fp8_dtype)
        return x_fp8, scale

    m, n = x.shape
    block_amax_vals = block_amax(x_abs, block_size)
    scale_blocks = torch.clamp(block_amax_vals / fp8_max, min=_EPS)
    scale_expanded = expand_block_scale(scale_blocks, block_size, m, n)
    x_fp8 = (x32 / scale_expanded).to(fp8_dtype)
    return x_fp8, scale_blocks


def dequantize(x_fp8: torch.Tensor, scale: torch.Tensor, granularity: str, block_size: int = _BLOCK_SIZE) -> torch.Tensor:
    """`x_fp8.float() * scale`, broadcasting a per-block scale back up
    to the full matrix shape for `granularity="block"`.
    """
    if granularity == "tensor":
        return x_fp8.to(torch.float32) * scale
    m, n = x_fp8.shape
    scale_expanded = expand_block_scale(scale, block_size, m, n)
    return x_fp8.to(torch.float32) * scale_expanded


def reference_fp64(
    x: torch.Tensor, fp8_format: str, granularity: str, block_size: int = _BLOCK_SIZE
) -> torch.Tensor:
    """fp64 ground truth for the *scale* computation only (correctness
    tests only) — the fp8 cast itself is inherently lossy, so there is
    no meaningful "fp64 ground truth" for `x_fp8`'s bit pattern, only
    for the scale that should have been derived from `x`.
    """
    fp8_dtype = FP8_DTYPES[fp8_format]
    fp8_max = torch.finfo(fp8_dtype).max
    x64 = x.to(torch.float64)
    x_abs = x64.abs()

    if granularity == "tensor":
        amax = x_abs.max()
        return torch.clamp(amax / fp8_max, min=_EPS)

    block_amax_vals = block_amax(x_abs, block_size)
    return torch.clamp(block_amax_vals / fp8_max, min=_EPS)
