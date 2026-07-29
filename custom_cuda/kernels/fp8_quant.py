"""Python entrypoint for Kernel 12 (FP8 Dynamic Quantization & Casting).

Allocates the fp8 output, scale, and (for tensor granularity) amax
scratch tensors, then calls the matching
`custom_cuda._native.fp8_quant_{block,tensor}_fwd`. See
`baselines/fp8_quant.py::eager_fp8_quant` for the reference semantics
this must match.
"""

from __future__ import annotations

import math

import torch

from custom_cuda import _native

__all__ = ["fp8_quant"]

_FP8_TORCH_DTYPES: dict[str, torch.dtype] = {
    "e4m3": torch.float8_e4m3fn,
    "e5m2": torch.float8_e5m2,
}

_BLOCK_SIZE = 128


def fp8_quant(
    x: torch.Tensor, fp8_format: str = "e4m3", granularity: str = "block"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic FP8 quantization: `scale = max(amax / FP8_MAX, eps)`,
    `x_fp8 = (x / scale).to(fp8_dtype)`, fused into a single kernel pass
    (block granularity) or two (tensor granularity — see
    csrc/includes/fp8_quant.h's docstring for why).

    `x`: `[M, N]`, contiguous CUDA tensor (float32/float16/bfloat16).
    `fp8_format`: `"e4m3"` or `"e5m2"`. `granularity`: `"block"` (one
    scale per 128x128 tile) or `"tensor"` (one scale for the whole
    matrix). Returns `(x_fp8, scale)`: `x_fp8: [M, N]` in the requested
    fp8 dtype; `scale`: `[ceil(M/128), ceil(N/128)]` float32 for
    `"block"`, `[1]` float32 for `"tensor"`.
    """
    if fp8_format not in _FP8_TORCH_DTYPES:
        raise ValueError(f"fp8_format must be 'e4m3' or 'e5m2', got {fp8_format!r}")
    if granularity not in ("block", "tensor"):
        raise ValueError(f"granularity must be 'block' or 'tensor', got {granularity!r}")

    m, n = x.shape
    device = x.device
    fp8_dtype = _FP8_TORCH_DTYPES[fp8_format]
    x_fp8 = torch.empty((m, n), dtype=fp8_dtype, device=device)

    if granularity == "block":
        num_row_blocks = math.ceil(m / _BLOCK_SIZE)
        num_col_blocks = math.ceil(n / _BLOCK_SIZE)
        scale = torch.empty(num_row_blocks, num_col_blocks, dtype=torch.float32, device=device)
        _native.fp8_quant_block_fwd(x, x_fp8, scale, fp8_format)
        return x_fp8, scale

    scale = torch.empty(1, dtype=torch.float32, device=device)
    amax_scratch = torch.zeros(1, dtype=torch.int32, device=device)
    _native.fp8_quant_tensor_fwd(x, x_fp8, scale, amax_scratch, fp8_format)
    return x_fp8, scale
