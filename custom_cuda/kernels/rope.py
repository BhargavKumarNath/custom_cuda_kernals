"""Python entrypoint for Kernel 3 (Fused Rotary Position Embedding).

Thin wrapper over `custom_cuda._native.rope_fwd`. Half-split convention,
`position_ids = arange(seq_len)` — see `baselines/rope.py` for the
documented scope and reference semantics this must match.
"""

from __future__ import annotations

import torch
from custom_cuda import _native

__all__ = ["rope"]


def rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate q/k with precomputed cos/sin tables (see
    `baselines.rope.compute_cos_sin`).

    q: `[B, S, Hq, D]`, k: `[B, S, Hkv, D]` (GQA: Hkv <= Hq), same dtype,
    contiguous CUDA tensors. cos, sin: `[S, D/2]`, float32.
    """
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    _native.rope_fwd(q, k, cos, sin, q_out, k_out)
    return q_out, k_out
