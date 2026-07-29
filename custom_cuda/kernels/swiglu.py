"""Python entrypoint for Kernel 2 (Fused SwiGLU Gated Activation).

Thin wrapper over `custom_cuda._native.swiglu_fwd`. See
`baselines/swiglu.py::eager_swiglu` for the reference semantics this must
match.
"""

from __future__ import annotations

import torch
from custom_cuda import _native

__all__ = ["swiglu"]


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """`y = SiLU(gate) * up`. gate, up: same-shape contiguous CUDA tensors."""
    y = torch.empty_like(gate)
    _native.swiglu_fwd(gate, up, y)
    return y
