"""Baseline references for Fused RMSNorm + Residual Addition (Kernel 1).

Semantics every implementation — eager, `torch.compile`, and eventually the
CUDA kernel — must reproduce exactly:

    residual_out = x + residual
    y = residual_out / sqrt(mean(residual_out**2, dim=-1) + eps) * weight

Two outputs are returned: `y` (the normalized activation fed to the next
sub-layer) and `residual_out` (the pre-norm sum, which becomes the residual
stream input to the *next* block). Returning both is what makes fusion
worthwhile — an unfused implementation writes `residual_out` to global
memory and then reads it straight back in to compute the norm.

This module is the single source of truth for shapes/dtypes used by both
`tests/test_rmsnorm_residual.py` (correctness) and
`benchmarks/rmsnorm_residual_bench.py` (performance), so neither can drift
from what the other exercises.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import torch
import torch._dynamo

# The benchmark and test suites deliberately compile this reference across
# many distinct (shape, dtype) combinations in a single process (Section 5/6
# shape-sweep plots), which exceeds torch._dynamo's default recompile guard
# budget (8) for one wrapped function. Raised rather than caching a separate
# compiled object per shape, since Dynamo's own per-shape specialization via
# guards is exactly the reuse behavior wanted here.
torch._dynamo.config.recompile_limit = 64

__all__ = [
    "RMSNormResidualCase",
    "STANDARD_CASES",
    "EDGE_CASES",
    "ALL_CASES",
    "make_inputs",
    "eager_rmsnorm_residual",
    "compiled_rmsnorm_residual",
    "reference_fp64",
]


@dataclasses.dataclass(frozen=True)
class RMSNormResidualCase:
    """One (shape, dtype) configuration shared by tests and benchmarks."""

    name: str
    batch: int
    seq_len: int
    hidden_dim: int
    dtype: torch.dtype
    eps: float = 1e-6

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.batch, self.seq_len, self.hidden_dim)


_DTYPES: tuple[torch.dtype, ...] = (torch.float32, torch.float16, torch.bfloat16)


def _cases_for(name: str, batch: int, seq_len: int, hidden_dim: int) -> list[RMSNormResidualCase]:
    return [
        RMSNormResidualCase(f"{name}_{dt}".replace("torch.", ""), batch, seq_len, hidden_dim, dt)
        for dt in _DTYPES
    ]


# Representative LLM-shaped cases (attention/MLP block activations).
STANDARD_CASES: list[RMSNormResidualCase] = [
    *_cases_for("small", batch=4, seq_len=128, hidden_dim=768),      # BERT-base-ish
    *_cases_for("medium", batch=2, seq_len=2048, hidden_dim=4096),   # Llama-7b-ish
    *_cases_for("large", batch=1, seq_len=4096, hidden_dim=4096),    # long-context single sequence
]

# Section 4.3 edge-case battery: non-power-of-two sizes, extreme sequence
# lengths, empty and single-element inputs.
EDGE_CASES: list[RMSNormResidualCase] = [
    *_cases_for("npot_batch", batch=3, seq_len=17, hidden_dim=4096),
    *_cases_for("npot_hidden", batch=2, seq_len=64, hidden_dim=100),
    *_cases_for("seq_len_1", batch=8, seq_len=1, hidden_dim=4096),
    *_cases_for("long_seq", batch=1, seq_len=8192, hidden_dim=2048),
    *_cases_for("empty_batch", batch=0, seq_len=128, hidden_dim=4096),
    *_cases_for("single_elem", batch=1, seq_len=1, hidden_dim=128),
]

ALL_CASES: list[RMSNormResidualCase] = [*STANDARD_CASES, *EDGE_CASES]


def make_inputs(
    case: RMSNormResidualCase,
    device: str = "cuda",
    seed: int = 0,
    contiguous: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (x, residual, weight) for a case.

    When `contiguous=False`, x/residual are sliced from a wider allocation
    (stride-2 along the last dim's parent buffer) to exercise the
    non-contiguous handling required by Section 4.2 — still logically the
    requested shape, but with non-unit last-dim stride.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    b, s, h = case.shape

    if contiguous:
        x = torch.randn(b, s, h, dtype=case.dtype, device=device, generator=gen)
        residual = torch.randn(b, s, h, dtype=case.dtype, device=device, generator=gen)
    else:
        x = torch.randn(b, s, h * 2, dtype=case.dtype, device=device, generator=gen)[..., ::2]
        residual = torch.randn(b, s, h * 2, dtype=case.dtype, device=device, generator=gen)[..., ::2]

    weight = torch.randn(h, dtype=case.dtype, device=device, generator=gen)
    return x, residual, weight


def eager_rmsnorm_residual(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch eager reference. Reduction/normalization is always computed in
    fp32 regardless of input dtype (standard RMSNorm numerical-stability
    practice), then cast back to the input dtype.
    """
    orig_dtype = x.dtype
    residual_out = x + residual
    upcast = residual_out.to(torch.float32)
    variance = upcast.pow(2).mean(dim=-1, keepdim=True)
    normed = upcast * torch.rsqrt(variance + eps)
    y = normed.to(orig_dtype) * weight
    return y, residual_out


_compiled_cache: dict[str, Callable] = {}


def compiled_rmsnorm_residual(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """`torch.compile(mode="max-autotune")`-fused reference — the real
    performance bar the CUDA kernel has to clear, since Inductor already
    fuses elementwise chains like this one.
    """
    fn = _compiled_cache.get("fn")
    if fn is None:
        fn = torch.compile(eager_rmsnorm_residual, mode="max-autotune", fullgraph=True)
        _compiled_cache["fn"] = fn
    return fn(x, residual, weight, eps)


def reference_fp64(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp64 ground truth for correctness tests only — never used as a
    benchmark baseline (fp64 throughput is not representative).
    """
    x64 = x.to(torch.float64)
    r64 = residual.to(torch.float64)
    w64 = weight.to(torch.float64)
    residual_out = x64 + r64
    variance = residual_out.pow(2).mean(dim=-1, keepdim=True)
    normed = residual_out * torch.rsqrt(variance + eps)
    y = normed * w64
    return y, residual_out
