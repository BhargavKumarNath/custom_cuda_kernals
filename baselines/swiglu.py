"""Baseline references for Fused SwiGLU Gated Activation (Kernel 2).

Semantics every implementation — eager, `torch.compile`, and the CUDA
kernel — must reproduce exactly:

    y = SiLU(gate) * up = (gate * sigmoid(gate)) * up

SiLU's reduction/reference math is computed in fp32 regardless of storage
dtype (same convention as Kernel 1 — see baselines/rmsnorm_residual.py),
then cast back to the input dtype before the final multiply.

Shared source of truth for both tests/test_swiglu.py (correctness) and
benchmarks/swiglu_bench.py (performance).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
import torch
import torch._dynamo

# See baselines/rmsnorm_residual.py for why this is raised: the benchmark
# and test suites compile this reference across many distinct (shape,
# dtype) combinations in one process.
torch._dynamo.config.recompile_limit = 64

__all__ = [
    "SwiGLUCase",
    "STANDARD_CASES",
    "EDGE_CASES",
    "ALL_CASES",
    "make_inputs",
    "eager_swiglu",
    "compiled_swiglu",
    "reference_fp64",
]


@dataclasses.dataclass(frozen=True)
class SwiGLUCase:
    """One (shape, dtype) configuration shared by tests and benchmarks."""

    name: str
    batch: int
    seq_len: int
    intermediate_dim: int
    dtype: torch.dtype

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.batch, self.seq_len, self.intermediate_dim)


_DTYPES: tuple[torch.dtype, ...] = (torch.float32, torch.float16, torch.bfloat16)


def _cases_for(name: str, batch: int, seq_len: int, intermediate_dim: int) -> list[SwiGLUCase]:
    return [
        SwiGLUCase(f"{name}_{dt}".replace("torch.", ""), batch, seq_len, intermediate_dim, dt)
        for dt in _DTYPES
    ]


# Representative LLM FFN intermediate-activation shapes.
STANDARD_CASES: list[SwiGLUCase] = [
    *_cases_for("small", batch=4, seq_len=128, intermediate_dim=3072),   # BERT-base-ish FFN
    *_cases_for("medium", batch=2, seq_len=2048, intermediate_dim=11008),  # Llama-7b FFN
    *_cases_for("large", batch=1, seq_len=4096, intermediate_dim=11008),
]

# Section 4.3 edge-case battery.
EDGE_CASES: list[SwiGLUCase] = [
    *_cases_for("npot_batch", batch=3, seq_len=17, intermediate_dim=11008),
    *_cases_for("npot_dim", batch=2, seq_len=64, intermediate_dim=100),
    *_cases_for("seq_len_1", batch=8, seq_len=1, intermediate_dim=11008),
    *_cases_for("long_seq", batch=1, seq_len=8192, intermediate_dim=4096),
    *_cases_for("empty_batch", batch=0, seq_len=128, intermediate_dim=11008),
    *_cases_for("single_elem", batch=1, seq_len=1, intermediate_dim=128),
]

ALL_CASES: list[SwiGLUCase] = [*STANDARD_CASES, *EDGE_CASES]


def make_inputs(
    case: SwiGLUCase,
    device: str = "cuda",
    seed: int = 0,
    contiguous: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (gate, up) for a case. `contiguous=False` slices from a wider
    allocation to exercise non-contiguous handling (Section 4.2).
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    b, s, d = case.shape

    if contiguous:
        gate = torch.randn(b, s, d, dtype=case.dtype, device=device, generator=gen)
        up = torch.randn(b, s, d, dtype=case.dtype, device=device, generator=gen)
    else:
        gate = torch.randn(b, s, d * 2, dtype=case.dtype, device=device, generator=gen)[..., ::2]
        up = torch.randn(b, s, d * 2, dtype=case.dtype, device=device, generator=gen)[..., ::2]

    return gate, up


def eager_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """PyTorch eager reference: `SiLU(gate) * up`, SiLU computed in fp32."""
    orig_dtype = gate.dtype
    g32 = gate.to(torch.float32)
    silu = g32 * torch.sigmoid(g32)
    return silu.to(orig_dtype) * up


_compiled_cache: dict[str, Callable] = {}


def compiled_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """`torch.compile(mode="max-autotune")`-fused reference."""
    fn = _compiled_cache.get("fn")
    if fn is None:
        fn = torch.compile(eager_swiglu, mode="max-autotune", fullgraph=True)
        _compiled_cache["fn"] = fn
    return fn(gate, up)


def reference_fp64(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """fp64 ground truth for correctness tests only."""
    g64 = gate.to(torch.float64)
    u64 = up.to(torch.float64)
    silu = g64 * torch.sigmoid(g64)
    return silu * u64
