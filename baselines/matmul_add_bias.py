"""Baseline references for Fused MatMul + Add Bias (Kernel 5).

Semantics every implementation must reproduce exactly (the `nn.Linear`
convention):

    y = x @ weight.T + bias

`x: [M, K]`, `weight: [N, K]` (`[out_features, in_features]`), `bias: [N]`
or `None`, `y: [M, N]`.

Two eager baselines are provided, deliberately distinguishing the two
comparison points in project_plan.md Section 3.5's success criterion
("cuBLAS GEMM + a discrete bias-add kernel"):

  - `eager_matmul_add_bias_unfused`: `(x @ weight.T) + bias` — two
    separate ops/kernel launches, the naive pattern the fusion target is
    measured against.
  - `eager_matmul_add_bias_fused`: `F.linear(x, weight, bias)` — PyTorch's
    own fused cuBLAS(Lt) epilogue path, a much higher bar (beating raw
    cuBLAS itself is not this kernel's target — see the module docstring
    in csrc/kernels/matmul_add_bias.cu).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import torch
import torch._dynamo
import torch.nn.functional as F

torch._dynamo.config.recompile_limit = 64

__all__ = [
    "MatmulBiasCase",
    "STANDARD_CASES",
    "EDGE_CASES",
    "ALL_CASES",
    "make_inputs",
    "eager_matmul_add_bias_unfused",
    "eager_matmul_add_bias_fused",
    "compiled_matmul_add_bias",
    "reference_fp64",
]


@dataclasses.dataclass(frozen=True)
class MatmulBiasCase:
    """One (shape, dtype) configuration shared by tests and benchmarks."""

    name: str
    m: int
    k: int
    n: int
    dtype: torch.dtype
    has_bias: bool = True

    @property
    def x_shape(self) -> tuple[int, int]:
        return (self.m, self.k)

    @property
    def weight_shape(self) -> tuple[int, int]:
        return (self.n, self.k)

    @property
    def y_shape(self) -> tuple[int, int]:
        return (self.m, self.n)


_DTYPES: tuple[torch.dtype, ...] = (torch.float32, torch.float16, torch.bfloat16)


def _cases_for(name: str, m: int, k: int, n: int, **kwargs) -> list[MatmulBiasCase]:
    return [
        MatmulBiasCase(f"{name}_{dt}".replace("torch.", ""), m, k, n, dt, **kwargs) for dt in _DTYPES
    ]


# Representative LLM linear-layer shapes: QKV/output projection, MLP
# up/down projection.
STANDARD_CASES: list[MatmulBiasCase] = [
    *_cases_for("qkv_proj", m=2048, k=4096, n=4096),
    *_cases_for("mlp_up", m=2048, k=4096, n=11008),
    *_cases_for("mlp_down", m=2048, k=11008, n=4096),
]

# Section 4.3 edge-case battery, plus cases specific to this kernel:
# no-bias, tall-skinny / short-wide, K=1 (degenerate reduction).
EDGE_CASES: list[MatmulBiasCase] = [
    *_cases_for("npot_dims", m=257, k=513, n=129),
    *_cases_for("no_bias", m=256, k=512, n=256, has_bias=False),
    *_cases_for("tall_skinny", m=4096, k=64, n=64),
    *_cases_for("short_wide", m=64, k=64, n=4096),
    *_cases_for("k_eq_1", m=128, k=1, n=128),
    *_cases_for("single_row", m=1, k=256, n=256),
    *_cases_for("empty_m", m=0, k=256, n=256),
]

ALL_CASES: list[MatmulBiasCase] = [*STANDARD_CASES, *EDGE_CASES]


def make_inputs(
    case: MatmulBiasCase, device: str = "cuda", seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Build (x, weight, bias) for a case."""
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(case.x_shape, dtype=case.dtype, device=device, generator=gen) * 0.02
    weight = torch.randn(case.weight_shape, dtype=case.dtype, device=device, generator=gen) * 0.02
    bias = None
    if case.has_bias:
        bias = torch.randn(case.n, dtype=case.dtype, device=device, generator=gen) * 0.02
    return x, weight, bias


def eager_matmul_add_bias_unfused(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    """Naive two-op pattern: a plain matmul, then a separate broadcast add
    — the baseline Section 3.5's success criterion is measured against.
    """
    y = x @ weight.T
    if bias is not None:
        y = y + bias
    return y


def eager_matmul_add_bias_fused(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    """PyTorch's own fused cuBLAS(Lt) epilogue path — a much higher bar."""
    return F.linear(x, weight, bias)


_compiled_cache: dict[str, Callable] = {}


def compiled_matmul_add_bias(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    fn = _compiled_cache.get("fn")
    if fn is None:
        fn = torch.compile(eager_matmul_add_bias_unfused, mode="max-autotune", fullgraph=True)
        _compiled_cache["fn"] = fn
    return fn(x, weight, bias)


def reference_fp64(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    """fp64 ground truth for correctness tests only."""
    y = x.to(torch.float64) @ weight.to(torch.float64).T
    if bias is not None:
        y = y + bias.to(torch.float64)
    return y
