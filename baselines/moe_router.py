"""Baseline references for Mixture of Experts (MoE) Top-K Router (Kernel 6).

Scope (documented per the precedent set in baselines/rope.py): this
implements **softmax** gating only (Mixtral/Switch-Transformer/GShard-style
routing) — the sigmoid-gating variant used by some MoE architectures
(e.g. DeepSeek-V3) is not implemented. The two require different
normalization semantics, and softmax is the more widely used convention
across current open-weight MoE models. Noted here rather than silently
omitted.

Semantics every implementation must reproduce exactly:

    probs = softmax(logits, dim=-1)             # [T, E], fp32 always
    topk_weights, topk_indices = topk(probs, k, dim=-1)
    if renormalize:
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

`logits: [T, E]` (T tokens, E experts), `k <= E`. Softmax is computed in
fp32 regardless of storage dtype (same convention as every other kernel in
this library), and `topk_weights` is returned as fp32 regardless of
`logits`'s dtype (it's consumed as a multiplicative scale in Kernel 7's
combine step, where precision matters more than matching the activation
dtype).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import torch
import torch._dynamo
import torch.nn.functional as F

torch._dynamo.config.recompile_limit = 64

__all__ = [
    "MoERouterCase",
    "STANDARD_CASES",
    "EDGE_CASES",
    "ALL_CASES",
    "make_inputs",
    "eager_moe_router",
    "compiled_moe_router",
    "reference_fp64",
]


@dataclasses.dataclass(frozen=True)
class MoERouterCase:
    """One (shape, dtype) configuration shared by tests and benchmarks."""

    name: str
    n_tokens: int
    n_experts: int
    k: int
    dtype: torch.dtype
    renormalize: bool = True

    @property
    def logits_shape(self) -> tuple[int, int]:
        return (self.n_tokens, self.n_experts)


_DTYPES: tuple[torch.dtype, ...] = (torch.float32, torch.float16, torch.bfloat16)


def _cases_for(name: str, n_tokens: int, n_experts: int, k: int, **kwargs) -> list[MoERouterCase]:
    return [
        MoERouterCase(f"{name}_{dt}".replace("torch.", ""), n_tokens, n_experts, k, dt, **kwargs)
        for dt in _DTYPES
    ]


# Representative MoE shapes: Mixtral (8 experts, top-2), DeepSeek-V2-ish
# (160 experts, top-6 rounded to top-8 for the k-sweep), and a k=1/k=4
# sweep at a mid-size expert count, per the k in {1,2,4,8} requirement.
STANDARD_CASES: list[MoERouterCase] = [
    *_cases_for("mixtral_like", n_tokens=2048, n_experts=8, k=2),
    *_cases_for("deepseek_like", n_tokens=2048, n_experts=160, k=6),
    *_cases_for("ksweep_k1", n_tokens=1024, n_experts=32, k=1),
    *_cases_for("ksweep_k2", n_tokens=1024, n_experts=32, k=2),
    *_cases_for("ksweep_k4", n_tokens=1024, n_experts=32, k=4),
    *_cases_for("ksweep_k8", n_tokens=1024, n_experts=32, k=8),
]

# Section 4.3 edge-case battery, plus cases specific to this kernel:
# single expert, k == n_experts (select-all), non-power-of-two experts,
# no renormalization.
EDGE_CASES: list[MoERouterCase] = [
    *_cases_for("npot_experts", n_tokens=256, n_experts=100, k=4),
    *_cases_for("npot_tokens", n_tokens=257, n_experts=16, k=2),
    *_cases_for("single_expert", n_tokens=64, n_experts=1, k=1),
    *_cases_for("k_eq_experts", n_tokens=64, n_experts=8, k=8),
    *_cases_for("large_expert_count", n_tokens=128, n_experts=256, k=8),
    *_cases_for("no_renorm", n_tokens=256, n_experts=16, k=2, renormalize=False),
    *_cases_for("single_token", n_tokens=1, n_experts=16, k=2),
    *_cases_for("empty_batch", n_tokens=0, n_experts=16, k=2),
]

ALL_CASES: list[MoERouterCase] = [*STANDARD_CASES, *EDGE_CASES]


def make_inputs(case: MoERouterCase, device: str = "cuda", seed: int = 0) -> torch.Tensor:
    """Build `logits: [T, E]` for a case."""
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(case.logits_shape, dtype=case.dtype, device=device, generator=gen)


def eager_moe_router(
    logits: torch.Tensor, k: int, renormalize: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch eager reference. Softmax computed in fp32 regardless of
    `logits`'s storage dtype; returned weights stay fp32.
    """
    probs = F.softmax(logits.to(torch.float32), dim=-1)
    topk_weights, topk_indices = torch.topk(probs, k, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_indices


_compiled_cache: dict[str, Callable] = {}


def compiled_moe_router(
    logits: torch.Tensor, k: int, renormalize: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    key = f"{k}_{renormalize}"
    fn = _compiled_cache.get(key)
    if fn is None:
        def _fn(logits):
            return eager_moe_router(logits, k, renormalize)

        fn = torch.compile(_fn, mode="max-autotune", fullgraph=True)
        _compiled_cache[key] = fn
    return fn(logits)


def reference_fp64(
    logits: torch.Tensor, k: int, renormalize: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp64 ground truth for correctness tests only."""
    probs = F.softmax(logits.to(torch.float64), dim=-1)
    topk_weights, topk_indices = torch.topk(probs, k, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_indices
