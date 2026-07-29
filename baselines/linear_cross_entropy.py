"""Baseline references for Fused Linear Cross Entropy Loss (Kernel 4).

Scope (documented per the precedent set in baselines/rope.py): this
implements the **forward pass only** (loss value) — no fused backward
(gradient w.r.t. hidden/weight) kernel, and no gradients at all: the
custom kernel call is a raw PyO3 function, opaque to autograd, so
`.backward()` on the result does not propagate through it. A production
fused-CE implementation (Liger-Kernel, Unsloth) also fuses the backward
pass so training never materializes the full `[N, V]` logits *or* its
gradient; building a correct chunked backward (as a proper
`torch.autograd.Function`) is a materially larger undertaking than the
forward streaming reduction, and forward-only already demonstrates the
core technique this kernel is about (online-softmax streaming avoiding
full logits materialization) and is directly useful for eval/perplexity
computation. Noted here rather than silently omitted — see
custom_cuda/kernels/linear_cross_entropy.py's docstring for the concrete
autograd caveat.

Semantics every implementation must reproduce exactly:

    logits = hidden @ weight.T                     # [N, V], never materialized whole by the kernel
    loss_i = -log_softmax(logits_i)[target_i]        # standard cross-entropy
    (targets == ignore_index are excluded from the reduction, matching
    torch.nn.functional.cross_entropy's default ignore_index=-100)

`hidden: [N, H]`, `weight: [V, H]` (an LM head weight, no bias — see
project_plan.md Section 3.4), `targets: [N]` int64.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import torch
import torch._dynamo
import torch.nn.functional as F

torch._dynamo.config.recompile_limit = 64

__all__ = [
    "LinearCECase",
    "STANDARD_CASES",
    "EDGE_CASES",
    "ALL_CASES",
    "make_inputs",
    "eager_linear_cross_entropy",
    "compiled_linear_cross_entropy",
    "reference_fp64",
]

IGNORE_INDEX = -100


@dataclasses.dataclass(frozen=True)
class LinearCECase:
    """One (shape, dtype) configuration shared by tests and benchmarks."""

    name: str
    n_tokens: int
    hidden_dim: int
    vocab_size: int
    dtype: torch.dtype
    ignore_fraction: float = 0.0
    reduction: str = "mean"

    @property
    def hidden_shape(self) -> tuple[int, int]:
        return (self.n_tokens, self.hidden_dim)

    @property
    def weight_shape(self) -> tuple[int, int]:
        return (self.vocab_size, self.hidden_dim)


_DTYPES: tuple[torch.dtype, ...] = (torch.float32, torch.float16, torch.bfloat16)


def _cases_for(name: str, n_tokens: int, hidden_dim: int, vocab_size: int, **kwargs) -> list[LinearCECase]:
    return [
        LinearCECase(f"{name}_{dt}".replace("torch.", ""), n_tokens, hidden_dim, vocab_size, dt, **kwargs)
        for dt in _DTYPES
    ]


# Representative LLM LM-head shapes: GPT-2-ish small vocab, Llama-2 vocab,
# Llama-3 vocab (128k — the shape this kernel exists to make tractable).
STANDARD_CASES: list[LinearCECase] = [
    *_cases_for("small_vocab", n_tokens=256, hidden_dim=768, vocab_size=8000),
    *_cases_for("llama2_vocab", n_tokens=512, hidden_dim=4096, vocab_size=32000),
    *_cases_for("llama3_vocab", n_tokens=256, hidden_dim=4096, vocab_size=128256),
]

# Section 4.3 edge-case battery, plus cases specific to this kernel:
# ignored tokens (padding), a single class, vocab smaller than one chunk,
# vocab spanning many chunks.
EDGE_CASES: list[LinearCECase] = [
    *_cases_for("npot_tokens", n_tokens=257, hidden_dim=256, vocab_size=1009),
    *_cases_for("with_ignored", n_tokens=256, hidden_dim=256, vocab_size=4000, ignore_fraction=0.3),
    *_cases_for("all_ignored", n_tokens=64, hidden_dim=128, vocab_size=1000, ignore_fraction=1.0),
    *_cases_for("single_token", n_tokens=1, hidden_dim=128, vocab_size=1000),
    *_cases_for("empty_batch", n_tokens=0, hidden_dim=128, vocab_size=1000),
    *_cases_for("tiny_vocab", n_tokens=64, hidden_dim=64, vocab_size=3),
    *_cases_for("reduction_sum", n_tokens=128, hidden_dim=128, vocab_size=2000, reduction="sum"),
    *_cases_for("reduction_none", n_tokens=128, hidden_dim=128, vocab_size=2000, reduction="none"),
]

ALL_CASES: list[LinearCECase] = [*STANDARD_CASES, *EDGE_CASES]


def make_inputs(
    case: LinearCECase, device: str = "cuda", seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (hidden, weight, targets) for a case."""
    gen = torch.Generator(device=device).manual_seed(seed)
    hidden = torch.randn(case.hidden_shape, dtype=case.dtype, device=device, generator=gen) * 0.02
    weight = torch.randn(case.weight_shape, dtype=case.dtype, device=device, generator=gen) * 0.02

    if case.n_tokens == 0:
        targets = torch.empty(0, dtype=torch.long, device=device)
        return hidden, weight, targets

    targets = torch.randint(
        0, case.vocab_size, (case.n_tokens,), device=device, generator=gen, dtype=torch.long
    )
    if case.ignore_fraction > 0:
        mask = torch.rand(case.n_tokens, device=device, generator=gen) < case.ignore_fraction
        targets = torch.where(mask, torch.full_like(targets, IGNORE_INDEX), targets)

    return hidden, weight, targets


def eager_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    reduction: str = "mean",
) -> torch.Tensor:
    """PyTorch eager reference: materializes the full `[N, V]` logits
    tensor at hidden/weight's native dtype (this is the memory cost the
    fused kernel exists to avoid), then `F.cross_entropy` (which handles
    numerically-stable log-softmax + NLL internally).
    """
    logits = hidden @ weight.T
    return F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction=reduction)


_compiled_cache: dict[str, Callable] = {}


def compiled_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    reduction: str = "mean",
) -> torch.Tensor:
    key = reduction
    fn = _compiled_cache.get(key)
    if fn is None:
        def _fn(hidden, weight, targets, ignore_index):
            logits = hidden @ weight.T
            return F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction=reduction)

        fn = torch.compile(_fn, mode="max-autotune", fullgraph=True)
        _compiled_cache[key] = fn
    return fn(hidden, weight, targets, ignore_index)


def reference_fp64(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    reduction: str = "mean",
) -> torch.Tensor:
    """fp64 ground truth for correctness tests only."""
    logits = hidden.to(torch.float64) @ weight.to(torch.float64).T
    return F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction=reduction)
