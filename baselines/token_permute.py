"""Baseline references for Token Scatter and Gather / Permute-Unpermute
(Kernel 7).

Architecture (documented per the precedent set by Kernel 4's
matmul-delegated-to-cuBLAS split): computing *which* row goes *where* —
the permutation/inverse-permutation index arrays — is index bookkeeping,
not the bandwidth-critical operation this kernel is about, so it's
computed with a plain `argsort` in `compute_permutation` below. The custom
CUDA kernels (csrc/kernels/token_permute.cu) do the actual work: gathering
full hidden-dim rows according to a precomputed index, vectorized. Both
directions are expressed as pure **gathers** (never scatter-add/atomics):

  - Permute: `permuted[i] = hidden[permute_index[i]]` — reorders T*k
    (token, expert) assignments into contiguous per-expert order, ready
    for each expert's batched FFN.
  - Unpermute (combine): `combined[t] = sum_j weight[t,j] *
    expert_output[unpermute_index[t,j]]` — gathers each token's k
    expert outputs back by their original assignment and fuses the
    weighted-gate combination into the same pass, rather than a separate
    gather + multiply + sum chain.

`compute_permutation` derives `permute_index: [T*k]` and
`unpermute_index: [T, k]` from Kernel 6's `topk_indices: [T, k]` via one
`argsort` over the flattened expert assignments (stable, so ties within
an expert preserve token order) and its inverse — `unpermute_index` is
exactly the position each (token, k-slot) assignment landed at in the
sorted (permuted) order, so gathering `expert_output` at those positions
recovers per-token results with no atomics required.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import torch
import torch._dynamo

torch._dynamo.config.recompile_limit = 64

__all__ = [
    "TokenGatherCase",
    "TokenCombineCase",
    "GATHER_STANDARD_CASES",
    "GATHER_EDGE_CASES",
    "GATHER_ALL_CASES",
    "COMBINE_STANDARD_CASES",
    "COMBINE_EDGE_CASES",
    "COMBINE_ALL_CASES",
    "compute_permutation",
    "make_gather_inputs",
    "make_combine_inputs",
    "eager_token_gather",
    "compiled_token_gather",
    "eager_token_combine",
    "compiled_token_combine",
    "reference_gather_fp64",
    "reference_combine_fp64",
]


def compute_permutation(topk_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """From `topk_indices: [T, k]` (expert assignment per token per slot),
    derive `permute_index: [T*k]` (gather source token for each permuted
    slot, sorted by expert) and `unpermute_index: [T, k]` (inverse: which
    permuted slot each (token, k-slot) assignment landed at).
    """
    t, k = topk_indices.shape
    flat_experts = topk_indices.reshape(-1)
    sort_order = torch.argsort(flat_experts, stable=True)  # [T*k]
    permute_index = sort_order // k
    inverse_sort_order = torch.empty_like(sort_order)
    inverse_sort_order[sort_order] = torch.arange(t * k, device=topk_indices.device)
    unpermute_index = inverse_sort_order.reshape(t, k)
    return permute_index, unpermute_index


@dataclasses.dataclass(frozen=True)
class TokenGatherCase:
    """One (shape, dtype) configuration for the permute (gather) op."""

    name: str
    n_src_rows: int
    n_dst_rows: int
    hidden_dim: int
    dtype: torch.dtype

    @property
    def src_shape(self) -> tuple[int, int]:
        return (self.n_src_rows, self.hidden_dim)

    @property
    def dst_shape(self) -> tuple[int, int]:
        return (self.n_dst_rows, self.hidden_dim)


@dataclasses.dataclass(frozen=True)
class TokenCombineCase:
    """One (shape, dtype) configuration for the unpermute (weighted
    combine) op.
    """

    name: str
    n_tokens: int
    k: int
    hidden_dim: int
    dtype: torch.dtype

    @property
    def n_expert_rows(self) -> int:
        return self.n_tokens * self.k

    @property
    def expert_output_shape(self) -> tuple[int, int]:
        return (self.n_expert_rows, self.hidden_dim)

    @property
    def combined_shape(self) -> tuple[int, int]:
        return (self.n_tokens, self.hidden_dim)


_DTYPES: tuple[torch.dtype, ...] = (torch.float32, torch.float16, torch.bfloat16)


def _gather_cases_for(name: str, n_src_rows: int, n_dst_rows: int, hidden_dim: int) -> list[TokenGatherCase]:
    return [
        TokenGatherCase(f"{name}_{dt}".replace("torch.", ""), n_src_rows, n_dst_rows, hidden_dim, dt)
        for dt in _DTYPES
    ]


def _combine_cases_for(name: str, n_tokens: int, k: int, hidden_dim: int) -> list[TokenCombineCase]:
    return [
        TokenCombineCase(f"{name}_{dt}".replace("torch.", ""), n_tokens, k, hidden_dim, dt)
        for dt in _DTYPES
    ]


# Representative MoE dispatch shapes: Mixtral-ish and Llama-3-ish hidden
# dims, T*k in the thousands (realistic per-step token*top-k count).
GATHER_STANDARD_CASES: list[TokenGatherCase] = [
    *_gather_cases_for("mixtral_like", n_src_rows=2048, n_dst_rows=4096, hidden_dim=4096),
    *_gather_cases_for("large_hidden", n_src_rows=2048, n_dst_rows=4096, hidden_dim=11008),
]

GATHER_EDGE_CASES: list[TokenGatherCase] = [
    *_gather_cases_for("npot_hidden", n_src_rows=257, n_dst_rows=513, hidden_dim=100),
    *_gather_cases_for("single_row", n_src_rows=1, n_dst_rows=1, hidden_dim=256),
    *_gather_cases_for("empty_dst", n_src_rows=64, n_dst_rows=0, hidden_dim=256),
    *_gather_cases_for("small_hidden", n_src_rows=128, n_dst_rows=256, hidden_dim=16),
]

GATHER_ALL_CASES: list[TokenGatherCase] = [*GATHER_STANDARD_CASES, *GATHER_EDGE_CASES]

COMBINE_STANDARD_CASES: list[TokenCombineCase] = [
    *_combine_cases_for("mixtral_like", n_tokens=2048, k=2, hidden_dim=4096),
    *_combine_cases_for("deepseek_like", n_tokens=2048, k=6, hidden_dim=4096),
]

COMBINE_EDGE_CASES: list[TokenCombineCase] = [
    *_combine_cases_for("k1", n_tokens=256, k=1, hidden_dim=512),
    *_combine_cases_for("k8", n_tokens=256, k=8, hidden_dim=512),
    *_combine_cases_for("npot_hidden", n_tokens=257, k=3, hidden_dim=100),
    *_combine_cases_for("single_token", n_tokens=1, k=2, hidden_dim=256),
    *_combine_cases_for("empty_batch", n_tokens=0, k=2, hidden_dim=256),
    *_combine_cases_for("small_hidden", n_tokens=128, k=4, hidden_dim=16),
]

COMBINE_ALL_CASES: list[TokenCombineCase] = [*COMBINE_STANDARD_CASES, *COMBINE_EDGE_CASES]


def make_gather_inputs(
    case: TokenGatherCase, device: str = "cuda", seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(seed)
    src = torch.randn(case.src_shape, dtype=case.dtype, device=device, generator=gen)
    if case.n_dst_rows == 0 or case.n_src_rows == 0:
        indices = torch.empty(case.n_dst_rows, dtype=torch.long, device=device)
    else:
        indices = torch.randint(
            0, case.n_src_rows, (case.n_dst_rows,), device=device, generator=gen, dtype=torch.long
        )
    return src, indices


def make_combine_inputs(
    case: TokenCombineCase, device: str = "cuda", seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(seed)
    expert_output = torch.randn(case.expert_output_shape, dtype=case.dtype, device=device, generator=gen)
    weights = torch.rand(case.n_tokens, case.k, dtype=torch.float32, device=device, generator=gen)
    if case.n_expert_rows == 0:
        unpermute_index = torch.empty(case.n_tokens, case.k, dtype=torch.long, device=device)
    else:
        # A random permutation of range(n_expert_rows) reshaped to [T, k]
        # — each expert-output row is claimed by exactly one (token,
        # k-slot), matching the real dispatch/combine invariant.
        unpermute_index = torch.randperm(case.n_expert_rows, device=device, generator=gen).reshape(
            case.n_tokens, case.k
        )
    return expert_output, unpermute_index, weights


def eager_token_gather(src: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """PyTorch eager reference: `dst[i] = src[indices[i]]`."""
    return src.index_select(0, indices)


_compiled_gather_cache: dict[str, Callable] = {}


def compiled_token_gather(src: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    fn = _compiled_gather_cache.get("fn")
    if fn is None:
        fn = torch.compile(eager_token_gather, mode="max-autotune", fullgraph=True)
        _compiled_gather_cache["fn"] = fn
    return fn(src, indices)


def reference_gather_fp64(src: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return src.to(torch.float64).index_select(0, indices)


def eager_token_combine(
    expert_output: torch.Tensor, unpermute_index: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """PyTorch eager reference:
    `combined[t] = sum_j weights[t,j] * expert_output[unpermute_index[t,j]]`.
    """
    if unpermute_index.numel() == 0:
        t = unpermute_index.shape[0]
        h = expert_output.shape[-1]
        return torch.zeros(t, h, dtype=expert_output.dtype, device=expert_output.device)
    gathered = expert_output[unpermute_index]  # [T, k, H]
    combined = (gathered.to(torch.float32) * weights.unsqueeze(-1)).sum(dim=1)
    return combined.to(expert_output.dtype)


_compiled_combine_cache: dict[str, Callable] = {}


def compiled_token_combine(
    expert_output: torch.Tensor, unpermute_index: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    fn = _compiled_combine_cache.get("fn")
    if fn is None:
        fn = torch.compile(eager_token_combine, mode="max-autotune", fullgraph=True)
        _compiled_combine_cache["fn"] = fn
    return fn(expert_output, unpermute_index, weights)


def reference_combine_fp64(
    expert_output: torch.Tensor, unpermute_index: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    if unpermute_index.numel() == 0:
        t = unpermute_index.shape[0]
        h = expert_output.shape[-1]
        return torch.zeros(t, h, dtype=torch.float64, device=expert_output.device)
    gathered = expert_output.to(torch.float64)[unpermute_index]
    return (gathered * weights.to(torch.float64).unsqueeze(-1)).sum(dim=1)
