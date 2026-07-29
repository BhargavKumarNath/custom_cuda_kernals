"""Baseline references for Fused Rotary Position Embedding (Kernel 3).

Scope (documented in project_plan.md Section 3.3 / Phase 1 checklist):
this implements the **half-split** rotation convention used by
HuggingFace/Llama/Mistral-family models (rotate `(x[i], x[i+d/2])` pairs),
not the interleaved-pairs (original GPT-NeoX, `(x[2i], x[2i+1])`)
convention — the two require materially different kernels, and half-split
is the dominant convention in current open-weight LLMs. Position ids are
assumed to be `arange(seq_len)` (no custom/packed position ids).

Semantics every implementation must reproduce exactly, given precomputed
`cos`, `sin` tables of shape `[seq_len, head_dim/2]` (see
`compute_cos_sin`):

    x1, x2 = x[..., :d/2], x[..., d/2:]
    out[..., :d/2] = x1 * cos - x2 * sin
    out[..., d/2:] = x2 * cos + x1 * sin

applied independently to `q: [B, S, Hq, D]` and `k: [B, S, Hkv, D]` (GQA:
`Hkv <= Hq`, both divide evenly is not required). Reduction/rotation math
is computed in fp32 regardless of storage dtype (same convention as
Kernels 1-2), then cast back.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
import torch
import torch._dynamo
torch._dynamo.config.recompile_limit = 64

__all__ = [
    "RopeCase",
    "STANDARD_CASES",
    "EDGE_CASES",
    "ALL_CASES",
    "compute_cos_sin",
    "make_inputs",
    "eager_rope",
    "compiled_rope",
    "reference_fp64",
]


@dataclasses.dataclass(frozen=True)
class RopeCase:
    """One (shape, dtype) configuration shared by tests and benchmarks."""

    name: str
    batch: int
    seq_len: int
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    theta: float = 10000.0

    @property
    def q_shape(self) -> tuple[int, int, int, int]:
        return (self.batch, self.seq_len, self.n_q_heads, self.head_dim)

    @property
    def k_shape(self) -> tuple[int, int, int, int]:
        return (self.batch, self.seq_len, self.n_kv_heads, self.head_dim)


_DTYPES: tuple[torch.dtype, ...] = (torch.float32, torch.float16, torch.bfloat16)


def _cases_for(
    name: str, batch: int, seq_len: int, n_q_heads: int, n_kv_heads: int, head_dim: int
) -> list[RopeCase]:
    return [
        RopeCase(f"{name}_{dt}".replace("torch.", ""), batch, seq_len, n_q_heads, n_kv_heads, head_dim, dt)
        for dt in _DTYPES
    ]


# Representative LLM attention shapes. "small"/"medium" use MHA
# (n_q_heads == n_kv_heads); "gqa" uses grouped-query attention
# (n_kv_heads < n_q_heads, Llama-3-8b-ish ratio).
STANDARD_CASES: list[RopeCase] = [
    *_cases_for("small_mha", batch=4, seq_len=128, n_q_heads=12, n_kv_heads=12, head_dim=64),
    *_cases_for("medium_gqa", batch=2, seq_len=2048, n_q_heads=32, n_kv_heads=8, head_dim=128),
    *_cases_for("large_gqa", batch=1, seq_len=4096, n_q_heads=32, n_kv_heads=8, head_dim=128),
]

# Section 4.3 edge-case battery. head_dim must be even (RoPE precondition —
# real head_dims always are: 64/128/256), so no odd-head_dim case.
EDGE_CASES: list[RopeCase] = [
    *_cases_for("npot_batch", batch=3, seq_len=17, n_q_heads=8, n_kv_heads=2, head_dim=64),
    *_cases_for("seq_len_1", batch=8, seq_len=1, n_q_heads=8, n_kv_heads=8, head_dim=64),
    *_cases_for("long_seq", batch=1, seq_len=8192, n_q_heads=8, n_kv_heads=8, head_dim=64),
    *_cases_for("empty_batch", batch=0, seq_len=128, n_q_heads=8, n_kv_heads=8, head_dim=64),
    *_cases_for("single_elem", batch=1, seq_len=1, n_q_heads=1, n_kv_heads=1, head_dim=64),
    *_cases_for("small_head_dim", batch=2, seq_len=32, n_q_heads=4, n_kv_heads=1, head_dim=16),
]

ALL_CASES: list[RopeCase] = [*STANDARD_CASES, *EDGE_CASES]


def compute_cos_sin(
    seq_len: int, head_dim: int, theta: float = 10000.0, device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard RoPE frequency table, fp32 (kept fp32 regardless of q/k's
    storage dtype — standard practice for precision, see module docstring).
    Returns (cos, sin), each `[seq_len, head_dim/2]`.
    """
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)  # [seq_len, head_dim/2]
    return freqs.cos().contiguous(), freqs.sin().contiguous()


def make_inputs(
    case: RopeCase, device: str = "cuda", seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (q, k, cos, sin) for a case."""
    gen = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(case.q_shape, dtype=case.dtype, device=device, generator=gen)
    k = torch.randn(case.k_shape, dtype=case.dtype, device=device, generator=gen)
    cos, sin = compute_cos_sin(case.seq_len, case.head_dim, case.theta, device=device)
    return q, k, cos, sin


def _rotate_half_split(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype
    x32 = x.to(torch.float32)
    d = x32.shape[-1]
    x1, x2 = x32[..., : d // 2], x32[..., d // 2 :]
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    out1 = x1 * c - x2 * s
    out2 = x2 * c + x1 * s
    return torch.cat([out1, out2], dim=-1).to(orig_dtype)


def eager_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch eager reference: half-split rotation applied to q and k."""
    return _rotate_half_split(q, cos, sin), _rotate_half_split(k, cos, sin)


_compiled_cache: dict[str, Callable] = {}


def compiled_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    fn = _compiled_cache.get("fn")
    if fn is None:
        fn = torch.compile(eager_rope, mode="max-autotune", fullgraph=True)
        _compiled_cache["fn"] = fn
    return fn(q, k, cos, sin)


def _rotate_half_split_fp64(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x64 = x.to(torch.float64)
    d = x64.shape[-1]
    x1, x2 = x64[..., : d // 2], x64[..., d // 2 :]
    c = cos.to(torch.float64)[None, :, None, :]
    s = sin.to(torch.float64)[None, :, None, :]
    out1 = x1 * c - x2 * s
    out2 = x2 * c + x1 * s
    return torch.cat([out1, out2], dim=-1)


def reference_fp64(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp64 ground truth for correctness tests only."""
    return _rotate_half_split_fp64(q, cos, sin), _rotate_half_split_fp64(k, cos, sin)
