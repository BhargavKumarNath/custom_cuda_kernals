"""Baseline references for Spatiotemporal Graph Message Passing (Kernel 10).

Scope: one step of neighborhood aggregation over a node set with two
independent, directed edge sets — **spatial** edges (within the current
timestep) and **temporal** edges (from the previous timestep's node
features into the current timestep) — matching project_plan.md Section
3.10's spec exactly:

    out[dst] = sum over spatial edges (src -> dst):  spatial_weight * x_curr[src]
             + sum over temporal edges (src -> dst): temporal_weight * x_prev[src]

Both edge sets are given as plain COO `(src, dst, weight)` triples (the
same node-index space for `x_curr`/`x_prev`, matching the standard
ST-GNN assumption of a fixed node set observed across timesteps).
Duplicate `(src, dst)` pairs are valid and their contributions sum (this
is exactly what `index_add_`/`scatter_add_` do, and what the CUDA kernel
must reproduce). `x_prev` and temporal edges may be entirely absent
(`num_temporal_edges=0`) for a first-timestep call.

The "vendor" comparison point named in Section 3.10 is PyTorch
Geometric's `scatter_add`-based message passing; `torch_geometric` is a
heavy, platform-specific optional dependency (not installed here — see
project_build_environment notes), so rather than requiring it,
`eager_spatiotemporal_mp` below is written directly against
`torch.Tensor.index_add_`, the exact primitive PyG's `MessagePassing`
uses internally for `aggr="add"` — the same "build the actual comparison
primitive natively" choice Kernel 9 made with `torch.cdist`.

Accumulation is always in fp32 regardless of storage dtype (RMSNorm/
SwiGLU/RoPE convention), with the result cast back to the input dtype.
Edge weights are always fp32 (derived/precomputed quantities, like
Kernel 9's row norms — not a "real" feature tensor to quantize).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import torch
import torch._dynamo

torch._dynamo.config.recompile_limit = 64

__all__ = [
    "GraphMessagePassingCase",
    "STANDARD_CASES",
    "EDGE_CASES",
    "ALL_CASES",
    "make_inputs",
    "eager_spatiotemporal_mp",
    "compiled_spatiotemporal_mp",
    "reference_fp64",
]


@dataclasses.dataclass(frozen=True)
class GraphMessagePassingCase:
    """One (shape, dtype) configuration shared by tests and benchmarks.

    `spatial_pattern`/`temporal_pattern` control how edges are
    synthesized:
      - "random": `num_*_edges` edges with uniform-random (src, dst) in
        `[0, num_nodes)` — duplicates and self-loops both occur
        naturally, exercising the aggregation's handling of both.
      - "self_loop": exactly `num_nodes` edges, `src[i] = dst[i] = i` —
        the common "temporal edge = identity across time" ST-GNN design.
      - "star_hub": every node except node 0 sends a single edge to node
        0 — a deliberate degree-skew stress case (one node has in-degree
        `num_nodes - 1`, every other node has in-degree 0).
      - "none": zero edges in that set.
    """

    name: str
    num_nodes: int
    feature_dim: int
    num_spatial_edges: int
    num_temporal_edges: int
    dtype: torch.dtype
    spatial_pattern: str = "random"
    temporal_pattern: str = "random"

    @property
    def x_shape(self) -> tuple[int, int]:
        return (self.num_nodes, self.feature_dim)


_DTYPES: tuple[torch.dtype, ...] = (torch.float32, torch.float16, torch.bfloat16)


def _cases_for(name: str, **kwargs) -> list[GraphMessagePassingCase]:
    return [
        GraphMessagePassingCase(f"{name}_{dt}".replace("torch.", ""), dtype=dt, **kwargs)
        for dt in _DTYPES
    ]


# Representative traffic-forecasting/sensor-network shapes: moderate node
# counts with typical average degree, plus a 100K+-node case matching
# Section 3.10's stated success-criteria scale directly.
STANDARD_CASES: list[GraphMessagePassingCase] = [
    *_cases_for(
        "sensor_net_small", num_nodes=2000, feature_dim=64,
        num_spatial_edges=20_000, num_temporal_edges=2000, temporal_pattern="self_loop",
    ),
    *_cases_for(
        "sensor_net_medium", num_nodes=20_000, feature_dim=64,
        num_spatial_edges=300_000, num_temporal_edges=20_000, temporal_pattern="self_loop",
    ),
    *_cases_for(
        "sensor_net_large", num_nodes=120_000, feature_dim=32,
        num_spatial_edges=3_600_000, num_temporal_edges=120_000, temporal_pattern="self_loop",
    ),
]

# Section 4.3-style edge-case battery, plus this kernel's own: varying
# node counts, sparse connectivity, degree skew, duplicate edges, absent
# edge sets (first-timestep call), and dim=1.
EDGE_CASES: list[GraphMessagePassingCase] = [
    *_cases_for("tiny", num_nodes=4, feature_dim=8, num_spatial_edges=6, num_temporal_edges=4),
    *_cases_for(
        "single_node", num_nodes=1, feature_dim=16, num_spatial_edges=1, num_temporal_edges=1
    ),
    *_cases_for(
        "isolated_nodes", num_nodes=200, feature_dim=32, num_spatial_edges=20, num_temporal_edges=20
    ),
    *_cases_for(
        "empty_spatial", num_nodes=64, feature_dim=16, num_spatial_edges=0, num_temporal_edges=64,
        spatial_pattern="none", temporal_pattern="self_loop",
    ),
    *_cases_for(
        "empty_temporal", num_nodes=64, feature_dim=16, num_spatial_edges=128, num_temporal_edges=0,
        temporal_pattern="none",
    ),
    *_cases_for(
        "empty_both", num_nodes=32, feature_dim=16, num_spatial_edges=0, num_temporal_edges=0,
        spatial_pattern="none", temporal_pattern="none",
    ),
    *_cases_for(
        "high_degree_hub", num_nodes=500, feature_dim=32, num_spatial_edges=0, num_temporal_edges=0,
        spatial_pattern="star_hub", temporal_pattern="none",
    ),
    *_cases_for(
        "duplicate_edges", num_nodes=10, feature_dim=16, num_spatial_edges=200, num_temporal_edges=10,
        temporal_pattern="self_loop",
    ),
    *_cases_for(
        "dim_eq_1", num_nodes=256, feature_dim=1, num_spatial_edges=1024, num_temporal_edges=256,
        temporal_pattern="self_loop",
    ),
    *_cases_for(
        "npot_dim", num_nodes=257, feature_dim=100, num_spatial_edges=1500, num_temporal_edges=257,
        temporal_pattern="self_loop",
    ),
]

ALL_CASES: list[GraphMessagePassingCase] = [*STANDARD_CASES, *EDGE_CASES]


def _make_edges(
    pattern: str, num_nodes: int, num_edges: int, generator: torch.Generator, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if pattern == "none" or num_nodes == 0:
        empty_i = torch.empty(0, dtype=torch.long, device=device)
        empty_w = torch.empty(0, dtype=torch.float32, device=device)
        return empty_i, empty_i, empty_w
    if pattern == "self_loop":
        idx = torch.arange(num_nodes, dtype=torch.long, device=device)
        weight = torch.rand(num_nodes, dtype=torch.float32, device=device, generator=generator)
        return idx, idx, weight
    if pattern == "star_hub":
        if num_nodes < 2:
            empty_i = torch.empty(0, dtype=torch.long, device=device)
            empty_w = torch.empty(0, dtype=torch.float32, device=device)
            return empty_i, empty_i, empty_w
        src = torch.arange(1, num_nodes, dtype=torch.long, device=device)
        dst = torch.zeros(num_nodes - 1, dtype=torch.long, device=device)
        weight = torch.rand(num_nodes - 1, dtype=torch.float32, device=device, generator=generator)
        return src, dst, weight
    if pattern == "random":
        src = torch.randint(0, num_nodes, (num_edges,), device=device, generator=generator, dtype=torch.long)
        dst = torch.randint(0, num_nodes, (num_edges,), device=device, generator=generator, dtype=torch.long)
        weight = torch.rand(num_edges, dtype=torch.float32, device=device, generator=generator)
        return src, dst, weight
    raise ValueError(f"unknown edge pattern: {pattern!r}")


def make_inputs(
    case: GraphMessagePassingCase, device: str = "cuda", seed: int = 0
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Returns `(x_curr, x_prev, spatial_src, spatial_dst, spatial_weight,
    temporal_src, temporal_dst, temporal_weight)`.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    x_curr = torch.randn(case.x_shape, dtype=case.dtype, device=device, generator=gen)
    x_prev = torch.randn(case.x_shape, dtype=case.dtype, device=device, generator=gen)
    spatial_src, spatial_dst, spatial_weight = _make_edges(
        case.spatial_pattern, case.num_nodes, case.num_spatial_edges, gen, device
    )
    temporal_src, temporal_dst, temporal_weight = _make_edges(
        case.temporal_pattern, case.num_nodes, case.num_temporal_edges, gen, device
    )
    return x_curr, x_prev, spatial_src, spatial_dst, spatial_weight, temporal_src, temporal_dst, temporal_weight


def _mp_impl(
    x_curr: torch.Tensor,
    x_prev: torch.Tensor,
    spatial_src: torch.Tensor,
    spatial_dst: torch.Tensor,
    spatial_weight: torch.Tensor,
    temporal_src: torch.Tensor,
    temporal_dst: torch.Tensor,
    temporal_weight: torch.Tensor,
    acc_dtype: torch.dtype,
) -> torch.Tensor:
    n, d = x_curr.shape
    out = torch.zeros(n, d, dtype=acc_dtype, device=x_curr.device)
    if spatial_src.numel() > 0:
        msg = x_curr.to(acc_dtype).index_select(0, spatial_src) * spatial_weight.to(acc_dtype).unsqueeze(-1)
        out.index_add_(0, spatial_dst, msg)
    if temporal_src.numel() > 0:
        msg_t = x_prev.to(acc_dtype).index_select(0, temporal_src) * temporal_weight.to(acc_dtype).unsqueeze(-1)
        out.index_add_(0, temporal_dst, msg_t)
    return out


def eager_spatiotemporal_mp(
    x_curr: torch.Tensor,
    x_prev: torch.Tensor,
    spatial_src: torch.Tensor,
    spatial_dst: torch.Tensor,
    spatial_weight: torch.Tensor,
    temporal_src: torch.Tensor,
    temporal_dst: torch.Tensor,
    temporal_weight: torch.Tensor,
) -> torch.Tensor:
    """PyTorch eager reference, built directly on `index_add_` (the same
    scatter-add primitive PyTorch Geometric's `MessagePassing` uses under
    the hood for `aggr="add"`).
    """
    out = _mp_impl(
        x_curr, x_prev, spatial_src, spatial_dst, spatial_weight,
        temporal_src, temporal_dst, temporal_weight, torch.float32,
    )
    return out.to(x_curr.dtype)


_compiled_cache: dict[str, Callable] = {}


def compiled_spatiotemporal_mp(
    x_curr: torch.Tensor,
    x_prev: torch.Tensor,
    spatial_src: torch.Tensor,
    spatial_dst: torch.Tensor,
    spatial_weight: torch.Tensor,
    temporal_src: torch.Tensor,
    temporal_dst: torch.Tensor,
    temporal_weight: torch.Tensor,
) -> torch.Tensor:
    fn = _compiled_cache.get("fn")
    if fn is None:
        fn = torch.compile(eager_spatiotemporal_mp, mode="max-autotune", fullgraph=True)
        _compiled_cache["fn"] = fn
    return fn(
        x_curr, x_prev, spatial_src, spatial_dst, spatial_weight, temporal_src, temporal_dst, temporal_weight
    )


def reference_fp64(
    x_curr: torch.Tensor,
    x_prev: torch.Tensor,
    spatial_src: torch.Tensor,
    spatial_dst: torch.Tensor,
    spatial_weight: torch.Tensor,
    temporal_src: torch.Tensor,
    temporal_dst: torch.Tensor,
    temporal_weight: torch.Tensor,
) -> torch.Tensor:
    """fp64 ground truth for correctness tests only."""
    return _mp_impl(
        x_curr, x_prev, spatial_src, spatial_dst, spatial_weight,
        temporal_src, temporal_dst, temporal_weight, torch.float64,
    )
