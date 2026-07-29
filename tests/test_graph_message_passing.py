"""Correctness tests for Kernel 10 (Spatiotemporal Graph Message Passing)
baselines. Mirrors tests/test_pairwise_distance.py's structure.
"""

from __future__ import annotations

import pytest
import torch

from baselines.graph_message_passing import (
    ALL_CASES,
    STANDARD_CASES,
    GraphMessagePassingCase,
    compiled_spatiotemporal_mp,
    eager_spatiotemporal_mp,
    make_inputs,
    reference_fp64,
)
from tests.numerics import DEFAULT_BF16_OUTLIER_FRACTION, assert_close_with_outliers

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA device required for Graph Message Passing baseline tests", allow_module_level=True)

TOLERANCES: dict[torch.dtype, dict[str, float]] = {
    torch.float32: dict(rtol=1e-4, atol=1e-4),
    torch.float16: dict(rtol=1e-2, atol=1e-2),
    torch.bfloat16: dict(rtol=1e-2, atol=2e-2),
}
OUTLIER_FRACTION: dict[torch.dtype, float] = {
    torch.float32: 0.0,
    torch.float16: 0.0,
    torch.bfloat16: DEFAULT_BF16_OUTLIER_FRACTION,
}


def _case_id(case: GraphMessagePassingCase) -> str:
    return case.name


@pytest.mark.parametrize("case", ALL_CASES, ids=_case_id)
def test_eager_matches_fp64_reference(case: GraphMessagePassingCase):
    inputs = make_inputs(case)
    x_curr = inputs[0]
    out = eager_spatiotemporal_mp(*inputs)

    assert out.shape == (case.num_nodes, case.feature_dim)
    assert out.dtype == x_curr.dtype

    if case.num_nodes == 0:
        return

    out_ref = reference_fp64(*inputs)
    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(
        out.to(torch.float32), out_ref.to(torch.float32), max_outlier_fraction=outliers, **tol
    )


def test_eager_empty_edges_gives_zero_output():
    case = GraphMessagePassingCase(
        "explicit_empty", num_nodes=32, feature_dim=16, num_spatial_edges=0, num_temporal_edges=0,
        dtype=torch.float32, spatial_pattern="none", temporal_pattern="none",
    )
    out = eager_spatiotemporal_mp(*make_inputs(case))
    torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)


def test_eager_self_loop_temporal_matches_identity():
    """With `temporal_pattern="self_loop"` and no spatial edges, each
    node's output must equal `temporal_weight[i] * x_prev[i]` exactly.
    """
    case = GraphMessagePassingCase(
        "self_loop_only", num_nodes=64, feature_dim=32, num_spatial_edges=0, num_temporal_edges=64,
        dtype=torch.float32, spatial_pattern="none", temporal_pattern="self_loop",
    )
    x_curr, x_prev, s_src, s_dst, s_w, t_src, t_dst, t_w = make_inputs(case)
    out = eager_spatiotemporal_mp(x_curr, x_prev, s_src, s_dst, s_w, t_src, t_dst, t_w)
    expected = x_prev.to(torch.float32) * t_w.unsqueeze(-1)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_eager_duplicate_edges_sum_not_overwrite():
    """Two edges with the identical (src, dst) pair must both contribute
    — `index_add_` sums, it doesn't overwrite — so the aggregate must
    differ from (and generally exceed in magnitude) a single-edge
    aggregate.
    """
    device = "cuda"
    x_curr = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    x_prev = torch.zeros(2, 2, device=device)
    spatial_src = torch.tensor([0, 0], dtype=torch.long, device=device)
    spatial_dst = torch.tensor([1, 1], dtype=torch.long, device=device)
    spatial_weight = torch.tensor([1.0, 1.0], device=device)
    empty_i = torch.empty(0, dtype=torch.long, device=device)
    empty_w = torch.empty(0, device=device)

    out = eager_spatiotemporal_mp(x_curr, x_prev, spatial_src, spatial_dst, spatial_weight, empty_i, empty_i, empty_w)
    torch.testing.assert_close(out[1], torch.tensor([2.0, 4.0], device=device), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(out[0], torch.zeros(2, device=device), rtol=0, atol=0)


def test_eager_high_degree_hub_aggregates_all_neighbors():
    """`star_hub` pattern: node 0 receives one edge from every other
    node, so its aggregate must equal the (weighted) sum over all of
    them, while every other node's output stays zero (no incoming
    edges).
    """
    case = GraphMessagePassingCase(
        "hub_check", num_nodes=50, feature_dim=8, num_spatial_edges=0, num_temporal_edges=0,
        dtype=torch.float32, spatial_pattern="star_hub", temporal_pattern="none",
    )
    x_curr, x_prev, s_src, s_dst, s_w, t_src, t_dst, t_w = make_inputs(case)
    out = eager_spatiotemporal_mp(x_curr, x_prev, s_src, s_dst, s_w, t_src, t_dst, t_w)

    expected_hub = (x_curr[s_src].to(torch.float32) * s_w.unsqueeze(-1)).sum(dim=0)
    torch.testing.assert_close(out[0], expected_hub, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out[1:], torch.zeros(49, 8, device="cuda"), rtol=0, atol=0)


@pytest.mark.slow
@pytest.mark.parametrize("case", STANDARD_CASES, ids=_case_id)
def test_compiled_matches_eager(case: GraphMessagePassingCase):
    inputs = make_inputs(case)
    out_eager = eager_spatiotemporal_mp(*inputs)
    out_compiled = compiled_spatiotemporal_mp(*inputs)

    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(out_compiled, out_eager, max_outlier_fraction=outliers, **tol)
