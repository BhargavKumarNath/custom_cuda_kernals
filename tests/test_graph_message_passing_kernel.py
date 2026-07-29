"""Correctness tests for Kernel 10's CUDA implementation
(csrc/kernels/graph_message_passing.cu, via
custom_cuda.kernels.graph_message_passing) against the PyTorch eager
baseline. Mirrors tests/test_pairwise_distance_kernel.py.
"""

from __future__ import annotations

import pytest
import torch
from baselines.graph_message_passing import ALL_CASES, GraphMessagePassingCase, eager_spatiotemporal_mp, make_inputs
from tests.numerics import DEFAULT_BF16_OUTLIER_FRACTION, assert_close_with_outliers

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA device required for Graph Message Passing kernel tests", allow_module_level=True)

pytest.importorskip(
    "custom_cuda._native", reason="Rust extension not built — run `maturin develop` first"
)
from custom_cuda.kernels.graph_message_passing import spatiotemporal_message_passing  # noqa: E402

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
def test_kernel_matches_eager(case: GraphMessagePassingCase):
    inputs = make_inputs(case)
    x_curr = inputs[0]
    out_eager = eager_spatiotemporal_mp(*inputs)
    out_kernel = spatiotemporal_message_passing(*inputs)

    assert out_kernel.shape == (case.num_nodes, case.feature_dim)
    assert out_kernel.dtype == x_curr.dtype

    if case.num_nodes == 0:
        return

    tol = TOLERANCES[case.dtype]
    outliers = OUTLIER_FRACTION[case.dtype]
    assert_close_with_outliers(
        out_kernel.to(torch.float32), out_eager.to(torch.float32), max_outlier_fraction=outliers, **tol
    )


def test_kernel_empty_edges_gives_zero_output():
    case = GraphMessagePassingCase(
        "explicit_empty", num_nodes=32, feature_dim=16, num_spatial_edges=0, num_temporal_edges=0,
        dtype=torch.float32, spatial_pattern="none", temporal_pattern="none",
    )
    out = spatiotemporal_message_passing(*make_inputs(case))
    torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)


def test_kernel_high_degree_hub_aggregates_all_neighbors():
    case = GraphMessagePassingCase(
        "hub_check", num_nodes=500, feature_dim=32, num_spatial_edges=0, num_temporal_edges=0,
        dtype=torch.float32, spatial_pattern="star_hub", temporal_pattern="none",
    )
    inputs = make_inputs(case)
    x_curr, x_prev, s_src, s_dst, s_w, t_src, t_dst, t_w = inputs
    out = spatiotemporal_message_passing(*inputs)

    expected_hub = (x_curr[s_src].to(torch.float32) * s_w.unsqueeze(-1)).sum(dim=0)
    torch.testing.assert_close(out[0], expected_hub, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out[1:], torch.zeros(499, 32, device="cuda"), rtol=0, atol=0)


def test_kernel_duplicate_edges_sum_not_overwrite():
    device = "cuda"
    x_curr = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    x_prev = torch.zeros(2, 2, device=device)
    spatial_src = torch.tensor([0, 0], dtype=torch.long, device=device)
    spatial_dst = torch.tensor([1, 1], dtype=torch.long, device=device)
    spatial_weight = torch.tensor([1.0, 1.0], device=device)
    empty_i = torch.empty(0, dtype=torch.long, device=device)
    empty_w = torch.empty(0, device=device)

    out = spatiotemporal_message_passing(
        x_curr, x_prev, spatial_src, spatial_dst, spatial_weight, empty_i, empty_i, empty_w
    )
    torch.testing.assert_close(out[1], torch.tensor([2.0, 4.0], device=device), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(out[0], torch.zeros(2, device=device), rtol=0, atol=0)


def test_kernel_rejects_cpu_tensor():
    device = "cuda"
    x_curr = torch.randn(4, 8)
    x_prev = torch.randn(4, 8)
    empty_i = torch.empty(0, dtype=torch.long, device=device)
    empty_w = torch.empty(0, device=device)
    with pytest.raises(ValueError):
        spatiotemporal_message_passing(x_curr, x_prev, empty_i, empty_i, empty_w, empty_i, empty_i, empty_w)


def test_kernel_rejects_dtype_mismatch():
    device = "cuda"
    x_curr = torch.randn(4, 8, device=device, dtype=torch.float32)
    x_prev = torch.randn(4, 8, device=device, dtype=torch.float16)
    empty_i = torch.empty(0, dtype=torch.long, device=device)
    empty_w = torch.empty(0, device=device)
    with pytest.raises(ValueError):
        spatiotemporal_message_passing(x_curr, x_prev, empty_i, empty_i, empty_w, empty_i, empty_i, empty_w)
