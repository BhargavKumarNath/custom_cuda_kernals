"""Hardware-level benchmark harness for Kernel 10 (Spatiotemporal Graph
Message Passing). Same methodology as benchmarks/token_permute_bench.py
(Section 5: CUDA events, L2 flush, warm-up/measurement counts). This op
is bandwidth-bound (one FMA per feature element read — arithmetic
intensity ~1, same as Kernel 1-3/6/7), so bandwidth (not TFLOPS) is the
reported throughput metric, matching those kernels' convention rather
than Kernel 5/9's compute-bound TFLOPS convention.

`cuda_kernel`'s timed call includes the COO->CSR conversion
(`custom_cuda.kernels.graph_message_passing._to_csr`) — necessary,
real overhead of this implementation, not hidden setup (same choice
Kernel 9 made by including norm precomputation in its timed call).

Run directly: `python benchmarks/graph_message_passing_bench.py`. Writes
results to benchmarks/graph_message_passing_results.csv (git-ignored).
"""

from __future__ import annotations

import csv
import dataclasses
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines.graph_message_passing import (
    GraphMessagePassingCase,
    compiled_spatiotemporal_mp,
    eager_spatiotemporal_mp,
    make_inputs,
)

WARMUP_ITERS = 10
MEASURE_ITERS = 100
_L2_FLUSH_BYTES = 256 * 1024 * 1024
PEAK_BANDWIDTH_GB_S = 256.0  # RTX 4070 Laptop GPU — see rmsnorm_residual_bench.py

_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _cases_for(name: str, **kwargs) -> list[GraphMessagePassingCase]:
    return [
        GraphMessagePassingCase(f"{name}_{dt}".replace("torch.", ""), dtype=dt, **kwargs)
        for dt in _DTYPES
    ]


BENCH_CASES: list[GraphMessagePassingCase] = [
    # Node-count sweep at fixed avg spatial degree (~20) and feature dim.
    *_cases_for(
        "nsweep_2000", num_nodes=2_000, feature_dim=64,
        num_spatial_edges=40_000, num_temporal_edges=2_000, temporal_pattern="self_loop",
    ),
    *_cases_for(
        "nsweep_20000", num_nodes=20_000, feature_dim=64,
        num_spatial_edges=400_000, num_temporal_edges=20_000, temporal_pattern="self_loop",
    ),
    *_cases_for(
        "nsweep_100000", num_nodes=100_000, feature_dim=64,
        num_spatial_edges=2_000_000, num_temporal_edges=100_000, temporal_pattern="self_loop",
    ),
    *_cases_for(
        "nsweep_500000", num_nodes=500_000, feature_dim=64,
        num_spatial_edges=10_000_000, num_temporal_edges=500_000, temporal_pattern="self_loop",
    ),
    # Avg-degree sweep at fixed node count (Section 3.10's stated
    # "average degree 10-50" range, plus a sparser point).
    *_cases_for(
        "dsweep_5", num_nodes=50_000, feature_dim=64,
        num_spatial_edges=250_000, num_temporal_edges=50_000, temporal_pattern="self_loop",
    ),
    *_cases_for(
        "dsweep_10", num_nodes=50_000, feature_dim=64,
        num_spatial_edges=500_000, num_temporal_edges=50_000, temporal_pattern="self_loop",
    ),
    *_cases_for(
        "dsweep_20", num_nodes=50_000, feature_dim=64,
        num_spatial_edges=1_000_000, num_temporal_edges=50_000, temporal_pattern="self_loop",
    ),
    *_cases_for(
        "dsweep_50", num_nodes=50_000, feature_dim=64,
        num_spatial_edges=2_500_000, num_temporal_edges=50_000, temporal_pattern="self_loop",
    ),
    # 100K+-node, avg-degree-10-50 shape matching Section 3.10's stated
    # success-criteria scale directly.
    *_cases_for(
        "sensor_net_realistic", num_nodes=120_000, feature_dim=32,
        num_spatial_edges=3_600_000, num_temporal_edges=120_000, temporal_pattern="self_loop",
    ),
]


@dataclasses.dataclass
class BenchResult:
    impl: str
    case_name: str
    dtype: str
    num_nodes: int
    feature_dim: int
    num_spatial_edges: int
    num_temporal_edges: int
    median_ms: float
    iqr_ms: float
    bandwidth_gb_s: float
    bandwidth_pct_peak: float


def _l2_flush_buffer(device: str = "cuda") -> torch.Tensor:
    return torch.empty(_L2_FLUSH_BYTES // 4, dtype=torch.float32, device=device)


def _flush_l2(buf: torch.Tensor) -> None:
    buf.fill_(1.0)


def _ideal_bytes_moved(case: GraphMessagePassingCase) -> int:
    """gather reads (feature vector + CSR col/weight per edge) for both
    edge sets, plus one output write per node.
    """
    elem_size = torch.tensor([], dtype=case.dtype).element_size()
    num_edges = case.num_spatial_edges + (
        case.num_nodes if case.temporal_pattern == "self_loop" else case.num_temporal_edges
    )
    edge_bytes = num_edges * (case.feature_dim * elem_size + 12)  # +8 (int64 col) +4 (fp32 weight)
    write_bytes = case.num_nodes * case.feature_dim * elem_size
    return edge_bytes + write_bytes


def _time_call(fn: Callable[[], object], flush_buf: torch.Tensor) -> list[float]:
    for _ in range(WARMUP_ITERS):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms: list[float] = []
    for _ in range(MEASURE_ITERS):
        _flush_l2(flush_buf)
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return times_ms


def _stats(times_ms: list[float], case: GraphMessagePassingCase) -> tuple[float, float, float, float]:
    s = sorted(times_ms)
    median = statistics.median(s)
    q1 = s[int(0.25 * len(s))]
    q3 = s[int(0.75 * len(s))]
    iqr = q3 - q1
    seconds = median / 1000.0
    bandwidth = _ideal_bytes_moved(case) / seconds / 1e9
    bandwidth_pct = 100.0 * bandwidth / PEAK_BANDWIDTH_GB_S
    return median, iqr, bandwidth, bandwidth_pct


def run_benchmarks(device: str = "cuda") -> list[BenchResult]:
    results: list[BenchResult] = []
    flush_buf = _l2_flush_buffer(device)

    try:
        from custom_cuda.kernels.graph_message_passing import spatiotemporal_message_passing

        have_kernel = True
    except ImportError:
        have_kernel = False
        print("custom_cuda._native not built — skipping cuda_kernel benchmarks.\n")

    for case in BENCH_CASES:
        inputs = make_inputs(case, device=device)

        impls: dict[str, Callable[[], object]] = {
            "eager": lambda inputs=inputs: eager_spatiotemporal_mp(*inputs),
            "compiled": lambda inputs=inputs: compiled_spatiotemporal_mp(*inputs),
        }
        if have_kernel:
            impls["cuda_kernel"] = lambda inputs=inputs: spatiotemporal_message_passing(*inputs)

        for impl_name, fn in impls.items():
            times = _time_call(fn, flush_buf)
            median, iqr, bw, bw_pct = _stats(times, case)
            results.append(
                BenchResult(
                    impl_name, case.name, str(case.dtype), case.num_nodes, case.feature_dim,
                    case.num_spatial_edges, case.num_temporal_edges, median, iqr, bw, bw_pct,
                )
            )
            print(
                f"{impl_name:12s} {case.name:22s} median={median:9.4f}ms  iqr={iqr:8.4f}ms  "
                f"bw={bw:8.2f} GB/s ({bw_pct:5.1f}% peak)"
            )

    return results


def main() -> None:
    results = run_benchmarks()
    out_path = Path(__file__).resolve().parent / "graph_message_passing_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dataclasses.fields(BenchResult)])
        writer.writeheader()
        for r in results:
            writer.writerow(dataclasses.asdict(r))
    print(f"\nWrote {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
