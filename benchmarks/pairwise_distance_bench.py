"""Hardware-level benchmark harness for Kernel 9 (Block Pairwise Distance
Matrix). Same methodology as benchmarks/matmul_add_bias_bench.py (Section
5: CUDA events, L2 flush, warm-up/measurement counts), plus TFLOPS — this
kernel reuses Kernel 5's tiled-GEMM structure and is compute-bound at
realistic sizes the same way.

Three implementations compared: `eager` (formula-based unfused PyTorch —
this kernel's actual comparison point), `cdist` (`torch.cdist(...) ** 2` —
project_plan.md Section 3.9's stated "beat cdist" success criterion),
`compiled` (torch.compile over the eager formula), and `cuda_kernel` (this
library's hand-written tiled kernel).

Run directly: `python benchmarks/pairwise_distance_bench.py`. Writes
results to benchmarks/pairwise_distance_results.csv (git-ignored).
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

from baselines.pairwise_distance import (
    PairwiseDistanceCase,
    cdist_distance_sq,
    compiled_pairwise_distance_sq,
    eager_pairwise_distance_sq,
    make_inputs,
)

WARMUP_ITERS = 10
MEASURE_ITERS = 100
_L2_FLUSH_BYTES = 256 * 1024 * 1024
PEAK_BANDWIDTH_GB_S = 256.0  # RTX 4070 Laptop GPU — see rmsnorm_residual_bench.py

_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _cases_for(name: str, m: int, n: int, dim: int) -> list[PairwiseDistanceCase]:
    return [
        PairwiseDistanceCase(f"{name}_{dt}".replace("torch.", ""), m, n, dim, dt) for dt in _DTYPES
    ]


BENCH_CASES: list[PairwiseDistanceCase] = [
    *_cases_for("small", m=256, n=256, dim=128),
    *_cases_for("medium", m=1024, n=1024, dim=384),
    *_cases_for("large", m=4096, n=4096, dim=256),
    *_cases_for("xlarge", m=8192, n=8192, dim=128),
    # Nearest-neighbor-search-style shapes: a modest query batch against a
    # much larger reference pool (and its transpose).
    *_cases_for("tall_skinny", m=4096, n=32, dim=64),
    *_cases_for("short_wide", m=32, n=4096, dim=64),
    # Embedding-dim sweep at fixed M, N (typical embedding sizes).
    *_cases_for("dimsweep_128", m=2048, n=2048, dim=128),
    *_cases_for("dimsweep_384", m=2048, n=2048, dim=384),
    *_cases_for("dimsweep_768", m=2048, n=2048, dim=768),
    *_cases_for("dimsweep_1536", m=2048, n=2048, dim=1536),
]


@dataclasses.dataclass
class BenchResult:
    impl: str
    case_name: str
    dtype: str
    m: int
    n: int
    dim: int
    median_ms: float
    iqr_ms: float
    bandwidth_gb_s: float
    tflops: float


def _l2_flush_buffer(device: str = "cuda") -> torch.Tensor:
    return torch.empty(_L2_FLUSH_BYTES // 4, dtype=torch.float32, device=device)


def _flush_l2(buf: torch.Tensor) -> None:
    buf.fill_(1.0)


def _ideal_bytes_moved(case: PairwiseDistanceCase) -> int:
    """read a, read b (input dtype), write dist_sq (always float32)."""
    elem_size = torch.tensor([], dtype=case.dtype).element_size()
    out_elem_size = 4
    return (case.m * case.dim + case.n * case.dim) * elem_size + (case.m * case.n) * out_elem_size


def _flop_count(case: PairwiseDistanceCase) -> int:
    return 2 * case.m * case.n * case.dim  # dominant dot-product term


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


def _stats(times_ms: list[float], case: PairwiseDistanceCase) -> tuple[float, float, float, float]:
    s = sorted(times_ms)
    median = statistics.median(s)
    q1 = s[int(0.25 * len(s))]
    q3 = s[int(0.75 * len(s))]
    iqr = q3 - q1
    seconds = median / 1000.0
    bandwidth = _ideal_bytes_moved(case) / seconds / 1e9
    tflops = _flop_count(case) / seconds / 1e12
    return median, iqr, bandwidth, tflops


def run_benchmarks(device: str = "cuda") -> list[BenchResult]:
    results: list[BenchResult] = []
    flush_buf = _l2_flush_buffer(device)

    try:
        from custom_cuda.kernels.pairwise_distance import pairwise_distance_sq

        have_kernel = True
    except ImportError:
        have_kernel = False
        print("custom_cuda._native not built — skipping cuda_kernel benchmarks.\n")

    for case in BENCH_CASES:
        a, b = make_inputs(case, device=device)

        impls: dict[str, Callable[[], object]] = {
            "eager": lambda a=a, b=b: eager_pairwise_distance_sq(a, b),
            "cdist": lambda a=a, b=b: cdist_distance_sq(a, b),
            "compiled": lambda a=a, b=b: compiled_pairwise_distance_sq(a, b),
        }
        if have_kernel:
            impls["cuda_kernel"] = lambda a=a, b=b: pairwise_distance_sq(a, b)

        for impl_name, fn in impls.items():
            times = _time_call(fn, flush_buf)
            median, iqr, bw, tflops = _stats(times, case)
            results.append(
                BenchResult(impl_name, case.name, str(case.dtype), case.m, case.n, case.dim, median, iqr, bw, tflops)
            )
            print(
                f"{impl_name:12s} {case.name:18s} median={median:9.4f}ms  iqr={iqr:8.4f}ms  "
                f"bw={bw:8.2f} GB/s  tflops={tflops:7.3f}"
            )

    return results


def main() -> None:
    results = run_benchmarks()
    out_path = Path(__file__).resolve().parent / "pairwise_distance_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dataclasses.fields(BenchResult)])
        writer.writeheader()
        for r in results:
            writer.writerow(dataclasses.asdict(r))
    print(f"\nWrote {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
