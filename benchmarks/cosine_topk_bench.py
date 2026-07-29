"""Hardware-level benchmark harness for Kernel 8 (Fused Cosine Similarity
+ Top-K). Same methodology as benchmarks/rmsnorm_residual_bench.py
(Section 5: CUDA events, L2 flush, warm-up/measurement counts).

Run directly: `python benchmarks/cosine_topk_bench.py`. Writes results to
benchmarks/cosine_topk_results.csv (git-ignored).
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

from baselines.cosine_topk import CosineTopKCase, compiled_cosine_topk, eager_cosine_topk, make_inputs

WARMUP_ITERS = 10
MEASURE_ITERS = 100
_L2_FLUSH_BYTES = 256 * 1024 * 1024
PEAK_BANDWIDTH_GB_S = 256.0  # RTX 4070 Laptop GPU — see rmsnorm_residual_bench.py

_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _cases_for(name: str, n_queries: int, n_candidates: int, dim: int, k: int) -> list[CosineTopKCase]:
    return [
        CosineTopKCase(f"{name}_{dt}".replace("torch.", ""), n_queries, n_candidates, dim, k, dt)
        for dt in _DTYPES
    ]


BENCH_CASES: list[CosineTopKCase] = [
    *_cases_for("small_pool", n_queries=8, n_candidates=2000, dim=384, k=10),
    *_cases_for("large_pool", n_queries=8, n_candidates=50000, dim=768, k=10),
    *_cases_for("ksweep_k1", n_queries=8, n_candidates=20000, dim=384, k=1),
    *_cases_for("ksweep_k4", n_queries=8, n_candidates=20000, dim=384, k=4),
    *_cases_for("ksweep_k16", n_queries=8, n_candidates=20000, dim=384, k=16),
    *_cases_for("ksweep_k32", n_queries=8, n_candidates=20000, dim=384, k=32),
    # candidate-pool-size sweep at fixed dim/k
    *_cases_for("nsweep_1000", n_queries=8, n_candidates=1000, dim=384, k=10),
    *_cases_for("nsweep_10000", n_queries=8, n_candidates=10000, dim=384, k=10),
    *_cases_for("nsweep_100000", n_queries=8, n_candidates=100000, dim=384, k=10),
    *_cases_for("nsweep_500000", n_queries=8, n_candidates=500000, dim=384, k=10),
]


@dataclasses.dataclass
class BenchResult:
    impl: str
    case_name: str
    dtype: str
    n_queries: int
    n_candidates: int
    dim: int
    k: int
    median_ms: float
    iqr_ms: float
    bandwidth_gb_s: float
    bandwidth_pct_peak: float


def _l2_flush_buffer(device: str = "cuda") -> torch.Tensor:
    return torch.empty(_L2_FLUSH_BYTES // 4, dtype=torch.float32, device=device)


def _flush_l2(buf: torch.Tensor) -> None:
    buf.fill_(1.0)


def _ideal_bytes_moved(case: CosineTopKCase) -> int:
    """Ideal traffic: read all candidates once + read all queries once
    (candidates dominate — queries are tiny and read Q times, not N*Q
    times, in an ideal implementation). Outputs (topk_scores/indices) are
    negligible and omitted, matching the RMSNorm/SwiGLU precedent of
    scoring against an ideal roofline rather than any one implementation's
    actual traffic.
    """
    elem_size = torch.tensor([], dtype=case.dtype).element_size()
    return (case.n_candidates + case.n_queries) * case.dim * elem_size


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


def _stats(times_ms: list[float], case: CosineTopKCase) -> tuple[float, float, float, float]:
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
        from custom_cuda.kernels.cosine_topk import cosine_topk

        have_kernel = True
    except ImportError:
        have_kernel = False
        print("custom_cuda._native not built — skipping cuda_kernel benchmarks.\n")

    for case in BENCH_CASES:
        queries, candidates = make_inputs(case, device=device)

        impls: dict[str, Callable[[], object]] = {
            "eager": lambda q=queries, c=candidates, case=case: eager_cosine_topk(q, c, case.k),
            "compiled": lambda q=queries, c=candidates, case=case: compiled_cosine_topk(q, c, case.k),
        }
        if have_kernel:
            impls["cuda_kernel"] = lambda q=queries, c=candidates, case=case: cosine_topk(q, c, case.k)

        for impl_name, fn in impls.items():
            times = _time_call(fn, flush_buf)
            median, iqr, bw, bw_pct = _stats(times, case)
            results.append(
                BenchResult(
                    impl_name, case.name, str(case.dtype), case.n_queries, case.n_candidates, case.dim,
                    case.k, median, iqr, bw, bw_pct,
                )
            )
            print(
                f"{impl_name:12s} {case.name:16s} median={median:9.4f}ms  iqr={iqr:8.4f}ms  "
                f"bw={bw:8.2f} GB/s ({bw_pct:5.1f}% peak)"
            )

    return results


def main() -> None:
    results = run_benchmarks()
    out_path = Path(__file__).resolve().parent / "cosine_topk_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dataclasses.fields(BenchResult)])
        writer.writeheader()
        for r in results:
            writer.writerow(dataclasses.asdict(r))
    print(f"\nWrote {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
