"""Hardware-level benchmark harness for Kernel 12 (FP8 Dynamic
Quantization & Casting). Same methodology as
benchmarks/pairwise_distance_bench.py (Section 5: CUDA events, L2
flush, warm-up/measurement counts).

Bandwidth is computed against the *theoretical* single-pass minimum —
one read of `x` plus one write of `x_fp8` — for both `eager` and
`cuda_kernel`, so the reported bandwidth-vs-peak percentage directly
reflects how close each implementation gets to that ideal (project_plan.md
Section 3.12's "≥50% memory-traffic reduction vs. separate amax + cast
kernels" framing: eager's actual 2-pass traffic is close to *double* this
ideal — a full extra read of `x` — while `cuda_kernel`'s block-wise path
re-reads `x` from likely-still-L2-resident tile data rather than a
separate global buffer).

Run directly: `python benchmarks/fp8_quant_bench.py`. Writes results to
benchmarks/fp8_quant_results.csv (git-ignored).
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

from baselines.fp8_quant import Fp8QuantCase, eager_fp8_quant, make_inputs

WARMUP_ITERS = 10
MEASURE_ITERS = 100
_L2_FLUSH_BYTES = 256 * 1024 * 1024
PEAK_BANDWIDTH_GB_S = 256.0  # RTX 4070 Laptop GPU — see rmsnorm_residual_bench.py

_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_FORMATS = ("e4m3", "e5m2")
_GRANULARITIES = ("tensor", "block")


def _cases_for(name: str, m: int, n: int) -> list[Fp8QuantCase]:
    return [
        Fp8QuantCase(f"{name}_{dt}".replace("torch.", "") + f"_{fmt}_{gran}", m, n, dt, fmt, gran)
        for dt in _DTYPES
        for fmt in _FORMATS
        for gran in _GRANULARITIES
    ]


BENCH_CASES: list[Fp8QuantCase] = [
    *_cases_for("small", m=256, n=256),
    *_cases_for("medium", m=1024, n=1024),
    *_cases_for("large", m=4096, n=4096),
    *_cases_for("xlarge", m=8192, n=8192),
]


@dataclasses.dataclass
class BenchResult:
    impl: str
    case_name: str
    dtype: str
    fp8_format: str
    granularity: str
    m: int
    n: int
    median_ms: float
    iqr_ms: float
    bandwidth_gb_s: float
    bandwidth_pct_peak: float


def _l2_flush_buffer(device: str = "cuda") -> torch.Tensor:
    return torch.empty(_L2_FLUSH_BYTES // 4, dtype=torch.float32, device=device)


def _flush_l2(buf: torch.Tensor) -> None:
    buf.fill_(1.0)


def _ideal_bytes_moved(case: Fp8QuantCase) -> int:
    """Theoretical single-pass minimum: one read of x, one write of
    x_fp8 (1 byte/element) — the target both eager and cuda_kernel are
    measured against.
    """
    elem_size = torch.tensor([], dtype=case.dtype).element_size()
    return case.m * case.n * elem_size + case.m * case.n * 1


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


def _stats(times_ms: list[float], case: Fp8QuantCase) -> tuple[float, float, float, float]:
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
        from custom_cuda.kernels.fp8_quant import fp8_quant

        have_kernel = True
    except ImportError:
        have_kernel = False
        print("custom_cuda._native not built — skipping cuda_kernel benchmarks.\n")

    for case in BENCH_CASES:
        x = make_inputs(case, device=device)

        impls: dict[str, Callable[[], object]] = {
            "eager": lambda x=x, fmt=case.fp8_format, gran=case.granularity: eager_fp8_quant(x, fmt, gran),
        }
        if have_kernel:
            impls["cuda_kernel"] = lambda x=x, fmt=case.fp8_format, gran=case.granularity: fp8_quant(x, fmt, gran)

        for impl_name, fn in impls.items():
            times = _time_call(fn, flush_buf)
            median, iqr, bw, bw_pct = _stats(times, case)
            results.append(
                BenchResult(
                    impl_name, case.name, str(case.dtype), case.fp8_format, case.granularity, case.m, case.n,
                    median, iqr, bw, bw_pct,
                )
            )
            print(
                f"{impl_name:12s} {case.name:28s} median={median:9.4f}ms  iqr={iqr:8.4f}ms  "
                f"bw={bw:8.2f} GB/s ({bw_pct:5.1f}% peak)"
            )

    return results


def main() -> None:
    results = run_benchmarks()
    out_path = Path(__file__).resolve().parent / "fp8_quant_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dataclasses.fields(BenchResult)])
        writer.writeheader()
        for r in results:
            writer.writerow(dataclasses.asdict(r))
    print(f"\nWrote {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
