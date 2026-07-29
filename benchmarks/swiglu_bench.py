"""Hardware-level benchmark harness for Kernel 2 (Fused SwiGLU Gated
Activation). Same methodology as benchmarks/rmsnorm_residual_bench.py — see
that file's docstring for the full Section 5 methodology notes.

Run directly: `python benchmarks/swiglu_bench.py`. Writes results to
benchmarks/swiglu_results.csv (git-ignored).
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

from baselines.swiglu import SwiGLUCase, compiled_swiglu, eager_swiglu, make_inputs

WARMUP_ITERS = 10
MEASURE_ITERS = 100
_L2_FLUSH_BYTES = 256 * 1024 * 1024
PEAK_BANDWIDTH_GB_S = 256.0  # RTX 4070 Laptop GPU — see rmsnorm_residual_bench.py

_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _cases_for(name: str, batch: int, seq_len: int, intermediate_dim: int) -> list[SwiGLUCase]:
    return [
        SwiGLUCase(f"{name}_{dt}".replace("torch.", ""), batch, seq_len, intermediate_dim, dt)
        for dt in _DTYPES
    ]


BENCH_CASES: list[SwiGLUCase] = [
    *_cases_for("bs4_seq128_d3072", batch=4, seq_len=128, intermediate_dim=3072),
    *_cases_for("bs2_seq2048_d11008", batch=2, seq_len=2048, intermediate_dim=11008),
    *_cases_for("bs1_seq4096_d11008", batch=1, seq_len=4096, intermediate_dim=11008),
    *_cases_for("seqsweep_128", batch=1, seq_len=128, intermediate_dim=11008),
    *_cases_for("seqsweep_512", batch=1, seq_len=512, intermediate_dim=11008),
    *_cases_for("seqsweep_2048", batch=1, seq_len=2048, intermediate_dim=11008),
    *_cases_for("seqsweep_8192", batch=1, seq_len=8192, intermediate_dim=11008),
    *_cases_for("batchsweep_1", batch=1, seq_len=1024, intermediate_dim=11008),
    *_cases_for("batchsweep_4", batch=4, seq_len=1024, intermediate_dim=11008),
    *_cases_for("batchsweep_16", batch=16, seq_len=1024, intermediate_dim=11008),
]


@dataclasses.dataclass
class BenchResult:
    impl: str
    case_name: str
    dtype: str
    rows: int
    cols: int
    median_ms: float
    iqr_ms: float
    bandwidth_gb_s: float
    bandwidth_pct_peak: float
    tflops: float


def _l2_flush_buffer(device: str = "cuda") -> torch.Tensor:
    return torch.empty(_L2_FLUSH_BYTES // 4, dtype=torch.float32, device=device)


def _flush_l2(buf: torch.Tensor) -> None:
    buf.fill_(1.0)


def _ideal_bytes_moved(rows: int, cols: int, dtype: torch.dtype) -> int:
    """read gate, read up, write y — SwiGLU has no reduction/redundant read,
    so this is both the ideal *and* the actual traffic for a correct
    single-pass kernel.
    """
    elem_size = torch.tensor([], dtype=dtype).element_size()
    return rows * cols * elem_size * 3


def _flop_count(rows: int, cols: int) -> int:
    """Approximate FLOPs/element: exp, add(1+.), div, mul(silu), mul(*up)."""
    return rows * cols * 5


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


def _stats(times_ms: list[float], rows: int, cols: int, dtype: torch.dtype) -> tuple[float, float, float, float, float]:
    s = sorted(times_ms)
    median = statistics.median(s)
    q1 = s[int(0.25 * len(s))]
    q3 = s[int(0.75 * len(s))]
    iqr = q3 - q1
    seconds = median / 1000.0
    bandwidth = _ideal_bytes_moved(rows, cols, dtype) / seconds / 1e9
    bandwidth_pct = 100.0 * bandwidth / PEAK_BANDWIDTH_GB_S
    tflops = _flop_count(rows, cols) / seconds / 1e12
    return median, iqr, bandwidth, bandwidth_pct, tflops


def run_benchmarks(device: str = "cuda") -> list[BenchResult]:
    results: list[BenchResult] = []
    flush_buf = _l2_flush_buffer(device)

    try:
        from custom_cuda.kernels.swiglu import swiglu

        have_kernel = True
    except ImportError:
        have_kernel = False
        print("custom_cuda._native not built — skipping cuda_kernel benchmarks.\n")

    for case in BENCH_CASES:
        gate, up = make_inputs(case, device=device)
        rows = case.batch * case.seq_len

        impls: dict[str, Callable[[], object]] = {
            "eager": lambda gate=gate, up=up: eager_swiglu(gate, up),
            "compiled": lambda gate=gate, up=up: compiled_swiglu(gate, up),
        }
        if have_kernel:
            impls["cuda_kernel"] = lambda gate=gate, up=up: swiglu(gate, up)

        for impl_name, fn in impls.items():
            times = _time_call(fn, flush_buf)
            median, iqr, bw, bw_pct, tflops = _stats(times, rows, case.intermediate_dim, case.dtype)
            results.append(
                BenchResult(
                    impl_name, case.name, str(case.dtype), rows, case.intermediate_dim, median, iqr, bw, bw_pct, tflops
                )
            )
            print(
                f"{impl_name:12s} {case.name:22s} median={median:9.4f}ms  iqr={iqr:8.4f}ms  "
                f"bw={bw:8.2f} GB/s ({bw_pct:5.1f}% peak)  tflops={tflops:6.3f}"
            )

    return results


def main() -> None:
    results = run_benchmarks()
    out_path = Path(__file__).resolve().parent / "swiglu_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dataclasses.fields(BenchResult)])
        writer.writeheader()
        for r in results:
            writer.writerow(dataclasses.asdict(r))
    print(f"\nWrote {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
