"""Hardware-level benchmark harness for Kernel 1 (Fused RMSNorm + Residual).

Follows project_plan.md Section 5 exactly:
  - `torch.cuda.Event(enable_timing=True)` timing with explicit
    `synchronize()` immediately before `start.record()` and after
    `end.record()`.
  - An L2-cache-exceeding dummy buffer is swept before every measured
    iteration (Section 5.2).
  - >=10 untimed warm-up iterations, >=100 timed measurement iterations
    (Section 5.3).
  - Reports median runtime, IQR, memory bandwidth (GB/s), and TFLOPS
    (Section 5.4).

Run directly: `python benchmarks/rmsnorm_residual_bench.py`. Writes results
to `benchmarks/rmsnorm_residual_results.csv` (git-ignored — regenerate on
demand, this is raw data for scripts/plot_rmsnorm_residual.py, not a
committed artifact).
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

from baselines.rmsnorm_residual import (
    RMSNormResidualCase,
    compiled_rmsnorm_residual,
    eager_rmsnorm_residual,
    make_inputs,
)

WARMUP_ITERS = 10
MEASURE_ITERS = 100

# Comfortably larger than this GPU generation's L2 cache (tens of MB at
# most) to guarantee a full flush between measured iterations.
_L2_FLUSH_BYTES = 256 * 1024 * 1024

# NVIDIA GeForce RTX 4070 Laptop GPU: 128-bit GDDR6 bus @ 8001 MHz
# (`nvidia-smi --query-gpu=clocks.max.memory`), effective data rate = 2x
# clock (DDR) -> 16002 MT/s * 16 bytes/transfer ~= 256 GB/s theoretical peak.
# Update this if benchmarking on different hardware.
PEAK_BANDWIDTH_GB_S = 256.0

# Shape sweep: small/medium/large representative LLM activation shapes,
# plus a sequence-length sweep and a batch-size sweep at fixed hidden_dim
# for Section 6's "scaling across sequence length / batch size" chart.
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _cases_for(name: str, batch: int, seq_len: int, hidden_dim: int) -> list[RMSNormResidualCase]:
    return [
        RMSNormResidualCase(f"{name}_{dt}".replace("torch.", ""), batch, seq_len, hidden_dim, dt)
        for dt in _DTYPES
    ]


BENCH_CASES: list[RMSNormResidualCase] = [
    *_cases_for("bs4_seq128_h768", batch=4, seq_len=128, hidden_dim=768),
    *_cases_for("bs2_seq2048_h4096", batch=2, seq_len=2048, hidden_dim=4096),
    *_cases_for("bs1_seq4096_h4096", batch=1, seq_len=4096, hidden_dim=4096),
    # sequence-length sweep at fixed batch/hidden
    *_cases_for("seqsweep_128", batch=1, seq_len=128, hidden_dim=4096),
    *_cases_for("seqsweep_512", batch=1, seq_len=512, hidden_dim=4096),
    *_cases_for("seqsweep_2048", batch=1, seq_len=2048, hidden_dim=4096),
    *_cases_for("seqsweep_8192", batch=1, seq_len=8192, hidden_dim=4096),
    # batch-size sweep at fixed seq/hidden
    *_cases_for("batchsweep_1", batch=1, seq_len=1024, hidden_dim=4096),
    *_cases_for("batchsweep_4", batch=4, seq_len=1024, hidden_dim=4096),
    *_cases_for("batchsweep_16", batch=16, seq_len=1024, hidden_dim=4096),
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
    n = _L2_FLUSH_BYTES // 4
    return torch.empty(n, dtype=torch.float32, device=device)


def _flush_l2(buf: torch.Tensor) -> None:
    buf.fill_(1.0)


def _ideal_bytes_moved(rows: int, cols: int, dtype: torch.dtype) -> int:
    """Bandwidth denominator: the traffic an *ideal* single-pass fused
    kernel needs (read x once, read residual once, write y once, write
    residual_out once; weight is tiny/L2-resident and omitted). All three
    implementations (eager, torch.compile, CUDA kernel) are scored against
    this same roofline rather than each implementation's own actual
    traffic, so the reported GB/s is directly comparable across them.
    """
    elem_size = torch.tensor([], dtype=dtype).element_size()
    return rows * cols * elem_size * 4


def _flop_count(rows: int, cols: int) -> int:
    """Approximate FLOPs/element: add (x+residual), square, reduction-add,
    multiply by rms_inv, multiply by weight = 5 flops/element.
    """
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
        from custom_cuda.kernels.rmsnorm_residual import rmsnorm_residual

        have_kernel = True
    except ImportError:
        have_kernel = False
        print("custom_cuda._native not built — skipping cuda_kernel benchmarks.\n")

    for case in BENCH_CASES:
        x, residual, weight = make_inputs(case, device=device)
        rows = case.batch * case.seq_len

        impls: dict[str, Callable[[], object]] = {
            "eager": lambda x=x, residual=residual, weight=weight, case=case: eager_rmsnorm_residual(
                x, residual, weight, case.eps
            ),
            "compiled": lambda x=x, residual=residual, weight=weight, case=case: compiled_rmsnorm_residual(
                x, residual, weight, case.eps
            ),
        }
        if have_kernel:
            impls["cuda_kernel"] = (
                lambda x=x, residual=residual, weight=weight, case=case: rmsnorm_residual(
                    x, residual, weight, case.eps
                )
            )

        for impl_name, fn in impls.items():
            times = _time_call(fn, flush_buf)
            median, iqr, bw, bw_pct, tflops = _stats(times, rows, case.hidden_dim, case.dtype)
            results.append(
                BenchResult(
                    impl_name, case.name, str(case.dtype), rows, case.hidden_dim, median, iqr, bw, bw_pct, tflops
                )
            )
            print(
                f"{impl_name:12s} {case.name:22s} median={median:9.4f}ms  iqr={iqr:8.4f}ms  "
                f"bw={bw:8.2f} GB/s ({bw_pct:5.1f}% peak)  tflops={tflops:6.3f}"
            )

    return results


def main() -> None:
    results = run_benchmarks()
    out_path = Path(__file__).resolve().parent / "rmsnorm_residual_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dataclasses.fields(BenchResult)])
        writer.writeheader()
        for r in results:
            writer.writerow(dataclasses.asdict(r))
    print(f"\nWrote {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
