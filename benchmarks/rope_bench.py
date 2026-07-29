"""Hardware-level benchmark harness for Kernel 3 (Fused RoPE). Same
methodology as benchmarks/rmsnorm_residual_bench.py — see that file's
docstring for the full Section 5 methodology notes.

Run directly: `python benchmarks/rope_bench.py`. Writes results to
benchmarks/rope_results.csv (git-ignored).
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

from baselines.rope import RopeCase, compiled_rope, eager_rope, make_inputs

WARMUP_ITERS = 10
MEASURE_ITERS = 100
_L2_FLUSH_BYTES = 256 * 1024 * 1024
PEAK_BANDWIDTH_GB_S = 256.0  # RTX 4070 Laptop GPU — see rmsnorm_residual_bench.py

_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _cases_for(name: str, batch: int, seq_len: int, n_q: int, n_kv: int, head_dim: int) -> list[RopeCase]:
    return [
        RopeCase(f"{name}_{dt}".replace("torch.", ""), batch, seq_len, n_q, n_kv, head_dim, dt)
        for dt in _DTYPES
    ]


BENCH_CASES: list[RopeCase] = [
    *_cases_for("bs4_seq128_mha", batch=4, seq_len=128, n_q=12, n_kv=12, head_dim=64),
    *_cases_for("bs2_seq2048_gqa", batch=2, seq_len=2048, n_q=32, n_kv=8, head_dim=128),
    *_cases_for("bs1_seq4096_gqa", batch=1, seq_len=4096, n_q=32, n_kv=8, head_dim=128),
    *_cases_for("seqsweep_128", batch=1, seq_len=128, n_q=32, n_kv=8, head_dim=128),
    *_cases_for("seqsweep_512", batch=1, seq_len=512, n_q=32, n_kv=8, head_dim=128),
    *_cases_for("seqsweep_2048", batch=1, seq_len=2048, n_q=32, n_kv=8, head_dim=128),
    *_cases_for("seqsweep_8192", batch=1, seq_len=8192, n_q=32, n_kv=8, head_dim=128),
    *_cases_for("batchsweep_1", batch=1, seq_len=1024, n_q=32, n_kv=8, head_dim=128),
    *_cases_for("batchsweep_4", batch=4, seq_len=1024, n_q=32, n_kv=8, head_dim=128),
    *_cases_for("batchsweep_16", batch=16, seq_len=1024, n_q=32, n_kv=8, head_dim=128),
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


def _total_rows(case: RopeCase) -> int:
    return case.batch * case.seq_len * (case.n_q_heads + case.n_kv_heads)


def _ideal_bytes_moved(case: RopeCase) -> int:
    """read q, read k, write q_out, write k_out (cos/sin tables are tiny
    and effectively L2-resident across rows, omitted).
    """
    elem_size = torch.tensor([], dtype=case.dtype).element_size()
    q_elems = case.batch * case.seq_len * case.n_q_heads * case.head_dim
    k_elems = case.batch * case.seq_len * case.n_kv_heads * case.head_dim
    return (q_elems + k_elems) * elem_size * 2


def _flop_count(case: RopeCase) -> int:
    """4 multiplies + 2 add/sub per rotated pair (2 elements) => 3 flops
    per element, across q and k.
    """
    q_elems = case.batch * case.seq_len * case.n_q_heads * case.head_dim
    k_elems = case.batch * case.seq_len * case.n_kv_heads * case.head_dim
    return (q_elems + k_elems) * 3


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


def _stats(times_ms: list[float], case: RopeCase) -> tuple[float, float, float, float, float]:
    s = sorted(times_ms)
    median = statistics.median(s)
    q1 = s[int(0.25 * len(s))]
    q3 = s[int(0.75 * len(s))]
    iqr = q3 - q1
    seconds = median / 1000.0
    bandwidth = _ideal_bytes_moved(case) / seconds / 1e9
    bandwidth_pct = 100.0 * bandwidth / PEAK_BANDWIDTH_GB_S
    tflops = _flop_count(case) / seconds / 1e12
    return median, iqr, bandwidth, bandwidth_pct, tflops


def run_benchmarks(device: str = "cuda") -> list[BenchResult]:
    results: list[BenchResult] = []
    flush_buf = _l2_flush_buffer(device)

    try:
        from custom_cuda.kernels.rope import rope

        have_kernel = True
    except ImportError:
        have_kernel = False
        print("custom_cuda._native not built — skipping cuda_kernel benchmarks.\n")

    for case in BENCH_CASES:
        q, k, cos, sin = make_inputs(case, device=device)
        rows = _total_rows(case)

        impls: dict[str, Callable[[], object]] = {
            "eager": lambda q=q, k=k, cos=cos, sin=sin: eager_rope(q, k, cos, sin),
            "compiled": lambda q=q, k=k, cos=cos, sin=sin: compiled_rope(q, k, cos, sin),
        }
        if have_kernel:
            impls["cuda_kernel"] = lambda q=q, k=k, cos=cos, sin=sin: rope(q, k, cos, sin)

        for impl_name, fn in impls.items():
            times = _time_call(fn, flush_buf)
            median, iqr, bw, bw_pct, tflops = _stats(times, case)
            results.append(
                BenchResult(
                    impl_name, case.name, str(case.dtype), rows, case.head_dim, median, iqr, bw, bw_pct, tflops
                )
            )
            print(
                f"{impl_name:12s} {case.name:22s} median={median:9.4f}ms  iqr={iqr:8.4f}ms  "
                f"bw={bw:8.2f} GB/s ({bw_pct:5.1f}% peak)  tflops={tflops:6.3f}"
            )

    return results


def main() -> None:
    results = run_benchmarks()
    out_path = Path(__file__).resolve().parent / "rope_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dataclasses.fields(BenchResult)])
        writer.writeheader()
        for r in results:
            writer.writerow(dataclasses.asdict(r))
    print(f"\nWrote {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
