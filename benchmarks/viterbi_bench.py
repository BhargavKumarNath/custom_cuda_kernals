"""Hardware-level benchmark harness for Kernel 11 (Parallel Viterbi
Algorithm). Same methodology as benchmarks/graph_message_passing_bench.py
(Section 5: CUDA events, L2 flush, warm-up/measurement counts).

Only `eager` (the per-timestep Python loop) and `cuda_kernel` (the
persistent single-launch kernel) are compared — `compiled` is
deliberately omitted. `torch.compile(fullgraph=True)` over
`eager_viterbi` traces and compiles a *fully unrolled* `seq_len`-step
graph; at this benchmark's longer sequence lengths that trace/compile
cost is itself impractically large (multi-minute), which is itself a
relevant empirical data point reinforcing exactly why a persistent,
internally-looping kernel is the right architecture here rather than
leaning on the compiler to fuse away per-step launches.

This op is latency-bound (an inherently sequential O(T) recursion, not a
bandwidth- or compute-bound bulk op), so — matching Section 3.11's own
"speedup vs. per-timestep loop" framing — only median latency and
derived speedup are reported, not bandwidth/TFLOPS.

Run directly: `python benchmarks/viterbi_bench.py`. Writes results to
benchmarks/viterbi_results.csv (git-ignored).
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

from baselines.viterbi import ViterbiCase, eager_viterbi, make_inputs

WARMUP_ITERS = 5
MEASURE_ITERS = 30
_L2_FLUSH_BYTES = 256 * 1024 * 1024

_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _cases_for(name: str, **kwargs) -> list[ViterbiCase]:
    return [ViterbiCase(f"{name}_{dt}".replace("torch.", ""), dtype=dt, **kwargs) for dt in _DTYPES]


BENCH_CASES: list[ViterbiCase] = [
    # Sequence-length sweep at fixed batch/state count (Section 3.11's
    # stated ">=512" success-criteria length sits in the middle).
    *_cases_for("tsweep_64", batch=64, seq_len=64, num_states=16),
    *_cases_for("tsweep_256", batch=64, seq_len=256, num_states=16),
    *_cases_for("tsweep_512", batch=64, seq_len=512, num_states=16),
    *_cases_for("tsweep_1024", batch=64, seq_len=1024, num_states=16),
    *_cases_for("tsweep_2048", batch=64, seq_len=2048, num_states=16),
    *_cases_for("tsweep_8192", batch=64, seq_len=8192, num_states=16),
    # Batch sweep at fixed seq_len/state count.
    *_cases_for("bsweep_1", batch=1, seq_len=512, num_states=16),
    *_cases_for("bsweep_16", batch=16, seq_len=512, num_states=16),
    *_cases_for("bsweep_64", batch=64, seq_len=512, num_states=16),
    *_cases_for("bsweep_256", batch=256, seq_len=512, num_states=16),
    # State-count sweep at fixed batch/seq_len.
    *_cases_for("ssweep_8", batch=64, seq_len=512, num_states=8),
    *_cases_for("ssweep_16", batch=64, seq_len=512, num_states=16),
    *_cases_for("ssweep_32", batch=64, seq_len=512, num_states=32),
    *_cases_for("ssweep_64", batch=64, seq_len=512, num_states=64),
    *_cases_for("ssweep_128", batch=64, seq_len=512, num_states=128),
]


@dataclasses.dataclass
class BenchResult:
    impl: str
    case_name: str
    dtype: str
    batch: int
    seq_len: int
    num_states: int
    median_ms: float
    iqr_ms: float


def _l2_flush_buffer(device: str = "cuda") -> torch.Tensor:
    return torch.empty(_L2_FLUSH_BYTES // 4, dtype=torch.float32, device=device)


def _flush_l2(buf: torch.Tensor) -> None:
    buf.fill_(1.0)


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


def _stats(times_ms: list[float]) -> tuple[float, float]:
    s = sorted(times_ms)
    median = statistics.median(s)
    q1 = s[int(0.25 * len(s))]
    q3 = s[int(0.75 * len(s))]
    return median, q3 - q1


def run_benchmarks(device: str = "cuda") -> list[BenchResult]:
    results: list[BenchResult] = []
    flush_buf = _l2_flush_buffer(device)

    try:
        from custom_cuda.kernels.viterbi import viterbi_decode

        have_kernel = True
    except ImportError:
        have_kernel = False
        print("custom_cuda._native not built — skipping cuda_kernel benchmarks.\n")

    for case in BENCH_CASES:
        log_emission, log_trans, log_pi = make_inputs(case, device=device)

        impls: dict[str, Callable[[], object]] = {
            "eager": lambda le=log_emission, lt=log_trans, lp=log_pi: eager_viterbi(le, lt, lp),
        }
        if have_kernel:
            impls["cuda_kernel"] = lambda le=log_emission, lt=log_trans, lp=log_pi: viterbi_decode(le, lt, lp)

        for impl_name, fn in impls.items():
            times = _time_call(fn, flush_buf)
            median, iqr = _stats(times)
            results.append(
                BenchResult(impl_name, case.name, str(case.dtype), case.batch, case.seq_len, case.num_states,
                             median, iqr)
            )
            print(f"{impl_name:12s} {case.name:18s} median={median:10.4f}ms  iqr={iqr:8.4f}ms")

    return results


def main() -> None:
    results = run_benchmarks()
    out_path = Path(__file__).resolve().parent / "viterbi_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dataclasses.fields(BenchResult)])
        writer.writeheader()
        for r in results:
            writer.writerow(dataclasses.asdict(r))
    print(f"\nWrote {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
