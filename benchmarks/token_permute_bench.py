"""Hardware-level benchmark harness for Kernel 7 (Token Scatter/Gather,
Permute-Unpermute). Same methodology as benchmarks/rmsnorm_residual_bench.py
(Section 5: CUDA events, L2 flush, warm-up/measurement counts).

Benchmarks both operations: `gather` (permute) and `combine` (unpermute +
weighted sum). Run directly: `python benchmarks/token_permute_bench.py`.
Writes results to benchmarks/token_permute_gather_results.csv and
benchmarks/token_permute_combine_results.csv (both git-ignored).
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

from baselines.token_permute import (
    TokenCombineCase,
    TokenGatherCase,
    compiled_token_combine,
    compiled_token_gather,
    eager_token_combine,
    eager_token_gather,
    make_combine_inputs,
    make_gather_inputs,
)

WARMUP_ITERS = 10
MEASURE_ITERS = 100
_L2_FLUSH_BYTES = 256 * 1024 * 1024
PEAK_BANDWIDTH_GB_S = 256.0  # RTX 4070 Laptop GPU rmsnorm_residual_bench.py

_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _gather_cases_for(name: str, n_src: int, n_dst: int, hidden_dim: int) -> list[TokenGatherCase]:
    return [
        TokenGatherCase(f"{name}_{dt}".replace("torch.", ""), n_src, n_dst, hidden_dim, dt)
        for dt in _DTYPES
    ]


def _combine_cases_for(name: str, n_tokens: int, k: int, hidden_dim: int) -> list[TokenCombineCase]:
    return [
        TokenCombineCase(f"{name}_{dt}".replace("torch.", ""), n_tokens, k, hidden_dim, dt)
        for dt in _DTYPES
    ]


GATHER_BENCH_CASES: list[TokenGatherCase] = [
    *_gather_cases_for("mixtral_like", n_src=2048, n_dst=4096, hidden_dim=4096),
    *_gather_cases_for("large_hidden", n_src=2048, n_dst=4096, hidden_dim=11008),
    *_gather_cases_for("nsweep_1024", n_src=1024, n_dst=1024, hidden_dim=4096),
    *_gather_cases_for("nsweep_4096", n_src=4096, n_dst=4096, hidden_dim=4096),
    *_gather_cases_for("nsweep_16384", n_src=16384, n_dst=16384, hidden_dim=4096),
    *_gather_cases_for("nsweep_65536", n_src=65536, n_dst=65536, hidden_dim=4096),
]

COMBINE_BENCH_CASES: list[TokenCombineCase] = [
    *_combine_cases_for("mixtral_like", n_tokens=2048, k=2, hidden_dim=4096),
    *_combine_cases_for("deepseek_like", n_tokens=2048, k=6, hidden_dim=4096),
    *_combine_cases_for("tsweep_512", n_tokens=512, k=2, hidden_dim=4096),
    *_combine_cases_for("tsweep_2048", n_tokens=2048, k=2, hidden_dim=4096),
    *_combine_cases_for("tsweep_8192", n_tokens=8192, k=2, hidden_dim=4096),
    *_combine_cases_for("tsweep_32768", n_tokens=32768, k=2, hidden_dim=4096),
]


@dataclasses.dataclass
class GatherResult:
    impl: str
    case_name: str
    dtype: str
    n_dst_rows: int
    hidden_dim: int
    median_ms: float
    iqr_ms: float
    bandwidth_gb_s: float
    bandwidth_pct_peak: float


@dataclasses.dataclass
class CombineResult:
    impl: str
    case_name: str
    dtype: str
    n_tokens: int
    k: int
    hidden_dim: int
    median_ms: float
    iqr_ms: float
    bandwidth_gb_s: float
    bandwidth_pct_peak: float


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


def _stats_from_bytes(times_ms: list[float], total_bytes: int) -> tuple[float, float, float, float]:
    s = sorted(times_ms)
    median = statistics.median(s)
    q1 = s[int(0.25 * len(s))]
    q3 = s[int(0.75 * len(s))]
    iqr = q3 - q1
    seconds = median / 1000.0
    bandwidth = total_bytes / seconds / 1e9
    bandwidth_pct = 100.0 * bandwidth / PEAK_BANDWIDTH_GB_S
    return median, iqr, bandwidth, bandwidth_pct


def run_gather_benchmarks(device: str = "cuda") -> list[GatherResult]:
    results: list[GatherResult] = []
    flush_buf = _l2_flush_buffer(device)

    try:
        from custom_cuda.kernels.token_permute import token_gather

        have_kernel = True
    except ImportError:
        have_kernel = False
        print("custom_cuda._native not built — skipping cuda_kernel gather benchmarks.\n")

    for case in GATHER_BENCH_CASES:
        src, indices = make_gather_inputs(case, device=device)
        elem_size = torch.tensor([], dtype=case.dtype).element_size()
        total_bytes = 2 * case.n_dst_rows * case.hidden_dim * elem_size

        impls: dict[str, Callable[[], object]] = {
            "eager": lambda s=src, idx=indices: eager_token_gather(s, idx),
            "compiled": lambda s=src, idx=indices: compiled_token_gather(s, idx),
        }
        if have_kernel:
            impls["cuda_kernel"] = lambda s=src, idx=indices: token_gather(s, idx)

        for impl_name, fn in impls.items():
            times = _time_call(fn, flush_buf)
            median, iqr, bw, bw_pct = _stats_from_bytes(times, total_bytes)
            results.append(
                GatherResult(impl_name, case.name, str(case.dtype), case.n_dst_rows, case.hidden_dim, median, iqr, bw, bw_pct)
            )
            print(
                f"{impl_name:12s} {case.name:16s} median={median:9.4f}ms  iqr={iqr:8.4f}ms  "
                f"bw={bw:8.2f} GB/s ({bw_pct:5.1f}% peak)"
            )

    return results


def run_combine_benchmarks(device: str = "cuda") -> list[CombineResult]:
    results: list[CombineResult] = []
    flush_buf = _l2_flush_buffer(device)

    try:
        from custom_cuda.kernels.token_permute import token_combine

        have_kernel = True
    except ImportError:
        have_kernel = False
        print("custom_cuda._native not built — skipping cuda_kernel combine benchmarks.\n")

    for case in COMBINE_BENCH_CASES:
        expert_output, unpermute_index, weights = make_combine_inputs(case, device=device)
        elem_size = torch.tensor([], dtype=case.dtype).element_size()
        # read k rows/token (expert_output) + write 1 row/token (combined)
        total_bytes = (case.k + 1) * case.n_tokens * case.hidden_dim * elem_size

        impls: dict[str, Callable[[], object]] = {
            "eager": lambda eo=expert_output, ui=unpermute_index, w=weights: eager_token_combine(eo, ui, w),
            "compiled": lambda eo=expert_output, ui=unpermute_index, w=weights: compiled_token_combine(eo, ui, w),
        }
        if have_kernel:
            impls["cuda_kernel"] = lambda eo=expert_output, ui=unpermute_index, w=weights: token_combine(eo, ui, w)

        for impl_name, fn in impls.items():
            times = _time_call(fn, flush_buf)
            median, iqr, bw, bw_pct = _stats_from_bytes(times, total_bytes)
            results.append(
                CombineResult(impl_name, case.name, str(case.dtype), case.n_tokens, case.k, case.hidden_dim, median, iqr, bw, bw_pct)
            )
            print(
                f"{impl_name:12s} {case.name:16s} median={median:9.4f}ms  iqr={iqr:8.4f}ms  "
                f"bw={bw:8.2f} GB/s ({bw_pct:5.1f}% peak)"
            )

    return results


def main() -> None:
    gather_results = run_gather_benchmarks()
    out_path = Path(__file__).resolve().parent / "token_permute_gather_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dataclasses.fields(GatherResult)])
        writer.writeheader()
        for r in gather_results:
            writer.writerow(dataclasses.asdict(r))
    print(f"\nWrote {len(gather_results)} rows to {out_path}")

    combine_results = run_combine_benchmarks()
    combine_path = Path(__file__).resolve().parent / "token_permute_combine_results.csv"
    with open(combine_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dataclasses.fields(CombineResult)])
        writer.writeheader()
        for r in combine_results:
            writer.writerow(dataclasses.asdict(r))
    print(f"Wrote {len(combine_results)} rows to {combine_path}")


if __name__ == "__main__":
    main()
