"""Hardware-level benchmark harness for Kernel 5 (Fused MatMul + Add
Bias). Same methodology as benchmarks/rmsnorm_residual_bench.py (Section
5: CUDA events, L2 flush, warm-up/measurement counts), plus TFLOPS (this
op is compute-bound at realistic sizes, unlike Phase 1's elementwise
kernels) alongside bandwidth.

Four implementations compared: `eager` (naive unfused matmul + separate
bias add — the actual comparison point for this kernel's success
criteria), `f_linear` (PyTorch's own cuBLAS-fused epilogue — a much
higher bar, see baselines/matmul_add_bias.py), `compiled`
(torch.compile over the unfused pattern), and `cuda_kernel` (this
library's hand-written tiled GEMM).

Run directly: `python benchmarks/matmul_add_bias_bench.py`. Writes
results to benchmarks/matmul_add_bias_results.csv (git-ignored).
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

from baselines.matmul_add_bias import (
    MatmulBiasCase,
    compiled_matmul_add_bias,
    eager_matmul_add_bias_fused,
    eager_matmul_add_bias_unfused,
    make_inputs,
)

WARMUP_ITERS = 10
MEASURE_ITERS = 100
_L2_FLUSH_BYTES = 256 * 1024 * 1024
PEAK_BANDWIDTH_GB_S = 256.0  # RTX 4070 Laptop GPU — see rmsnorm_residual_bench.py

_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _cases_for(name: str, m: int, k: int, n: int) -> list[MatmulBiasCase]:
    return [
        MatmulBiasCase(f"{name}_{dt}".replace("torch.", ""), m, k, n, dt) for dt in _DTYPES
    ]


BENCH_CASES: list[MatmulBiasCase] = [
    *_cases_for("qkv_proj", m=2048, k=4096, n=4096),
    *_cases_for("mlp_up", m=2048, k=4096, n=11008),
    *_cases_for("mlp_down", m=2048, k=11008, n=4096),
    # M sweep at fixed K, N (typical seq-length/batch scaling)
    *_cases_for("msweep_128", m=128, k=4096, n=4096),
    *_cases_for("msweep_512", m=512, k=4096, n=4096),
    *_cases_for("msweep_2048", m=2048, k=4096, n=4096),
    *_cases_for("msweep_8192", m=8192, k=4096, n=4096),
]


@dataclasses.dataclass
class BenchResult:
    impl: str
    case_name: str
    dtype: str
    m: int
    k: int
    n: int
    median_ms: float
    iqr_ms: float
    bandwidth_gb_s: float
    tflops: float


def _l2_flush_buffer(device: str = "cuda") -> torch.Tensor:
    return torch.empty(_L2_FLUSH_BYTES // 4, dtype=torch.float32, device=device)


def _flush_l2(buf: torch.Tensor) -> None:
    buf.fill_(1.0)


def _ideal_bytes_moved(case: MatmulBiasCase) -> int:
    """read x, read weight, write y (bias is tiny/L2-resident, omitted)."""
    elem_size = torch.tensor([], dtype=case.dtype).element_size()
    return (case.m * case.k + case.n * case.k + case.m * case.n) * elem_size


def _flop_count(case: MatmulBiasCase) -> int:
    return 2 * case.m * case.k * case.n


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


def _stats(times_ms: list[float], case: MatmulBiasCase) -> tuple[float, float, float, float]:
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
        from custom_cuda.kernels.matmul_add_bias import matmul_add_bias

        have_kernel = True
    except ImportError:
        have_kernel = False
        print("custom_cuda._native not built — skipping cuda_kernel benchmarks.\n")

    for case in BENCH_CASES:
        x, weight, bias = make_inputs(case, device=device)

        impls: dict[str, Callable[[], object]] = {
            "eager": lambda x=x, w=weight, b=bias: eager_matmul_add_bias_unfused(x, w, b),
            "f_linear": lambda x=x, w=weight, b=bias: eager_matmul_add_bias_fused(x, w, b),
            "compiled": lambda x=x, w=weight, b=bias: compiled_matmul_add_bias(x, w, b),
        }
        if have_kernel:
            impls["cuda_kernel"] = lambda x=x, w=weight, b=bias: matmul_add_bias(x, w, b)

        for impl_name, fn in impls.items():
            times = _time_call(fn, flush_buf)
            median, iqr, bw, tflops = _stats(times, case)
            results.append(
                BenchResult(impl_name, case.name, str(case.dtype), case.m, case.k, case.n, median, iqr, bw, tflops)
            )
            print(
                f"{impl_name:12s} {case.name:16s} median={median:9.4f}ms  iqr={iqr:8.4f}ms  "
                f"bw={bw:8.2f} GB/s  tflops={tflops:7.3f}"
            )

    return results


def main() -> None:
    results = run_benchmarks()
    out_path = Path(__file__).resolve().parent / "matmul_add_bias_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dataclasses.fields(BenchResult)])
        writer.writeheader()
        for r in results:
            writer.writerow(dataclasses.asdict(r))
    print(f"\nWrote {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
