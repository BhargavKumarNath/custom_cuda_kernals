"""Hardware-level benchmark harness for Kernel 6 (MoE Top-K Router). Same
methodology as benchmarks/rmsnorm_residual_bench.py (Section 5: CUDA
events, L2 flush, warm-up/measurement counts).

Run directly: `python benchmarks/moe_router_bench.py`. Writes results to
benchmarks/moe_router_results.csv (git-ignored).
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

from baselines.moe_router import MoERouterCase, compiled_moe_router, eager_moe_router, make_inputs

WARMUP_ITERS = 10
MEASURE_ITERS = 100
_L2_FLUSH_BYTES = 256 * 1024 * 1024
PEAK_BANDWIDTH_GB_S = 256.0  # RTX 4070 Laptop GPU — rmsnorm_residual_bench.py

_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _cases_for(name: str, n_tokens: int, n_experts: int, k: int) -> list[MoERouterCase]:
    return [
        MoERouterCase(f"{name}_{dt}".replace("torch.", ""), n_tokens, n_experts, k, dt)
        for dt in _DTYPES
    ]


BENCH_CASES: list[MoERouterCase] = [
    *_cases_for("mixtral_like", n_tokens=4096, n_experts=8, k=2),
    *_cases_for("deepseek_like", n_tokens=4096, n_experts=160, k=6),
    *_cases_for("ksweep_k1", n_tokens=4096, n_experts=32, k=1),
    *_cases_for("ksweep_k2", n_tokens=4096, n_experts=32, k=2),
    *_cases_for("ksweep_k4", n_tokens=4096, n_experts=32, k=4),
    *_cases_for("ksweep_k8", n_tokens=4096, n_experts=32, k=8),
    # token-count sweep at fixed experts/k (Mixtral-shaped)
    *_cases_for("tsweep_512", n_tokens=512, n_experts=8, k=2),
    *_cases_for("tsweep_2048", n_tokens=2048, n_experts=8, k=2),
    *_cases_for("tsweep_8192", n_tokens=8192, n_experts=8, k=2),
    *_cases_for("tsweep_32768", n_tokens=32768, n_experts=8, k=2),
]


@dataclasses.dataclass
class BenchResult:
    impl: str
    case_name: str
    dtype: str
    n_tokens: int
    n_experts: int
    k: int
    median_ms: float
    iqr_ms: float
    bandwidth_gb_s: float
    tokens_per_sec: float


def _l2_flush_buffer(device: str = "cuda") -> torch.Tensor:
    return torch.empty(_L2_FLUSH_BYTES // 4, dtype=torch.float32, device=device)


def _flush_l2(buf: torch.Tensor) -> None:
    buf.fill_(1.0)


def _ideal_bytes_moved(case: MoERouterCase) -> int:
    """read logits [T,E], write topk_weights [T,k] (fp32) + topk_indices
    [T,k] (int64).
    """
    elem_size = torch.tensor([], dtype=case.dtype).element_size()
    logits_bytes = case.n_tokens * case.n_experts * elem_size
    weights_bytes = case.n_tokens * case.k * 4
    indices_bytes = case.n_tokens * case.k * 8
    return logits_bytes + weights_bytes + indices_bytes


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


def _stats(times_ms: list[float], case: MoERouterCase) -> tuple[float, float, float, float]:
    s = sorted(times_ms)
    median = statistics.median(s)
    q1 = s[int(0.25 * len(s))]
    q3 = s[int(0.75 * len(s))]
    iqr = q3 - q1
    seconds = median / 1000.0
    bandwidth = _ideal_bytes_moved(case) / seconds / 1e9
    tokens_per_sec = case.n_tokens / seconds
    return median, iqr, bandwidth, tokens_per_sec


def run_benchmarks(device: str = "cuda") -> list[BenchResult]:
    results: list[BenchResult] = []
    flush_buf = _l2_flush_buffer(device)

    try:
        from custom_cuda.kernels.moe_router import moe_router

        have_kernel = True
    except ImportError:
        have_kernel = False
        print("custom_cuda._native not built — skipping cuda_kernel benchmarks.\n")

    for case in BENCH_CASES:
        logits = make_inputs(case, device=device)

        impls: dict[str, Callable[[], object]] = {
            "eager": lambda lg=logits, c=case: eager_moe_router(lg, c.k, c.renormalize),
            "compiled": lambda lg=logits, c=case: compiled_moe_router(lg, c.k, c.renormalize),
        }
        if have_kernel:
            impls["cuda_kernel"] = lambda lg=logits, c=case: moe_router(lg, c.k, c.renormalize)

        for impl_name, fn in impls.items():
            times = _time_call(fn, flush_buf)
            median, iqr, bw, tps = _stats(times, case)
            results.append(
                BenchResult(
                    impl_name, case.name, str(case.dtype), case.n_tokens, case.n_experts, case.k,
                    median, iqr, bw, tps,
                )
            )
            print(
                f"{impl_name:12s} {case.name:16s} median={median:9.4f}ms  iqr={iqr:8.4f}ms  "
                f"bw={bw:8.2f} GB/s  tokens/s={tps:14.0f}"
            )

    return results


def main() -> None:
    results = run_benchmarks()
    out_path = Path(__file__).resolve().parent / "moe_router_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dataclasses.fields(BenchResult)])
        writer.writeheader()
        for r in results:
            writer.writerow(dataclasses.asdict(r))
    print(f"\nWrote {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
