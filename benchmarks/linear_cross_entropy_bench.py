"""Hardware-level benchmark harness for Kernel 4 (Fused Linear Cross
Entropy Loss). Latency methodology matches
benchmarks/rmsnorm_residual_bench.py (Section 5: CUDA events, L2 flush,
warm-up/measurement counts). Additionally — per this kernel's special
instructions — measures **peak incremental VRAM**, since the whole point
of chunked online-softmax streaming is avoiding the full `[N, V]` logits
allocation eager materializes.

Bandwidth-vs-peak is not reported for this kernel (Section 5.4 allows
omitting metrics "where not meaningful"): this op is GEMM-bound with a
traffic pattern that depends on the chosen chunk_size, unlike the
elementwise/reduction kernels in Phase 1, so latency + peak memory + TFLOPS
are the meaningful axes here instead.

All measurements run under `torch.no_grad()` for both implementations —
the custom kernel is forward-only and autograd-opaque (see
custom_cuda/kernels/linear_cross_entropy.py), so an eager-vs-kernel
comparison is only apples-to-apples in inference mode; a `requires_grad`
eager run would additionally pay for autograd's saved-tensor bookkeeping,
which the kernel path doesn't (and currently can't) participate in.

Run directly: `python benchmarks/linear_cross_entropy_bench.py`. Writes
results to benchmarks/linear_cross_entropy_results.csv and
benchmarks/linear_cross_entropy_memory.csv (both git-ignored).
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

from baselines.linear_cross_entropy import LinearCECase, compiled_linear_cross_entropy, eager_linear_cross_entropy, make_inputs

WARMUP_ITERS = 10
MEASURE_ITERS = 100
_L2_FLUSH_BYTES = 256 * 1024 * 1024

_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _cases_for(name: str, n_tokens: int, hidden_dim: int, vocab_size: int) -> list[LinearCECase]:
    return [
        LinearCECase(f"{name}_{dt}".replace("torch.", ""), n_tokens, hidden_dim, vocab_size, dt)
        for dt in _DTYPES
    ]


# Latency/TFLOPS sweep — kept modest so torch.compile's max-autotune
# compilation across many (shape, dtype) pairs stays tractable.
LATENCY_CASES: list[LinearCECase] = [
    *_cases_for("small_vocab", n_tokens=512, hidden_dim=1024, vocab_size=8000),
    *_cases_for("llama2_vocab", n_tokens=512, hidden_dim=4096, vocab_size=32000),
    *_cases_for("llama3_vocab", n_tokens=512, hidden_dim=4096, vocab_size=128256),
]

# Dedicated VRAM comparison: larger, more realistic training-batch shapes
# where the full-logits-materialization cost is dramatic. fp16 only (the
# dtype VRAM-constrained training actually runs in).
MEMORY_CASES: list[LinearCECase] = [
    LinearCECase("mem_llama2_2k_tokens", 2048, 4096, 32000, torch.float16),
    LinearCECase("mem_llama3_2k_tokens", 2048, 4096, 128256, torch.float16),
    LinearCECase("mem_llama3_4k_tokens", 4096, 4096, 128256, torch.float16),
    LinearCECase("mem_llama3_8k_tokens", 8192, 4096, 128256, torch.float16),
]


@dataclasses.dataclass
class LatencyResult:
    impl: str
    case_name: str
    dtype: str
    n_tokens: int
    hidden_dim: int
    vocab_size: int
    median_ms: float
    iqr_ms: float
    tflops: float


@dataclasses.dataclass
class MemoryResult:
    impl: str
    case_name: str
    dtype: str
    n_tokens: int
    hidden_dim: int
    vocab_size: int
    peak_incremental_mb: float


@dataclasses.dataclass
class ChunkSweepResult:
    chunk_size: int
    n_chunks: int
    median_ms: float
    iqr_ms: float
    peak_incremental_mb: float


# chunk_size sweep case: large enough vocab that the memory/latency
# trade-off is visible; 131072 > vocab_size collapses to a single chunk
# (the eager-equivalent memory profile), anchoring one end of the curve.
CHUNK_SWEEP_CASE = LinearCECase("chunk_sweep", n_tokens=2048, hidden_dim=4096, vocab_size=128256, dtype=torch.float16)
CHUNK_SWEEP_SIZES = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]


def _l2_flush_buffer(device: str = "cuda") -> torch.Tensor:
    return torch.empty(_L2_FLUSH_BYTES // 4, dtype=torch.float32, device=device)


def _flush_l2(buf: torch.Tensor) -> None:
    buf.fill_(1.0)


def _flop_count(case: LinearCECase) -> int:
    """2 * N * V * H for the logits matmul (dominates FLOPs by orders of
    magnitude over the softmax/reduction arithmetic).
    """
    return 2 * case.n_tokens * case.vocab_size * case.hidden_dim


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


def _stats(times_ms: list[float], case: LinearCECase) -> tuple[float, float, float]:
    s = sorted(times_ms)
    median = statistics.median(s)
    q1 = s[int(0.25 * len(s))]
    q3 = s[int(0.75 * len(s))]
    iqr = q3 - q1
    tflops = _flop_count(case) / (median / 1000.0) / 1e12
    return median, iqr, tflops


def _measure_peak_incremental_mb(fn: Callable[[], object], device: str = "cuda") -> float:
    """Peak memory allocated *during* `fn()`, above whatever was already
    allocated beforehand (i.e. excluding hidden/weight/targets, which both
    implementations need equally) — isolates the transient/intermediate
    memory footprint the fused kernel exists to shrink.
    """
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    result = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(device)
    del result
    torch.cuda.synchronize()
    return (peak - baseline) / (1024 * 1024)


def run_latency_benchmarks(device: str = "cuda") -> list[LatencyResult]:
    results: list[LatencyResult] = []
    flush_buf = _l2_flush_buffer(device)

    try:
        from custom_cuda.kernels.linear_cross_entropy import linear_cross_entropy

        have_kernel = True
    except ImportError:
        have_kernel = False
        print("custom_cuda._native not built — skipping cuda_kernel benchmarks.\n")

    for case in LATENCY_CASES:
        hidden, weight, targets = make_inputs(case, device=device)

        impls: dict[str, Callable[[], object]] = {
            "eager": lambda h=hidden, w=weight, t=targets: eager_linear_cross_entropy(h, w, t),
            "compiled": lambda h=hidden, w=weight, t=targets: compiled_linear_cross_entropy(h, w, t),
        }
        if have_kernel:
            impls["cuda_kernel"] = lambda h=hidden, w=weight, t=targets: linear_cross_entropy(h, w, t)

        for impl_name, fn in impls.items():
            times = _time_call(fn, flush_buf)
            median, iqr, tflops = _stats(times, case)
            results.append(
                LatencyResult(
                    impl_name, case.name, str(case.dtype), case.n_tokens, case.hidden_dim,
                    case.vocab_size, median, iqr, tflops,
                )
            )
            print(
                f"{impl_name:12s} {case.name:20s} median={median:9.4f}ms  iqr={iqr:8.4f}ms  tflops={tflops:6.3f}"
            )

    return results


def run_memory_benchmarks(device: str = "cuda") -> list[MemoryResult]:
    results: list[MemoryResult] = []

    try:
        from custom_cuda.kernels.linear_cross_entropy import linear_cross_entropy

        have_kernel = True
    except ImportError:
        have_kernel = False
        print("custom_cuda._native not built — skipping cuda_kernel memory benchmarks.\n")

    print()
    for case in MEMORY_CASES:
        hidden, weight, targets = make_inputs(case, device=device)

        impls: dict[str, Callable[[], object]] = {
            "eager": lambda h=hidden, w=weight, t=targets: eager_linear_cross_entropy(h, w, t),
        }
        if have_kernel:
            impls["cuda_kernel"] = lambda h=hidden, w=weight, t=targets: linear_cross_entropy(h, w, t)

        for impl_name, fn in impls.items():
            peak_mb = _measure_peak_incremental_mb(fn, device)
            results.append(
                MemoryResult(impl_name, case.name, str(case.dtype), case.n_tokens, case.hidden_dim, case.vocab_size, peak_mb)
            )
            print(f"{impl_name:12s} {case.name:24s} peak_incremental={peak_mb:9.1f} MB")

    return results


def run_chunk_sweep(device: str = "cuda") -> list[ChunkSweepResult]:
    """Characterizes the chunk_size latency-vs-memory trade-off directly
    (the v1 default of 4096 traded significant latency for memory savings
    — see project_plan.md's Kernel 4 entry — this sweep is what a bigger
    default should be chosen from, rather than guessed).
    """
    from custom_cuda.kernels.linear_cross_entropy import linear_cross_entropy

    results: list[ChunkSweepResult] = []
    flush_buf = _l2_flush_buffer(device)
    hidden, weight, targets = make_inputs(CHUNK_SWEEP_CASE, device=device)

    print()
    for chunk_size in CHUNK_SWEEP_SIZES:
        fn = lambda cs=chunk_size: linear_cross_entropy(hidden, weight, targets, chunk_size=cs)  # noqa: E731
        times = _time_call(fn, flush_buf)
        median, iqr, _ = _stats(times, CHUNK_SWEEP_CASE)
        peak_mb = _measure_peak_incremental_mb(fn, device)
        n_chunks = (CHUNK_SWEEP_CASE.vocab_size + chunk_size - 1) // chunk_size
        results.append(ChunkSweepResult(chunk_size, n_chunks, median, iqr, peak_mb))
        print(
            f"chunk_size={chunk_size:7d} ({n_chunks:3d} chunks)  median={median:9.4f}ms  "
            f"iqr={iqr:8.4f}ms  peak_incremental={peak_mb:9.1f} MB"
        )

    return results


def main() -> None:
    torch.set_grad_enabled(False)  # see module docstring: inference-mode comparison
    latency_results = run_latency_benchmarks()
    out_path = Path(__file__).resolve().parent / "linear_cross_entropy_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dataclasses.fields(LatencyResult)])
        writer.writeheader()
        for r in latency_results:
            writer.writerow(dataclasses.asdict(r))
    print(f"\nWrote {len(latency_results)} rows to {out_path}")

    memory_results = run_memory_benchmarks()
    mem_path = Path(__file__).resolve().parent / "linear_cross_entropy_memory.csv"
    with open(mem_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dataclasses.fields(MemoryResult)])
        writer.writeheader()
        for r in memory_results:
            writer.writerow(dataclasses.asdict(r))
    print(f"Wrote {len(memory_results)} rows to {mem_path}")

    sweep_results = run_chunk_sweep()
    sweep_path = Path(__file__).resolve().parent / "linear_cross_entropy_chunk_sweep.csv"
    with open(sweep_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dataclasses.fields(ChunkSweepResult)])
        writer.writeheader()
        for r in sweep_results:
            writer.writerow(dataclasses.asdict(r))
    print(f"Wrote {len(sweep_results)} rows to {sweep_path}")


if __name__ == "__main__":
    main()
