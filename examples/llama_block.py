"""Integration Proof: a real Llama-3-8B-shaped transformer block, built
two ways — once with plain PyTorch eager ops, once with three of this
repo's fused CUDA kernels dropped in — to show they compose correctly
end-to-end in an actual model, not just in isolation against synthetic
[M, N] test tensors.

Swapped in: Kernel 1 (Fused RMSNorm + Residual), Kernel 2 (Fused
SwiGLU), Kernel 3 (Fused RoPE). Left as standard PyTorch in *both*
blocks (identical code, identical cuBLAS-backed `nn.Linear` calls,
identical `F.scaled_dot_product_attention`) since none of the 12
kernels in this library target QKV/output projection GEMMs, the FFN's
two big projections, or attention itself — those dominate a real
block's FLOPs, so this benchmark honestly measures what fusing three
comparatively cheap, memory-bound ops actually buys inside a real model,
not an inflated per-kernel microbenchmark number. See project_plan.md's
"Integration Proofs" section for the measured result and how to read it.

Architecture mirrors Hugging Face's `LlamaDecoderLayer` at Llama-3-8B
scale: pre-norm, GQA (32 query heads / 8 KV heads), RoPE with
`theta=500000`, SwiGLU FFN, `rms_norm_eps=1e-5` — the same numbers in
Meta's released Llama-3-8B `config.json`.

Run directly: `python examples/llama_block.py`.
"""

from __future__ import annotations

import dataclasses
import statistics
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from baselines.rmsnorm_residual import eager_rmsnorm_residual
from baselines.rope import compute_cos_sin, eager_rope
from baselines.swiglu import eager_swiglu
from custom_cuda.kernels.rmsnorm_residual import rmsnorm_residual
from custom_cuda.kernels.rope import rope
from custom_cuda.kernels.swiglu import swiglu
from tests.numerics import DEFAULT_BF16_OUTLIER_FRACTION, assert_close_with_outliers

# ---------------------------------------------------------------------------
# Llama-3-8B config.json values.
# ---------------------------------------------------------------------------

HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 14336
NUM_HEADS = 32
NUM_KV_HEADS = 8
RMS_EPS = 1e-5
ROPE_THETA = 500000.0

SEQ_LEN = 4096
BATCH_SIZE = 1
DTYPE = torch.bfloat16
DEVICE = "cuda"

WARMUP_ITERS = 10
MEASURE_ITERS = 50
_L2_FLUSH_BYTES = 256 * 1024 * 1024
_SEED = 0


# ---------------------------------------------------------------------------
# Shared block scaffolding: linear projections, GQA repeat, attention, and
# parameter/buffer registration are identical between the two blocks by
# construction (same base class, same init) — only the three methods each
# subclass overrides (`_norm`, `_apply_rope`, `_swiglu`) differ, which is
# exactly the set of ops this repo has a custom kernel for.
# ---------------------------------------------------------------------------


class _LlamaBlockBase(nn.Module):
    def __init__(
        self,
        hidden_size: int = HIDDEN_SIZE,
        intermediate_size: int = INTERMEDIATE_SIZE,
        num_heads: int = NUM_HEADS,
        num_kv_heads: int = NUM_KV_HEADS,
        seq_len: int = SEQ_LEN,
        rms_eps: float = RMS_EPS,
        rope_theta: float = ROPE_THETA,
        dtype: torch.dtype = DTYPE,
        device: str = DEVICE,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.n_rep = num_heads // num_kv_heads
        self.rms_eps = rms_eps

        def linear(in_features: int, out_features: int) -> nn.Linear:
            return nn.Linear(in_features, out_features, bias=False, dtype=dtype, device=device)

        self.input_layernorm_weight = nn.Parameter(
            torch.ones(hidden_size, dtype=dtype, device=device)
        )
        self.post_attention_layernorm_weight = nn.Parameter(
            torch.ones(hidden_size, dtype=dtype, device=device)
        )

        self.q_proj = linear(hidden_size, num_heads * self.head_dim)
        self.k_proj = linear(hidden_size, num_kv_heads * self.head_dim)
        self.v_proj = linear(hidden_size, num_kv_heads * self.head_dim)
        self.o_proj = linear(num_heads * self.head_dim, hidden_size)

        self.gate_proj = linear(hidden_size, intermediate_size)
        self.up_proj = linear(hidden_size, intermediate_size)
        self.down_proj = linear(intermediate_size, hidden_size)

        cos, sin = compute_cos_sin(seq_len, self.head_dim, rope_theta, device=device)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, S, Hkv, D] -> [B, S, Hkv * n_rep, D]` (standard GQA repeat)."""
        if self.n_rep == 1:
            return x
        b, s, hkv, d = x.shape
        x = x[:, :, :, None, :].expand(b, s, hkv, self.n_rep, d)
        return x.reshape(b, s, hkv * self.n_rep, d)

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        b, s = q.shape[0], q.shape[1]
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, s, -1)
        return self.o_proj(attn_out)

    # Overridden per block: the three fused-vs-eager ops.
    def _norm(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _swiglu(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s = x.shape[0], x.shape[1]
        zero = torch.zeros_like(x)
        h, residual = self._norm(x, zero, self.input_layernorm_weight)

        q = self.q_proj(h).view(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(h).view(b, s, self.num_kv_heads, self.head_dim)
        v = self.v_proj(h).view(b, s, self.num_kv_heads, self.head_dim)
        q, k = self._apply_rope(q, k)
        attn_out = self._attention(q, k, v)

        h2, residual2 = self._norm(attn_out, residual, self.post_attention_layernorm_weight)
        gate = self.gate_proj(h2)
        up = self.up_proj(h2)
        mlp_out = self.down_proj(self._swiglu(gate, up))
        return mlp_out + residual2


class BaselineLlamaBlock(_LlamaBlockBase):
    """Standard PyTorch eager ops throughout — mirrors HF's
    `LlamaDecoderLayer` numerically exactly (same RMSNorm/SiLU/RoPE
    formulas, just not fused into single kernels).
    """

    def _norm(self, x, residual, weight):
        return eager_rmsnorm_residual(x, residual, weight, self.rms_eps)

    def _apply_rope(self, q, k):
        return eager_rope(q, k, self.rope_cos, self.rope_sin)

    def _swiglu(self, gate, up):
        return eager_swiglu(gate, up)


class CustomCUDA_LlamaBlock(_LlamaBlockBase):
    """Identical architecture, with Kernels 1/2/3 replacing the
    corresponding eager ops.
    """

    def _norm(self, x, residual, weight):
        return rmsnorm_residual(x, residual, weight, self.rms_eps)

    def _apply_rope(self, q, k):
        return rope(q, k, self.rope_cos, self.rope_sin)

    def _swiglu(self, gate, up):
        return swiglu(gate, up)


# ---------------------------------------------------------------------------
# Benchmark harness — same CUDA-event + L2-cache-flush methodology as
# benchmarks/*.py (project_plan.md Section 5), plus a dedicated,
# flush-buffer-free pass for peak VRAM so that scratch buffer doesn't
# inflate the reported memory number for either block.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BlockBenchResult:
    name: str
    median_ms: float
    iqr_ms: float
    peak_mem_bytes: int


def _l2_flush_buffer(device: str = DEVICE) -> torch.Tensor:
    return torch.empty(_L2_FLUSH_BYTES // 4, dtype=torch.float32, device=device)


def _flush_l2(buf: torch.Tensor) -> None:
    buf.fill_(1.0)


def _stats(times_ms: list[float]) -> tuple[float, float]:
    s = sorted(times_ms)
    median = statistics.median(s)
    q1 = s[int(0.25 * len(s))]
    q3 = s[int(0.75 * len(s))]
    return median, q3 - q1


def benchmark_block(
    name: str, block: nn.Module, x: torch.Tensor, flush_buf: torch.Tensor
) -> BlockBenchResult:
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            block(x)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times_ms: list[float] = []
        for _ in range(MEASURE_ITERS):
            _flush_l2(flush_buf)
            torch.cuda.synchronize()
            start.record()
            block(x)
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))

        # Dedicated, flush-buffer-free pass so the 256MB scratch buffer
        # used for L2 flushing above doesn't inflate the reported peak.
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        block(x)
        torch.cuda.synchronize()
        peak_mem_bytes = torch.cuda.max_memory_allocated()

    median, iqr = _stats(times_ms)
    return BlockBenchResult(name=name, median_ms=median, iqr_ms=iqr, peak_mem_bytes=peak_mem_bytes)


def check_correctness(baseline: nn.Module, custom: nn.Module, x: torch.Tensor) -> None:
    """Both blocks must agree on the actual output before any performance
    number is trustworthy — the same "verify before you benchmark"
    discipline as every other kernel in this repo, applied once at the
    full-block level.
    """
    with torch.no_grad():
        out_baseline = baseline(x)
        out_custom = custom(x)
    assert_close_with_outliers(
        out_custom, out_baseline, rtol=1e-2, atol=2e-2,
        max_outlier_fraction=DEFAULT_BF16_OUTLIER_FRACTION,
    )


# ---------------------------------------------------------------------------
# Presentation.
# ---------------------------------------------------------------------------


def _mb(num_bytes: int) -> float:
    return num_bytes / (1024**2)


def print_report(baseline_result: BlockBenchResult, custom_result: BlockBenchResult) -> None:
    console = Console()

    speedup = baseline_result.median_ms / custom_result.median_ms
    vram_saved_mb = _mb(baseline_result.peak_mem_bytes) - _mb(custom_result.peak_mem_bytes)
    vram_saved_pct = 100.0 * vram_saved_mb / _mb(baseline_result.peak_mem_bytes)

    table = Table(
        title="Llama-3-8B Block: PyTorch Eager vs. Custom CUDA Kernels",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold white",
        show_lines=True,
    )
    table.add_column("Metric", style="bold")
    table.add_column("Baseline (PyTorch Eager)", justify="right")
    table.add_column("Custom CUDA Kernels", justify="right")
    table.add_column("Improvement", justify="right", style="bold green")

    table.add_row(
        "Latency",
        f"{baseline_result.median_ms:.3f} ms",
        f"{custom_result.median_ms:.3f} ms",
        f"{speedup:.3f}x faster",
    )
    table.add_row(
        "Peak Memory",
        f"{_mb(baseline_result.peak_mem_bytes):.1f} MB",
        f"{_mb(custom_result.peak_mem_bytes):.1f} MB",
        f"{vram_saved_mb:+.1f} MB ({vram_saved_pct:+.1f}%)",
    )

    console.print()
    console.print(table)
    console.print(
        f"[dim]Latency is the median of {MEASURE_ITERS} measured iterations "
        f"(IQR: baseline {baseline_result.iqr_ms:.3f} ms, custom {custom_result.iqr_ms:.3f} ms) "
        f"after {WARMUP_ITERS} warmup iterations, with an L2 cache flush between each "
        f"measured iteration. Peak memory is from a separate, flush-buffer-free forward "
        f"pass.[/dim]"
    )
    console.print()
    console.print(
        Panel(
            f"[bold]Kernels swapped in:[/bold] Kernel 1 (Fused RMSNorm + Residual), "
            f"Kernel 2 (Fused SwiGLU), Kernel 3 (Fused RoPE)\n"
            f"[bold]Shape:[/bold] hidden={HIDDEN_SIZE}, intermediate={INTERMEDIATE_SIZE}, "
            f"heads={NUM_HEADS} (kv_heads={NUM_KV_HEADS}), seq_len={SEQ_LEN}, "
            f"batch={BATCH_SIZE}, dtype={DTYPE}\n"
            f"[bold]Everything else[/bold] (QKV/O projections, FFN matmuls, attention) is "
            f"identical, cuBLAS-backed PyTorch in both blocks — the improvement above is "
            f"attributable *only* to the three fused kernels, not to a faster attention "
            f"implementation or different GEMMs.",
            title="Integration Proof",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )
    console.print()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required to run this integration proof.")

    torch.manual_seed(_SEED)
    baseline = BaselineLlamaBlock().eval()
    custom = CustomCUDA_LlamaBlock().eval()
    custom.load_state_dict(baseline.state_dict())  # identical weights -> a fair comparison

    gen = torch.Generator(device=DEVICE).manual_seed(_SEED)
    x = (
        torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, dtype=DTYPE, device=DEVICE, generator=gen)
        * 0.02
    )

    print("Verifying baseline and custom blocks agree numerically before benchmarking...")
    check_correctness(baseline, custom, x)
    print("Correctness check passed — outputs match within bf16 tolerance.\n")

    flush_buf = _l2_flush_buffer()
    iters_msg = f"({WARMUP_ITERS} warmup + {MEASURE_ITERS} measured iters)"
    print(f"Benchmarking baseline {iters_msg}...")
    baseline_result = benchmark_block("baseline", baseline, x, flush_buf)
    print(f"Benchmarking custom CUDA block {iters_msg}...")
    custom_result = benchmark_block("custom", custom, x, flush_buf)

    print_report(baseline_result, custom_result)


if __name__ == "__main__":
    main()
