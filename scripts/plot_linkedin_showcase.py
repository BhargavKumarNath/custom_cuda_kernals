"""Two publication-quality summary charts covering all 12 kernels at
once, for sharing the project outside the repo (LinkedIn, a blog post,
a portfolio page) rather than for auditing an individual kernel the way
`scripts/plot_<kernel>.py` do.

Reads real numbers out of the actual benchmark CSVs under `benchmarks/`
(run `custom_cuda_cli benchmark <kernel>` first for any kernel missing
one) — nothing here is a hand-typed figure. Every kernel's "headline"
case is chosen the same way: the largest-scale case available at the
most production-realistic dtype present (bf16, falling back to fp16,
then fp32) — see `_pick_headline_case`. That's a real methodological
choice, not a neutral default: it favors kernels' best-case numbers
over their worst-case ones (a small/edge-case shape would usually show
a smaller win). It's the right choice for a one-glance summary; anyone
wanting the full, honest range (including the shortfalls) should read
the per-kernel charts and `project_plan.md`, which this deliberately
doesn't replace.

Two charts:
  1. GPU Efficiency Matrix — a bubble scatter, one bubble per kernel,
     x = speedup vs. eager (log scale), y = hardware efficiency (0-100,
     meaning differs by kernel type — see `_METRIC_TYPE` below).
  2. Executive Scorecard — a two-panel horizontal bar chart, left =
     speedup leaderboard, right = the same efficiency metric, both in
     the same kernel order.

Color-by-domain palette note: 6 categorical groups shown simultaneously
in a scatter is an all-pairs-visible context, and this repo's validated
8-hue categorical palette (see the `dataviz` skill's `references/
palette.md`) only clears the automated colorblind-safety gate for its
first 3 slots under all-pairs comparison — checked with
`scripts/validate_palette.js`, not eyeballed, and it fails at 6 as
expected. The palette's own mitigation for that case is what's applied
here: every bubble/bar is directly labeled (never color-alone), plus a
distinct marker *shape* per domain as a second channel independent of
hue, so identity holds even for a colorblind viewer or a grayscale
printout.

Run directly: `python scripts/plot_linkedin_showcase.py`.
"""

from __future__ import annotations

import csv
import dataclasses
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from adjustText import adjust_text  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "benchmarks"
OUT_DIR = REPO_ROOT / "visualizations" / "linkedin_showcase"

PEAK_BANDWIDTH_GB_S = 256.0  # RTX 4070 Laptop GPU, consistent with every benchmarks/*.py

# ---------------------------------------------------------------------------
# Domain palette: validated 8-hue categorical order (dataviz skill), 6 of the
# 8 slots picked to avoid the two pairs the palette's own docs flag as weak
# under all-pairs viewing (yellow-vs-orange, aqua-vs-green). Each domain also
# gets a distinct marker shape — the secondary encoding channel the palette
# requires once a chart exceeds 3 simultaneously-visible categories.
# ---------------------------------------------------------------------------

DOMAIN_STYLE: dict[str, dict[str, str]] = {
    "Core Transformer Ops": {"color": "#2a78d6", "marker": "o"},
    "Compute & Loss": {"color": "#e87ba4", "marker": "s"},
    "MoE Routing & Permutation": {"color": "#4a3aa7", "marker": "^"},
    "RAG & Vector Search": {"color": "#008300", "marker": "D"},
    "Graph & Sequence": {"color": "#eb6834", "marker": "P"},
    "Precision & Quantization": {"color": "#e34948", "marker": "*"},
}

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

_DTYPE_PREFERENCE = ("torch.bfloat16", "torch.float16", "torch.float32")
_IDENTITY_COLS = {"impl", "case_name", "dtype", "fp8_format", "granularity"}
_METRIC_COLS = {
    "median_ms", "iqr_ms", "bandwidth_gb_s", "bandwidth_pct_peak", "tflops",
    "tokens_per_sec", "peak_incremental_mb",
}


# ---------------------------------------------------------------------------
# Generic CSV loading + "pick the headline case" — largest scale (product of
# every non-identity, non-metric numeric column) at the best available dtype.
# ---------------------------------------------------------------------------


def _load_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _size_score(row: dict[str, str], size_cols: list[str]) -> float:
    score = 1.0
    for c in size_cols:
        try:
            score *= max(float(row[c]), 1.0)
        except (ValueError, KeyError):
            continue
    return score


def _pick_headline_case(
    rows: list[dict[str, str]], extra_filter: Callable[[dict[str, str]], bool] | None = None
) -> str:
    """Return the `case_name` for the largest-scale case at the best
    available dtype, after `extra_filter` (e.g. fp8 format/granularity).
    """
    if extra_filter is not None:
        rows = [r for r in rows if extra_filter(r)]
    size_cols = [c for c in rows[0] if c not in _IDENTITY_COLS and c not in _METRIC_COLS]

    by_dtype: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        by_dtype.setdefault(r["dtype"], []).append(r)

    for dtype in _DTYPE_PREFERENCE:
        candidates = by_dtype.get(dtype)
        if not candidates:
            continue
        best = max(candidates, key=lambda r: _size_score(r, size_cols))
        return best["case_name"]
    # Fall back to whatever dtype is available.
    best = max(rows, key=lambda r: _size_score(r, size_cols))
    return best["case_name"]


def _rows_for_case(rows: list[dict[str, str]], case_name: str) -> dict[str, dict[str, str]]:
    """`{impl: row}` for every impl recorded at the given case_name."""
    return {r["impl"]: r for r in rows if r["case_name"] == case_name}


# ---------------------------------------------------------------------------
# Per-kernel result + the six metric-extraction strategies.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class KernelResult:
    name: str
    domain: str
    speedup: float
    efficiency_pct: float
    annotation: str


def _bandwidth_metric(
    display_name: str, domain: str, csv_path: Path,
    extra_filter: Callable[[dict[str, str]], bool] | None = None, label_suffix: str = "BW",
) -> KernelResult:
    rows = _load_csv(csv_path)
    if extra_filter is not None:
        rows = [r for r in rows if extra_filter(r)]
    case = _pick_headline_case(rows)
    by_impl = _rows_for_case(rows, case)
    eager_ms = float(by_impl["eager"]["median_ms"])
    cuda_ms = float(by_impl["cuda_kernel"]["median_ms"])
    speedup = eager_ms / cuda_ms

    cuda_row = by_impl["cuda_kernel"]
    if "bandwidth_pct_peak" in cuda_row:
        efficiency = float(cuda_row["bandwidth_pct_peak"])
    else:
        efficiency = float(cuda_row["bandwidth_gb_s"]) / PEAK_BANDWIDTH_GB_S * 100.0

    annotation = f"{display_name}: {efficiency:.0f}% {label_suffix}"
    return KernelResult(display_name, domain, speedup, efficiency, annotation)


def _cublas_ratio_metric(display_name: str, domain: str, csv_path: Path) -> KernelResult:
    """Kernel 5: compute-bound GEMM, efficiency expressed as % of
    cuBLAS's own fused `F.linear` throughput (the `f_linear` impl) —
    the honest yardstick project_plan.md itself uses for this kernel.
    """
    rows = _load_csv(csv_path)
    case = _pick_headline_case(rows)
    by_impl = _rows_for_case(rows, case)
    eager_ms = float(by_impl["eager"]["median_ms"])
    cuda_ms = float(by_impl["cuda_kernel"]["median_ms"])
    speedup = eager_ms / cuda_ms
    cuda_tflops = float(by_impl["cuda_kernel"]["tflops"])
    f_linear_tflops = float(by_impl["f_linear"]["tflops"])
    efficiency = cuda_tflops / f_linear_tflops * 100.0
    annotation = f"{display_name}: {efficiency:.0f}% of cuBLAS"
    return KernelResult(display_name, domain, speedup, efficiency, annotation)


def _vram_metric(
    display_name: str, domain: str, memory_csv: Path, results_csv: Path
) -> KernelResult:
    """Kernel 4: speedup from the latency sweep, efficiency from the
    dedicated peak-memory sweep (different case sets — each is a
    "largest scale" pick within its own file).
    """
    perf_rows = _load_csv(results_csv)
    perf_case = _pick_headline_case(perf_rows)
    perf_by_impl = _rows_for_case(perf_rows, perf_case)
    perf_eager_ms = float(perf_by_impl["eager"]["median_ms"])
    perf_cuda_ms = float(perf_by_impl["cuda_kernel"]["median_ms"])
    speedup = perf_eager_ms / perf_cuda_ms

    mem_rows = _load_csv(memory_csv)
    mem_case = _pick_headline_case(mem_rows)
    mem_by_impl = _rows_for_case(mem_rows, mem_case)
    eager_mb = float(mem_by_impl["eager"]["peak_incremental_mb"])
    cuda_mb = float(mem_by_impl["cuda_kernel"]["peak_incremental_mb"])
    reduction_pct = (eager_mb - cuda_mb) / eager_mb * 100.0
    ratio = eager_mb / cuda_mb

    annotation = f"{display_name}: {ratio:.0f}x VRAM Saved"
    return KernelResult(display_name, domain, speedup, reduction_pct, annotation)


def _launch_reduction_metric(display_name: str, domain: str, csv_path: Path) -> KernelResult:
    """Kernel 11: no bandwidth/TFLOPS ceiling to measure against (it's
    launch-latency-bound, not memory- or compute-bound) — efficiency is
    instead how much of the *theoretically possible* launch-count
    reduction (O(seq_len) -> O(1)) was actually achieved, which is a
    real, kernel-specific quantity rather than a borrowed one.
    """
    rows = _load_csv(csv_path)
    case = _pick_headline_case(rows)
    by_impl = _rows_for_case(rows, case)
    eager_ms = float(by_impl["eager"]["median_ms"])
    cuda_ms = float(by_impl["cuda_kernel"]["median_ms"])
    speedup = eager_ms / cuda_ms
    seq_len = float(by_impl["cuda_kernel"]["seq_len"])
    efficiency = (1.0 - 1.0 / seq_len) * 100.0
    annotation = f"{display_name}: {speedup:.0f}x, O(1) launches"
    return KernelResult(display_name, domain, speedup, efficiency, annotation)


def _average(
    results: list[KernelResult], name: str, domain: str, label_suffix: str
) -> KernelResult:
    speedup = sum(r.speedup for r in results) / len(results)
    efficiency = sum(r.efficiency_pct for r in results) / len(results)
    annotation = f"{name}: {efficiency:.0f}% {label_suffix}"
    return KernelResult(name, domain, speedup, efficiency, annotation)


def collect_results() -> list[KernelResult]:
    core = "Core Transformer Ops"
    compute = "Compute & Loss"
    moe = "MoE Routing & Permutation"
    rag = "RAG & Vector Search"
    graph = "Graph & Sequence"
    precision = "Precision & Quantization"

    results = [
        _bandwidth_metric("RMSNorm+Res", core, BENCH_DIR / "rmsnorm_residual_results.csv"),
        _bandwidth_metric("SwiGLU", core, BENCH_DIR / "swiglu_results.csv"),
        _bandwidth_metric("RoPE", core, BENCH_DIR / "rope_results.csv"),
        _vram_metric(
            "Linear CE", compute,
            BENCH_DIR / "linear_cross_entropy_memory.csv",
            BENCH_DIR / "linear_cross_entropy_results.csv",
        ),
        _cublas_ratio_metric("MatMul+Bias", compute, BENCH_DIR / "matmul_add_bias_results.csv"),
        _bandwidth_metric("MoE Router", moe, BENCH_DIR / "moe_router_results.csv"),
        _average(
            [
                _bandwidth_metric(
                    "Token Permute (gather)", moe, BENCH_DIR / "token_permute_gather_results.csv"
                ),
                _bandwidth_metric(
                    "Token Permute (combine)", moe, BENCH_DIR / "token_permute_combine_results.csv"
                ),
            ],
            "Token Permute", moe, "BW",
        ),
        _bandwidth_metric("Cosine TopK", rag, BENCH_DIR / "cosine_topk_results.csv"),
        _bandwidth_metric("Pairwise Dist", rag, BENCH_DIR / "pairwise_distance_results.csv"),
        _bandwidth_metric("Graph MsgPass", graph, BENCH_DIR / "graph_message_passing_results.csv"),
        _launch_reduction_metric("Viterbi", graph, BENCH_DIR / "viterbi_results.csv"),
        _bandwidth_metric(
            "FP8 Quant", precision, BENCH_DIR / "fp8_quant_results.csv",
            extra_filter=lambda r: r["fp8_format"] == "e4m3" and r["granularity"] == "block",
        ),
    ]
    return results


# ---------------------------------------------------------------------------
# Plot 1 — GPU Efficiency Matrix (bubble scatter).
# ---------------------------------------------------------------------------


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _bubble_size(speedup: float, all_speedups: list[float]) -> float:
    import math

    lo, hi = math.log10(min(all_speedups)), math.log10(max(all_speedups))
    if hi == lo:
        return 900.0
    t = (math.log10(speedup) - lo) / (hi - lo)
    return 180.0 + t * (1500.0 - 180.0)


def plot_efficiency_matrix(results: list[KernelResult]) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(15, 9.5))
    fig.patch.set_facecolor(SURFACE)

    all_speedups = [r.speedup for r in results]
    texts = []
    for r in results:
        style = DOMAIN_STYLE[r.domain]
        size = _bubble_size(r.speedup, all_speedups)
        ax.scatter(
            r.speedup, r.efficiency_pct, s=size, c=style["color"], marker=style["marker"],
            alpha=0.78, edgecolors="white", linewidths=1.8, zorder=3,
        )
        label_bbox = dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.85)
        texts.append(
            ax.text(
                r.speedup, r.efficiency_pct, r.annotation, fontsize=9.5, color=INK_PRIMARY,
                fontweight="bold", zorder=5, bbox=label_bbox,
            )
        )

    ax.axvline(1.0, color="#e34948", linewidth=1.6, linestyle="--", zorder=2)
    ax.text(
        1.05, 2, "PyTorch eager baseline (1x)", fontsize=8.5, color="#e34948", rotation=90,
        va="bottom",
    )

    ax.set_xscale("log")
    x_min = min(0.15, min(all_speedups) * 0.6)
    x_max = max(all_speedups) * 2.2
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-5, 108)

    ax.set_xlabel("Latency Speedup vs. PyTorch Eager (log scale)", fontsize=12, color=INK_PRIMARY)
    ax.set_ylabel(
        "Hardware Efficiency  (% peak bandwidth / % cuBLAS / % VRAM saved / % launch reduction)",
        fontsize=11, color=INK_PRIMARY,
    )
    ax.set_title(
        "The GPU Efficiency Matrix — 12 Hand-Written CUDA Kernels vs. PyTorch Eager",
        fontsize=15, color=INK_PRIMARY, fontweight="bold", pad=16,
    )
    _style_axes(ax)

    legend_handles = [
        Line2D(
            [0], [0], marker=style["marker"], color="none", markerfacecolor=style["color"],
            markeredgecolor="white", markersize=11, label=domain,
        )
        for domain, style in DOMAIN_STYLE.items()
    ]
    ax.legend(
        handles=legend_handles, loc="lower right", frameon=True, facecolor=SURFACE,
        edgecolor=BASELINE, fontsize=9.5, title="Kernel Domain", title_fontsize=10,
    )

    adjust_text(
        texts, ax=ax,
        arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.8, alpha=0.7),
        expand=(2.0, 2.4),
        force_text=(0.6, 0.9),
        force_points=(0.3, 0.4),
    )

    fig.text(
        0.5, -0.02,
        "Speedup and efficiency are both measured on real hardware (RTX 4070 Laptop, 256 GB/s "
        "peak) at each kernel's largest benchmarked, production-dtype case — see project_plan.md "
        "for the full range per kernel.",
        ha="center", fontsize=8.5, color=INK_MUTED,
    )

    _save(fig, "01_gpu_efficiency_matrix")


# ---------------------------------------------------------------------------
# Plot 2 — Executive Scorecard (dual-panel horizontal bars).
# ---------------------------------------------------------------------------


def plot_executive_scorecard(results: list[KernelResult]) -> None:
    sns.set_theme(style="whitegrid")
    ordered = sorted(results, key=lambda r: r.speedup, reverse=True)
    names = [r.name for r in ordered]
    y_pos = range(len(ordered))

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    colors = [DOMAIN_STYLE[r.domain]["color"] for r in ordered]

    left_vals = [r.speedup for r in ordered]
    bars_left = ax_left.barh(
        y_pos, left_vals, color=colors, edgecolor="white", linewidth=0.8, zorder=3
    )
    ax_left.axvline(
        1.0, color="#e34948", linewidth=1.8, linestyle="--", zorder=2, label="PyTorch eager (1x)"
    )
    ax_left.set_xlabel("Latency Speedup vs. Eager (x)", fontsize=11, color=INK_PRIMARY)
    ax_left.set_title("Speedup Leaderboard", fontsize=13, color=INK_PRIMARY, fontweight="bold")
    for bar, val in zip(bars_left, left_vals):
        offset = max(left_vals) * 0.015
        ax_left.text(
            bar.get_width() + offset, bar.get_y() + bar.get_height() / 2, f"{val:.1f}x",
            va="center", fontsize=9.5, color=INK_PRIMARY, fontweight="bold",
        )
    ax_left.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    right_vals = [r.efficiency_pct for r in ordered]
    bars_right = ax_right.barh(
        y_pos, right_vals, color=colors, edgecolor="white", linewidth=0.8, zorder=3
    )
    ax_right.set_xlabel("Hardware Saturation (%)", fontsize=11, color=INK_PRIMARY)
    ax_right.set_title("Hardware Saturation", fontsize=13, color=INK_PRIMARY, fontweight="bold")
    ax_right.set_xlim(0, 112)
    for bar, r in zip(bars_right, ordered):
        ax_right.text(
            bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2, f"{r.efficiency_pct:.0f}%",
            va="center", fontsize=9.5, color=INK_PRIMARY, fontweight="bold",
        )

    ax_left.set_yticks(list(y_pos))
    ax_left.set_yticklabels(names, fontsize=10.5, color=INK_PRIMARY)
    ax_left.invert_yaxis()

    for ax in (ax_left, ax_right):
        _style_axes(ax)
        ax.grid(axis="y", visible=False)

    legend_handles = [
        Line2D(
            [0], [0], marker="s", color="none", markerfacecolor=style["color"], markersize=11,
            label=domain,
        )
        for domain, style in DOMAIN_STYLE.items()
    ]
    fig.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=6,
        frameon=False, fontsize=9, labelcolor=INK_SECONDARY,
    )
    fig.suptitle(
        "Custom CUDA Kernels — Executive Scorecard (12 kernels, RTX 4070 Laptop)",
        fontsize=15, color=INK_PRIMARY, fontweight="bold", y=1.12,
    )
    fig.tight_layout()
    _save(fig, "02_executive_scorecard")


def _save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{name}.png"
    svg_path = OUT_DIR / f"{name}.svg"
    fig.savefig(png_path, dpi=300, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(svg_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path.relative_to(REPO_ROOT)} and {svg_path.relative_to(REPO_ROOT)}")


def main() -> None:
    missing = [
        p for p in [
            "rmsnorm_residual_results.csv", "swiglu_results.csv", "rope_results.csv",
            "linear_cross_entropy_memory.csv", "linear_cross_entropy_results.csv",
            "matmul_add_bias_results.csv", "moe_router_results.csv",
            "token_permute_gather_results.csv", "token_permute_combine_results.csv",
            "cosine_topk_results.csv", "pairwise_distance_results.csv",
            "graph_message_passing_results.csv", "viterbi_results.csv", "fp8_quant_results.csv",
        ]
        if not (BENCH_DIR / p).exists()
    ]
    if missing:
        print(f"error: missing benchmark CSVs: {missing}", file=sys.stderr)
        print(
            "Run `custom_cuda_cli benchmark <kernel>` for each missing kernel first.",
            file=sys.stderr,
        )
        sys.exit(1)

    results = collect_results()
    print(f"Collected headline results for {len(results)} kernels:")
    for r in results:
        print(
            f"  {r.name:16s} speedup={r.speedup:7.2f}x  efficiency={r.efficiency_pct:6.1f}%  "
            f"({r.domain})"
        )

    plot_efficiency_matrix(results)
    plot_executive_scorecard(results)


if __name__ == "__main__":
    main()
