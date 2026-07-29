"""Visualization pipeline for Kernel 11 (Parallel Viterbi Algorithm).
Latency-focused (not bandwidth/TFLOPS) — matches project_plan.md
Section 3.11's own "speedup vs. per-timestep loop" framing for this
inherently sequential, launch-overhead-dominated op. `compiled` is
absent from the underlying benchmark data (see
benchmarks/viterbi_bench.py's docstring for why) so every chart here
compares only `eager` vs. `cuda_kernel`.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "benchmarks" / "viterbi_results.csv"
OUT_DIR = REPO_ROOT / "visualizations" / "11_parallel_viterbi"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
IMPL_COLOR = {"eager": BLUE, "cuda_kernel": AQUA}
IMPL_LABEL = {"eager": "PyTorch eager (per-timestep loop)", "cuda_kernel": "CUDA kernel (persistent)"}
IMPL_ORDER = ["eager", "cuda_kernel"]

DTYPE_COLOR = {"torch.float32": BLUE, "torch.float16": ORANGE, "torch.bfloat16": AQUA}
DTYPE_LABEL = {"torch.float32": "fp32", "torch.float16": "fp16", "torch.bfloat16": "bf16"}
DTYPE_ORDER = ["torch.float32", "torch.float16", "torch.bfloat16"]

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _load(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def plot_speedup_vs_seq_len(rows: list[dict[str, str]]) -> None:
    sweep_prefix = "tsweep_"
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig.patch.set_facecolor(SURFACE)

    for dtype in DTYPE_ORDER:
        by_impl: dict[str, dict[float, float]] = defaultdict(dict)
        for r in rows:
            if r["dtype"] != dtype or not r["case_name"].startswith(sweep_prefix):
                continue
            by_impl[r["impl"]][_f(r, "seq_len")] = _f(r, "median_ms")

        seq_lens = sorted(by_impl["eager"].keys())
        speedups = [by_impl["eager"][t] / by_impl["cuda_kernel"][t] for t in seq_lens]
        ax.plot(seq_lens, speedups, color=DTYPE_COLOR[dtype], linewidth=2, marker="o", markersize=6,
                label=DTYPE_LABEL[dtype], zorder=3)

    ax.axhline(5.0, color=INK_MUTED, linewidth=1.2, linestyle=":", zorder=2)
    ax.text(60, 6.5, "5x target", fontsize=8.5, color=INK_MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length T (log scale), B=64, S=16", fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel("Speedup vs. eager per-timestep loop (x, log scale)", fontsize=10, color=INK_PRIMARY)
    _style_axes(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_PRIMARY)
    ax.set_title("Kernel 11: Speedup vs. Sequence Length", fontsize=13, color=INK_PRIMARY)
    fig.tight_layout()
    _save(fig, "01_speedup_vs_seq_len")


def plot_latency_vs_seq_len(rows: list[dict[str, str]]) -> None:
    sweep_prefix = "tsweep_"
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, dtype in zip(axes, DTYPE_ORDER):
        by_impl: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for r in rows:
            if r["dtype"] != dtype or not r["case_name"].startswith(sweep_prefix):
                continue
            by_impl[r["impl"]].append((_f(r, "seq_len"), _f(r, "median_ms")))

        for impl in IMPL_ORDER:
            pts = sorted(by_impl.get(impl, []))
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=IMPL_COLOR[impl], linewidth=2, marker="o", markersize=5,
                    label=IMPL_LABEL[impl], zorder=3)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(DTYPE_LABEL[dtype], fontsize=11, color=INK_PRIMARY, fontweight="bold")
        ax.set_xlabel("Sequence length T (log scale)", fontsize=9, color=INK_SECONDARY)
        _style_axes(ax)

    axes[0].set_ylabel("Median latency (ms, log scale)", fontsize=10, color=INK_PRIMARY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2,
               frameon=False, fontsize=8.5, labelcolor=INK_PRIMARY)
    fig.suptitle("Kernel 11: Latency vs. Sequence Length (log-log)", fontsize=13, color=INK_PRIMARY, y=1.18)
    fig.tight_layout()
    _save(fig, "02_latency_vs_seq_len")


def plot_latency_vs_batch(rows: list[dict[str, str]]) -> None:
    sweep_prefix = "bsweep_"
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, dtype in zip(axes, DTYPE_ORDER):
        by_impl: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for r in rows:
            if r["dtype"] != dtype or not r["case_name"].startswith(sweep_prefix):
                continue
            by_impl[r["impl"]].append((_f(r, "batch"), _f(r, "median_ms")))

        for impl in IMPL_ORDER:
            pts = sorted(by_impl.get(impl, []))
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=IMPL_COLOR[impl], linewidth=2, marker="o", markersize=5,
                    label=IMPL_LABEL[impl], zorder=3)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(DTYPE_LABEL[dtype], fontsize=11, color=INK_PRIMARY, fontweight="bold")
        ax.set_xlabel("Batch size (log scale), T=512, S=16", fontsize=9, color=INK_SECONDARY)
        _style_axes(ax)

    axes[0].set_ylabel("Median latency (ms, log scale)", fontsize=10, color=INK_PRIMARY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2,
               frameon=False, fontsize=8.5, labelcolor=INK_PRIMARY)
    fig.suptitle("Kernel 11: Latency vs. Batch Size — eager is launch-overhead-bound, not batch-bound",
                 fontsize=12, color=INK_PRIMARY, y=1.18)
    fig.tight_layout()
    _save(fig, "03_latency_vs_batch")


def plot_latency_vs_states(rows: list[dict[str, str]]) -> None:
    sweep_prefix = "ssweep_"
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig.patch.set_facecolor(SURFACE)

    for dtype in DTYPE_ORDER:
        pts = sorted(
            (_f(r, "num_states"), _f(r, "median_ms"))
            for r in rows
            if r["dtype"] == dtype and r["impl"] == "cuda_kernel" and r["case_name"].startswith(sweep_prefix)
        )
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=DTYPE_COLOR[dtype], linewidth=2, marker="o", markersize=6,
                    label=DTYPE_LABEL[dtype], zorder=3)

    ax.set_xlabel("Number of states S, T=512, B=64", fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel("Median latency (ms)", fontsize=10, color=INK_PRIMARY)
    _style_axes(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_PRIMARY)
    ax.set_title("Kernel 11 (CUDA): Scaling vs. Number of States", fontsize=13, color=INK_PRIMARY)
    fig.tight_layout()
    _save(fig, "04_latency_vs_states")


def _save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{name}.png"
    svg_path = OUT_DIR / f"{name}.svg"
    fig.savefig(png_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(svg_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path.relative_to(REPO_ROOT)} and {svg_path.relative_to(REPO_ROOT)}")


def main() -> None:
    if not CSV_PATH.exists():
        print(f"error: {CSV_PATH} not found — run benchmarks/viterbi_bench.py first", file=sys.stderr)
        sys.exit(1)

    rows = _load(CSV_PATH)
    plot_speedup_vs_seq_len(rows)
    plot_latency_vs_seq_len(rows)
    plot_latency_vs_batch(rows)
    plot_latency_vs_states(rows)


if __name__ == "__main__":
    main()
