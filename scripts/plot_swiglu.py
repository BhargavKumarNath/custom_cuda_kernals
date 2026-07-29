"""Visualization pipeline for Kernel 2 (Fused SwiGLU Gated Activation).

project_plan.md Section 6. Mirrors scripts/plot_rmsnorm_residual.py — see
that file for the palette/design rationale (dataviz skill).
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
CSV_PATH = REPO_ROOT / "benchmarks" / "swiglu_results.csv"
OUT_DIR = REPO_ROOT / "visualizations" / "02_fused_swiglu"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
IMPL_COLOR = {"eager": BLUE, "compiled": ORANGE, "cuda_kernel": AQUA}
IMPL_LABEL = {"eager": "PyTorch eager", "compiled": "torch.compile", "cuda_kernel": "CUDA kernel"}
IMPL_ORDER = ["eager", "compiled", "cuda_kernel"]

DTYPE_COLOR = {"torch.float32": BLUE, "torch.float16": ORANGE, "torch.bfloat16": AQUA}
DTYPE_LABEL = {"torch.float32": "fp32", "torch.float16": "fp16", "torch.bfloat16": "bf16"}
DTYPE_ORDER = ["torch.float32", "torch.float16", "torch.bfloat16"]

PEAK_BANDWIDTH_GB_S = 256.0

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


def _load_rows() -> list[dict[str, str]]:
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def plot_speedup_bar(rows: list[dict[str, str]]) -> None:
    shapes = ["bs4_seq128_d3072", "bs2_seq2048_d11008", "bs1_seq4096_d11008"]
    shape_labels = {
        "bs4_seq128_d3072": "B=4,S=128\nD=3072",
        "bs2_seq2048_d11008": "B=2,S=2048\nD=11008",
        "bs1_seq4096_d11008": "B=1,S=4096\nD=11008",
    }

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, dtype in zip(axes, DTYPE_ORDER):
        by_shape_impl = {(r["case_name"].rsplit("_", 1)[0], r["impl"]): _f(r, "median_ms") for r in rows if r["dtype"] == dtype}

        x = range(len(shapes))
        width = 0.35
        for i, impl in enumerate(["compiled", "cuda_kernel"]):
            speedups = []
            for shape in shapes:
                eager_ms = by_shape_impl.get((shape, "eager"))
                impl_ms = by_shape_impl.get((shape, impl))
                speedups.append(eager_ms / impl_ms if eager_ms and impl_ms else 0.0)
            offset = (i - 0.5) * width
            bars = ax.bar(
                [xi + offset for xi in x], speedups, width, color=IMPL_COLOR[impl],
                label=IMPL_LABEL[impl], zorder=3,
            )
            for b, v in zip(bars, speedups):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.2f}x", ha="center",
                        va="bottom", fontsize=7.5, color=INK_SECONDARY)

        ax.axhline(1.0, color=BASELINE, linewidth=1, linestyle="--", zorder=2)
        ax.set_xticks(list(x))
        ax.set_xticklabels([shape_labels[s] for s in shapes], fontsize=8, color=INK_SECONDARY)
        ax.set_title(DTYPE_LABEL[dtype], fontsize=11, color=INK_PRIMARY, fontweight="bold")
        _style_axes(ax)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Speedup vs. PyTorch eager (x)", fontsize=10, color=INK_PRIMARY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=2,
               frameon=False, fontsize=9, labelcolor=INK_PRIMARY)
    fig.suptitle("Kernel 2: Fused SwiGLU — Speedup vs. Eager", fontsize=13,
                 color=INK_PRIMARY, y=1.12)
    fig.tight_layout()
    _save(fig, "01_speedup_bar")


def plot_latency_vs_size(rows: list[dict[str, str]]) -> None:
    sweep_prefix = "seqsweep_"
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, dtype in zip(axes, DTYPE_ORDER):
        by_impl: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for r in rows:
            if r["dtype"] != dtype or not r["case_name"].startswith(sweep_prefix):
                continue
            elements = _f(r, "rows") * _f(r, "cols")
            by_impl[r["impl"]].append((elements, _f(r, "median_ms")))

        for impl in IMPL_ORDER:
            pts = sorted(by_impl.get(impl, []))
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=IMPL_COLOR[impl], linewidth=2, marker="o", markersize=6,
                    label=IMPL_LABEL[impl], zorder=3)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(DTYPE_LABEL[dtype], fontsize=11, color=INK_PRIMARY, fontweight="bold")
        ax.set_xlabel("Tensor size (elements)", fontsize=9, color=INK_SECONDARY)
        _style_axes(ax)

    axes[0].set_ylabel("Median latency (ms, log scale)", fontsize=10, color=INK_PRIMARY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=3,
               frameon=False, fontsize=9, labelcolor=INK_PRIMARY)
    fig.suptitle("Kernel 2: Execution Latency vs. Tensor Size (log-log)", fontsize=13,
                 color=INK_PRIMARY, y=1.12)
    fig.tight_layout()
    _save(fig, "02_latency_vs_size")


def plot_bandwidth_vs_peak(rows: list[dict[str, str]]) -> None:
    sweep_prefix = "seqsweep_"
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, dtype in zip(axes, DTYPE_ORDER):
        for impl in IMPL_ORDER:
            xs, ys = [], []
            for r in rows:
                if r["dtype"] != dtype or r["impl"] != impl or not r["case_name"].startswith(sweep_prefix):
                    continue
                xs.append(_f(r, "rows") * _f(r, "cols"))
                ys.append(_f(r, "bandwidth_gb_s"))
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            xs, ys = [xs[i] for i in order], [ys[i] for i in order]
            ax.scatter(xs, ys, color=IMPL_COLOR[impl], s=45, label=IMPL_LABEL[impl], zorder=3)
            ax.plot(xs, ys, color=IMPL_COLOR[impl], linewidth=1, alpha=0.5, zorder=2)

        ax.axhline(PEAK_BANDWIDTH_GB_S, color=INK_MUTED, linewidth=1.25, linestyle="--", zorder=2)
        ax.text(ax.get_xlim()[0], PEAK_BANDWIDTH_GB_S + 6, "theoretical peak (256 GB/s)",
                fontsize=7.5, color=INK_MUTED)
        ax.set_xscale("log")
        ax.set_ylim(0, PEAK_BANDWIDTH_GB_S * 1.15)
        ax.set_title(DTYPE_LABEL[dtype], fontsize=11, color=INK_PRIMARY, fontweight="bold")
        ax.set_xlabel("Tensor size (elements, log scale)", fontsize=9, color=INK_SECONDARY)
        _style_axes(ax)

    axes[0].set_ylabel("Achieved memory bandwidth (GB/s)", fontsize=10, color=INK_PRIMARY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=3,
               frameon=False, fontsize=9, labelcolor=INK_PRIMARY)
    fig.suptitle("Kernel 2: Bandwidth Utilization vs. Theoretical Peak", fontsize=13,
                 color=INK_PRIMARY, y=1.12)
    fig.tight_layout()
    _save(fig, "03_bandwidth_vs_peak")


def plot_scaling(rows: list[dict[str, str]]) -> None:
    fig, (ax_seq, ax_batch) = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.patch.set_facecolor(SURFACE)

    for dtype in DTYPE_ORDER:
        pts = sorted(
            (_f(r, "rows"), _f(r, "median_ms"))
            for r in rows
            if r["dtype"] == dtype and r["impl"] == "cuda_kernel" and r["case_name"].startswith("seqsweep_")
        )
        if pts:
            xs, ys = zip(*pts)
            ax_seq.plot(xs, ys, color=DTYPE_COLOR[dtype], linewidth=2, marker="o", markersize=6,
                        label=DTYPE_LABEL[dtype], zorder=3)

    ax_seq.set_xscale("log")
    ax_seq.set_yscale("log")
    ax_seq.set_xlabel("Sequence length x batch (rows), batch=1", fontsize=9, color=INK_SECONDARY)
    ax_seq.set_ylabel("Median latency (ms, log scale)", fontsize=10, color=INK_PRIMARY)
    ax_seq.set_title("Scaling vs. sequence length (D=11008)", fontsize=10.5, color=INK_PRIMARY)
    _style_axes(ax_seq)

    for dtype in DTYPE_ORDER:
        pts = sorted(
            (_f(r, "rows"), _f(r, "median_ms"))
            for r in rows
            if r["dtype"] == dtype and r["impl"] == "cuda_kernel" and r["case_name"].startswith("batchsweep_")
        )
        if pts:
            xs, ys = zip(*pts)
            ax_batch.plot(xs, ys, color=DTYPE_COLOR[dtype], linewidth=2, marker="o", markersize=6,
                          label=DTYPE_LABEL[dtype], zorder=3)

    ax_batch.set_xscale("log")
    ax_batch.set_yscale("log")
    ax_batch.set_xlabel("Batch x seq=1024 (rows)", fontsize=9, color=INK_SECONDARY)
    ax_batch.set_title("Scaling vs. batch size (S=1024, D=11008)", fontsize=10.5, color=INK_PRIMARY)
    _style_axes(ax_batch)

    handles, labels = ax_seq.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=3,
               frameon=False, fontsize=9, labelcolor=INK_PRIMARY)
    fig.suptitle("Kernel 2 (CUDA): Scaling Across Sequence Length / Batch Size", fontsize=13,
                 color=INK_PRIMARY, y=1.14)
    fig.tight_layout()
    _save(fig, "04_scaling_seqlen_batch")


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
        print(f"error: {CSV_PATH} not found — run benchmarks/swiglu_bench.py first", file=sys.stderr)
        sys.exit(1)

    rows = _load_rows()
    plot_speedup_bar(rows)
    plot_latency_vs_size(rows)
    plot_bandwidth_vs_peak(rows)
    plot_scaling(rows)


if __name__ == "__main__":
    main()
