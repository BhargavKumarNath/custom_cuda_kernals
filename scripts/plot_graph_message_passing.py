"""Visualization pipeline for Kernel 10 (Spatiotemporal Graph Message
Passing). Bandwidth-focused (not TFLOPS) — this op is bandwidth-bound
(arithmetic intensity ~1 FMA per feature element read), the same
convention as Kernels 1-3/6/7 rather than Kernel 5/9's compute-bound
TFLOPS convention.
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
CSV_PATH = REPO_ROOT / "benchmarks" / "graph_message_passing_results.csv"
OUT_DIR = REPO_ROOT / "visualizations" / "10_spatiotemporal_graph_message_passing"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
IMPL_COLOR = {"eager": BLUE, "compiled": ORANGE, "cuda_kernel": AQUA}
IMPL_LABEL = {
    "eager": "PyTorch eager (index_add_)",
    "compiled": "torch.compile",
    "cuda_kernel": "CUDA kernel",
}
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


def _load(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def plot_speedup_bar(rows: list[dict[str, str]]) -> None:
    shapes = ["nsweep_2000", "nsweep_20000", "nsweep_100000", "nsweep_500000"]
    shape_labels = {
        "nsweep_2000": "N=2K\ndeg~20",
        "nsweep_20000": "N=20K\ndeg~20",
        "nsweep_100000": "N=100K\ndeg~20",
        "nsweep_500000": "N=500K\ndeg~20",
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, dtype in zip(axes, DTYPE_ORDER):
        by_shape_impl = {
            (r["case_name"].rsplit("_", 1)[0], r["impl"]): _f(r, "median_ms")
            for r in rows
            if r["dtype"] == dtype
        }

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
                ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}x", ha="center",
                        va="bottom", fontsize=7.5, color=INK_SECONDARY)

        ax.axhline(1.0, color=BASELINE, linewidth=1, linestyle="--", zorder=2)
        ax.axhline(2.0, color=INK_MUTED, linewidth=1, linestyle=":", zorder=2)
        ax.set_xticks(list(x))
        ax.set_xticklabels([shape_labels[s] for s in shapes], fontsize=8, color=INK_SECONDARY)
        ax.set_title(DTYPE_LABEL[dtype], fontsize=11, color=INK_PRIMARY, fontweight="bold")
        _style_axes(ax)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Speedup vs. PyTorch eager (index_add_, x)", fontsize=10, color=INK_PRIMARY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2,
               frameon=False, fontsize=9, labelcolor=INK_PRIMARY)
    fig.suptitle("Kernel 10: Graph Message Passing — Speedup vs. Eager (dotted line = 2x target)",
                 fontsize=12.5, color=INK_PRIMARY, y=1.20)
    fig.tight_layout()
    _save(fig, "01_speedup_bar")


def plot_bandwidth_vs_nodes(rows: list[dict[str, str]]) -> None:
    sweep_prefix = "nsweep_"
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, dtype in zip(axes, DTYPE_ORDER):
        by_impl: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for r in rows:
            if r["dtype"] != dtype or not r["case_name"].startswith(sweep_prefix):
                continue
            by_impl[r["impl"]].append((_f(r, "num_nodes"), _f(r, "bandwidth_gb_s")))

        for impl in IMPL_ORDER:
            pts = sorted(by_impl.get(impl, []))
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=IMPL_COLOR[impl], linewidth=2, marker="o", markersize=5,
                    label=IMPL_LABEL[impl], zorder=3)

        ax.set_xscale("log")
        ax.set_title(DTYPE_LABEL[dtype], fontsize=11, color=INK_PRIMARY, fontweight="bold")
        ax.set_xlabel("Num nodes (log scale), avg degree~20", fontsize=9, color=INK_SECONDARY)
        _style_axes(ax)

    axes[0].set_ylabel("Achieved bandwidth (GB/s)", fontsize=10, color=INK_PRIMARY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3,
               frameon=False, fontsize=8.5, labelcolor=INK_PRIMARY)
    fig.suptitle("Kernel 10: Bandwidth vs. Node Count", fontsize=13, color=INK_PRIMARY, y=1.16)
    fig.tight_layout()
    _save(fig, "02_bandwidth_vs_nodes")


def plot_bandwidth_vs_degree(rows: list[dict[str, str]]) -> None:
    sweep_prefix = "dsweep_"
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, dtype in zip(axes, DTYPE_ORDER):
        by_impl: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for r in rows:
            if r["dtype"] != dtype or not r["case_name"].startswith(sweep_prefix):
                continue
            avg_degree = _f(r, "num_spatial_edges") / _f(r, "num_nodes")
            by_impl[r["impl"]].append((avg_degree, _f(r, "bandwidth_gb_s")))

        for impl in IMPL_ORDER:
            pts = sorted(by_impl.get(impl, []))
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=IMPL_COLOR[impl], linewidth=2, marker="o", markersize=5,
                    label=IMPL_LABEL[impl], zorder=3)

        ax.set_title(DTYPE_LABEL[dtype], fontsize=11, color=INK_PRIMARY, fontweight="bold")
        ax.set_xlabel("Avg spatial degree, N=50,000", fontsize=9, color=INK_SECONDARY)
        _style_axes(ax)

    axes[0].set_ylabel("Achieved bandwidth (GB/s)", fontsize=10, color=INK_PRIMARY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3,
               frameon=False, fontsize=8.5, labelcolor=INK_PRIMARY)
    fig.suptitle("Kernel 10: Bandwidth vs. Average Node Degree", fontsize=13, color=INK_PRIMARY, y=1.16)
    fig.tight_layout()
    _save(fig, "03_bandwidth_vs_degree")


def plot_bandwidth_vs_peak(rows: list[dict[str, str]]) -> None:
    story_cases = ["nsweep_2000", "nsweep_20000", "nsweep_100000", "nsweep_500000", "sensor_net_realistic"]
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 4.8))
    fig.patch.set_facecolor(SURFACE)

    x = range(len(story_cases))
    width = 0.25
    for i, dtype in enumerate(DTYPE_ORDER):
        by_case = {
            r["case_name"].rsplit("_", 1)[0]: _f(r, "bandwidth_gb_s")
            for r in rows
            if r["dtype"] == dtype and r["impl"] == "cuda_kernel" and r["case_name"].rsplit("_", 1)[0] in story_cases
        }
        ys = [by_case.get(c, 0.0) for c in story_cases]
        offset = (i - 1) * width
        ax.bar([xi + offset for xi in x], ys, width, color=DTYPE_COLOR[dtype],
               label=DTYPE_LABEL[dtype], zorder=3)

    ax.axhline(PEAK_BANDWIDTH_GB_S, color=BASELINE, linewidth=1.2, linestyle="--", zorder=2)
    ax.text(len(story_cases) - 0.5, PEAK_BANDWIDTH_GB_S + 3, "peak (256 GB/s)",
            ha="right", fontsize=8, color=INK_MUTED)

    ax.set_xticks(list(x))
    ax.set_xticklabels(story_cases, fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("Achieved bandwidth (GB/s)", fontsize=10, color=INK_PRIMARY)
    _style_axes(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_PRIMARY)
    ax.set_title("Kernel 10 (CUDA): Bandwidth vs. Peak (256 GB/s, RTX 4070 Laptop)",
                 fontsize=12.5, color=INK_PRIMARY)
    fig.tight_layout()
    _save(fig, "04_bandwidth_vs_peak")


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
        print(f"error: {CSV_PATH} not found — run benchmarks/graph_message_passing_bench.py first", file=sys.stderr)
        sys.exit(1)

    rows = _load(CSV_PATH)
    plot_speedup_bar(rows)
    plot_bandwidth_vs_nodes(rows)
    plot_bandwidth_vs_degree(rows)
    plot_bandwidth_vs_peak(rows)


if __name__ == "__main__":
    main()
