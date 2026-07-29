"""Visualization pipeline for Kernel 9 (Block Pairwise Distance Matrix).

Like Kernel 5's plot script, chart 3 reports achieved TFLOPS (incl.
`torch.cdist` — the vendor-ish comparison point project_plan.md Section
3.9 names directly) rather than only bandwidth. Chart 4 is specific to
this kernel: Section 3.9's stated success criteria include a bandwidth
target ("≥70% of peak"), unlike Kernel 5's compute-bound spec, so it's
plotted directly against the peak/70% reference lines.
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
CSV_PATH = REPO_ROOT / "benchmarks" / "pairwise_distance_results.csv"
OUT_DIR = REPO_ROOT / "visualizations" / "09_block_pairwise_distance"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
IMPL_COLOR = {"eager": BLUE, "cdist": "#4a3aa7", "compiled": ORANGE, "cuda_kernel": AQUA}
IMPL_LABEL = {
    "eager": "PyTorch eager (formula)",
    "cdist": "torch.cdist ** 2",
    "compiled": "torch.compile",
    "cuda_kernel": "CUDA kernel",
}
IMPL_ORDER = ["eager", "cdist", "compiled", "cuda_kernel"]

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
    shapes = ["small", "medium", "large", "xlarge"]
    shape_labels = {
        "small": "M=N=256\ndim=128",
        "medium": "M=N=1024\ndim=384",
        "large": "M=N=4096\ndim=256",
        "xlarge": "M=N=8192\ndim=128",
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
                cdist_ms = by_shape_impl.get((shape, "cdist"))
                impl_ms = by_shape_impl.get((shape, impl))
                speedups.append(cdist_ms / impl_ms if cdist_ms and impl_ms else 0.0)
            offset = (i - 0.5) * width
            bars = ax.bar(
                [xi + offset for xi in x], speedups, width, color=IMPL_COLOR[impl],
                label=IMPL_LABEL[impl], zorder=3,
            )
            for b, v in zip(bars, speedups):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}x", ha="center",
                        va="bottom", fontsize=7.5, color=INK_SECONDARY)

        ax.axhline(1.0, color=BASELINE, linewidth=1, linestyle="--", zorder=2)
        ax.set_ylim(0, 2.3)
        ax.set_xticks(list(x))
        ax.set_xticklabels([shape_labels[s] for s in shapes], fontsize=8, color=INK_SECONDARY)
        ax.set_title(DTYPE_LABEL[dtype], fontsize=11, color=INK_PRIMARY, fontweight="bold")
        _style_axes(ax)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Speedup vs. torch.cdist**2 (x)", fontsize=10, color=INK_PRIMARY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2,
               frameon=False, fontsize=9, labelcolor=INK_PRIMARY)
    fig.suptitle("Kernel 9: Block Pairwise Distance — Speedup vs. torch.cdist", fontsize=13,
                 color=INK_PRIMARY, y=1.20)
    fig.tight_layout()
    _save(fig, "01_speedup_bar")


def plot_latency_vs_dim(rows: list[dict[str, str]]) -> None:
    sweep_prefix = "dimsweep_"
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, dtype in zip(axes, DTYPE_ORDER):
        by_impl: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for r in rows:
            if r["dtype"] != dtype or not r["case_name"].startswith(sweep_prefix):
                continue
            by_impl[r["impl"]].append((_f(r, "dim"), _f(r, "median_ms")))

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
        ax.set_xlabel("Embedding dim (log scale), M=N=2048", fontsize=9, color=INK_SECONDARY)
        _style_axes(ax)

    axes[0].set_ylabel("Median latency (ms, log scale)", fontsize=10, color=INK_PRIMARY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2,
               frameon=False, fontsize=8.5, labelcolor=INK_PRIMARY)
    fig.suptitle("Kernel 9: Execution Latency vs. Embedding Dim (log-log)", fontsize=13,
                 color=INK_PRIMARY, y=1.16)
    fig.tight_layout()
    _save(fig, "02_latency_vs_dim")


def plot_tflops_vs_dim(rows: list[dict[str, str]]) -> None:
    sweep_prefix = "dimsweep_"
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, dtype in zip(axes, DTYPE_ORDER):
        for impl in IMPL_ORDER:
            xs, ys = [], []
            for r in rows:
                if r["dtype"] != dtype or r["impl"] != impl or not r["case_name"].startswith(sweep_prefix):
                    continue
                xs.append(_f(r, "dim"))
                ys.append(_f(r, "tflops"))
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            xs, ys = [xs[i] for i in order], [ys[i] for i in order]
            ax.plot(xs, ys, color=IMPL_COLOR[impl], linewidth=2, marker="o", markersize=5,
                    label=IMPL_LABEL[impl], zorder=3)

        ax.set_xscale("log")
        ax.set_title(DTYPE_LABEL[dtype], fontsize=11, color=INK_PRIMARY, fontweight="bold")
        ax.set_xlabel("Embedding dim (log scale), M=N=2048", fontsize=9, color=INK_SECONDARY)
        _style_axes(ax)

    axes[0].set_ylabel("Achieved TFLOPS", fontsize=10, color=INK_PRIMARY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2,
               frameon=False, fontsize=8.5, labelcolor=INK_PRIMARY)
    fig.suptitle("Kernel 9: Compute Throughput vs. Dim — cuda_kernel ceiling vs. cuBLAS scaling",
                 fontsize=12.5, color=INK_PRIMARY, y=1.16)
    fig.tight_layout()
    _save(fig, "03_tflops_vs_dim")


def plot_bandwidth_vs_peak(rows: list[dict[str, str]]) -> None:
    story_cases = ["small", "medium", "large", "xlarge", "tall_skinny", "short_wide"]
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.8))
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
    ax.axhline(PEAK_BANDWIDTH_GB_S * 0.7, color=INK_MUTED, linewidth=1, linestyle=":", zorder=2)
    ax.text(len(story_cases) - 0.5, PEAK_BANDWIDTH_GB_S * 0.7 + 3, "70% of peak (target)",
            ha="right", fontsize=8, color=INK_MUTED)

    ax.set_xticks(list(x))
    ax.set_xticklabels(story_cases, fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("Achieved bandwidth (GB/s)", fontsize=10, color=INK_PRIMARY)
    _style_axes(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_PRIMARY)
    ax.set_title("Kernel 9 (CUDA): Bandwidth vs. Peak (256 GB/s, RTX 4070 Laptop)",
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
        print(f"error: {CSV_PATH} not found — run benchmarks/pairwise_distance_bench.py first", file=sys.stderr)
        sys.exit(1)

    rows = _load(CSV_PATH)
    plot_speedup_bar(rows)
    plot_latency_vs_dim(rows)
    plot_tflops_vs_dim(rows)
    plot_bandwidth_vs_peak(rows)


if __name__ == "__main__":
    main()
