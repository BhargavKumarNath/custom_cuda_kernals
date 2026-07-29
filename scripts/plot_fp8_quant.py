"""Visualization pipeline for Kernel 12 (FP8 Dynamic Quantization &
Casting). Bandwidth is plotted against the theoretical single-pass
minimum (one read of x, one write of x_fp8) — see
benchmarks/fp8_quant_bench.py's docstring — so eager's bandwidth-vs-peak
naturally lands far below cuda_kernel's, reflecting its extra full read
of `x`.
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
CSV_PATH = REPO_ROOT / "benchmarks" / "fp8_quant_results.csv"
OUT_DIR = REPO_ROOT / "visualizations" / "12_fp8_dynamic_quant"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
IMPL_COLOR = {"eager": BLUE, "cuda_kernel": AQUA}
IMPL_LABEL = {"eager": "PyTorch eager (2-pass)", "cuda_kernel": "CUDA kernel (fused)"}

GRAN_COLOR = {"tensor": ORANGE, "block": AQUA}
GRAN_LABEL = {"tensor": "tensor-wide scale", "block": "128x128 block scale"}

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
    sizes = ["small", "medium", "large", "xlarge"]
    fmt = "e4m3"
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, dtype in zip(axes, DTYPE_ORDER):
        by_size_impl_gran = {
            (r["case_name"].split("_")[0], r["impl"], r["granularity"]): _f(r, "median_ms")
            for r in rows
            if r["dtype"] == dtype and r["fp8_format"] == fmt
        }

        x = range(len(sizes))
        width = 0.35
        for i, gran in enumerate(["tensor", "block"]):
            speedups = []
            for size in sizes:
                eager_ms = by_size_impl_gran.get((size, "eager", gran))
                kernel_ms = by_size_impl_gran.get((size, "cuda_kernel", gran))
                speedups.append(eager_ms / kernel_ms if eager_ms and kernel_ms else 0.0)
            offset = (i - 0.5) * width
            bars = ax.bar([xi + offset for xi in x], speedups, width, color=GRAN_COLOR[gran],
                          label=GRAN_LABEL[gran], zorder=3)
            for b, v in zip(bars, speedups):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}x", ha="center",
                        va="bottom", fontsize=7.5, color=INK_SECONDARY)

        ax.axhline(1.0, color=BASELINE, linewidth=1, linestyle="--", zorder=2)
        ax.axhline(2.0, color=INK_MUTED, linewidth=1, linestyle=":", zorder=2)
        ax.set_xticks(list(x))
        ax.set_xticklabels(sizes, fontsize=9, color=INK_SECONDARY)
        ax.set_title(DTYPE_LABEL[dtype], fontsize=11, color=INK_PRIMARY, fontweight="bold")
        _style_axes(ax)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Speedup vs. PyTorch eager 2-pass (x)", fontsize=10, color=INK_PRIMARY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2,
               frameon=False, fontsize=9, labelcolor=INK_PRIMARY)
    fig.suptitle("Kernel 12: FP8 Quant (e4m3) — Speedup vs. Eager (dotted line = 2x target)",
                 fontsize=12.5, color=INK_PRIMARY, y=1.20)
    fig.tight_layout()
    _save(fig, "01_speedup_bar")


def _plot_bandwidth_vs_size(rows: list[dict[str, str]], granularity: str, chart_name: str, title: str) -> None:
    fmt = "e4m3"
    size_order = {"small": 256, "medium": 1024, "large": 4096, "xlarge": 8192}
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, dtype in zip(axes, DTYPE_ORDER):
        by_impl: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for r in rows:
            if r["dtype"] != dtype or r["fp8_format"] != fmt or r["granularity"] != granularity:
                continue
            size_key = r["case_name"].split("_")[0]
            by_impl[r["impl"]].append((size_order[size_key], _f(r, "bandwidth_gb_s")))

        for impl in ["eager", "cuda_kernel"]:
            pts = sorted(by_impl.get(impl, []))
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=IMPL_COLOR[impl], linewidth=2, marker="o", markersize=5,
                    label=IMPL_LABEL[impl], zorder=3)

        ax.set_xscale("log")
        ax.set_title(DTYPE_LABEL[dtype], fontsize=11, color=INK_PRIMARY, fontweight="bold")
        ax.set_xlabel("M=N (log scale)", fontsize=9, color=INK_SECONDARY)
        _style_axes(ax)

    axes[0].set_ylabel("Bandwidth vs. ideal single-pass (GB/s)", fontsize=9.5, color=INK_PRIMARY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2,
               frameon=False, fontsize=8.5, labelcolor=INK_PRIMARY)
    fig.suptitle(title, fontsize=12.5, color=INK_PRIMARY, y=1.18)
    fig.tight_layout()
    _save(fig, chart_name)


def plot_bandwidth_vs_peak(rows: list[dict[str, str]]) -> None:
    sizes = ["small", "medium", "large", "xlarge"]
    fmt = "e4m3"
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.8))
    fig.patch.set_facecolor(SURFACE)

    x = range(len(sizes))
    width = 0.25
    for i, dtype in enumerate(DTYPE_ORDER):
        by_size = {
            r["case_name"].split("_")[0]: _f(r, "bandwidth_gb_s")
            for r in rows
            if r["dtype"] == dtype and r["impl"] == "cuda_kernel" and r["fp8_format"] == fmt
            and r["granularity"] == "block"
        }
        ys = [by_size.get(s, 0.0) for s in sizes]
        offset = (i - 1) * width
        ax.bar([xi + offset for xi in x], ys, width, color=DTYPE_COLOR[dtype],
               label=DTYPE_LABEL[dtype], zorder=3)

    ax.axhline(PEAK_BANDWIDTH_GB_S, color=BASELINE, linewidth=1.2, linestyle="--", zorder=2)
    ax.text(len(sizes) - 0.5, PEAK_BANDWIDTH_GB_S + 3, "peak (256 GB/s)", ha="right", fontsize=8, color=INK_MUTED)

    ax.set_xticks(list(x))
    ax.set_xticklabels(sizes, fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("Bandwidth vs. ideal single-pass (GB/s)", fontsize=9.5, color=INK_PRIMARY)
    _style_axes(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_PRIMARY)
    ax.set_title("Kernel 12 (CUDA, block granularity): Bandwidth vs. Peak (256 GB/s, RTX 4070 Laptop)",
                 fontsize=11.5, color=INK_PRIMARY)
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
        print(f"error: {CSV_PATH} not found — run benchmarks/fp8_quant_bench.py first", file=sys.stderr)
        sys.exit(1)

    rows = _load(CSV_PATH)
    plot_speedup_bar(rows)
    _plot_bandwidth_vs_size(rows, "block", "02_bandwidth_vs_size_block", "Kernel 12: Bandwidth vs. Size (block granularity)")
    _plot_bandwidth_vs_size(rows, "tensor", "03_bandwidth_vs_size_tensor", "Kernel 12: Bandwidth vs. Size (tensor granularity)")
    plot_bandwidth_vs_peak(rows)


if __name__ == "__main__":
    main()
