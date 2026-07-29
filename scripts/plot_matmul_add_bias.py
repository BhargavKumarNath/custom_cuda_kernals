"""Visualization pipeline for Kernel 5 (Fused MatMul + Add Bias).

Deviates from the standard Section 6 template in the same spirit as
Kernel 4's plot script: this op is compute-bound at realistic sizes, so
chart 3 reports achieved TFLOPS (including PyTorch's cuBLAS-fused
`F.linear` as a fourth series — the honest "how far from a vendor GEMM
library" comparison, see project_plan.md's Kernel 5 entry) rather than
bandwidth-vs-peak.
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
CSV_PATH = REPO_ROOT / "benchmarks" / "matmul_add_bias_results.csv"
OUT_DIR = REPO_ROOT / "visualizations" / "05_fused_matmul_add_bias"

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
IMPL_COLOR = {"eager": BLUE, "f_linear": VIOLET, "compiled": ORANGE, "cuda_kernel": AQUA}
IMPL_LABEL = {
    "eager": "PyTorch eager (unfused)",
    "f_linear": "F.linear (cuBLAS-fused)",
    "compiled": "torch.compile",
    "cuda_kernel": "CUDA kernel",
}
IMPL_ORDER = ["eager", "f_linear", "compiled", "cuda_kernel"]

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


def plot_speedup_bar(rows: list[dict[str, str]]) -> None:
    shapes = ["qkv_proj", "mlp_up", "mlp_down"]
    shape_labels = {
        "qkv_proj": "M=2048,K=4096\nN=4096",
        "mlp_up": "M=2048,K=4096\nN=11008",
        "mlp_down": "M=2048,K=11008\nN=4096",
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
        width = 0.25
        for i, impl in enumerate(["f_linear", "compiled", "cuda_kernel"]):
            speedups = []
            for shape in shapes:
                eager_ms = by_shape_impl.get((shape, "eager"))
                impl_ms = by_shape_impl.get((shape, impl))
                speedups.append(eager_ms / impl_ms if eager_ms and impl_ms else 0.0)
            offset = (i - 1) * width
            bars = ax.bar(
                [xi + offset for xi in x], speedups, width, color=IMPL_COLOR[impl],
                label=IMPL_LABEL[impl], zorder=3,
            )
            for b, v in zip(bars, speedups):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}x", ha="center",
                        va="bottom", fontsize=7.5, color=INK_SECONDARY)

        ax.axhline(1.0, color=BASELINE, linewidth=1, linestyle="--", zorder=2)
        ax.set_ylim(0, 1.35)
        ax.set_xticks(list(x))
        ax.set_xticklabels([shape_labels[s] for s in shapes], fontsize=8, color=INK_SECONDARY)
        ax.set_title(DTYPE_LABEL[dtype], fontsize=11, color=INK_PRIMARY, fontweight="bold")
        _style_axes(ax)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Speedup vs. PyTorch eager (unfused, x)", fontsize=10, color=INK_PRIMARY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3,
               frameon=False, fontsize=9, labelcolor=INK_PRIMARY)
    fig.suptitle("Kernel 5: Fused MatMul + Bias — Speedup vs. Eager", fontsize=13,
                 color=INK_PRIMARY, y=1.20)
    fig.tight_layout()
    _save(fig, "01_speedup_bar")


def plot_latency_vs_m(rows: list[dict[str, str]]) -> None:
    sweep_prefix = "msweep_"
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, dtype in zip(axes, DTYPE_ORDER):
        by_impl: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for r in rows:
            if r["dtype"] != dtype or not r["case_name"].startswith(sweep_prefix):
                continue
            by_impl[r["impl"]].append((_f(r, "m"), _f(r, "median_ms")))

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
        ax.set_xlabel("M (log scale), K=N=4096", fontsize=9, color=INK_SECONDARY)
        _style_axes(ax)

    axes[0].set_ylabel("Median latency (ms, log scale)", fontsize=10, color=INK_PRIMARY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2,
               frameon=False, fontsize=8.5, labelcolor=INK_PRIMARY)
    fig.suptitle("Kernel 5: Execution Latency vs. M (log-log)", fontsize=13,
                 color=INK_PRIMARY, y=1.16)
    fig.tight_layout()
    _save(fig, "02_latency_vs_m")


def plot_tflops(rows: list[dict[str, str]]) -> None:
    sweep_prefix = "msweep_"
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, dtype in zip(axes, DTYPE_ORDER):
        for impl in IMPL_ORDER:
            xs, ys = [], []
            for r in rows:
                if r["dtype"] != dtype or r["impl"] != impl or not r["case_name"].startswith(sweep_prefix):
                    continue
                xs.append(_f(r, "m"))
                ys.append(_f(r, "tflops"))
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            xs, ys = [xs[i] for i in order], [ys[i] for i in order]
            ax.plot(xs, ys, color=IMPL_COLOR[impl], linewidth=2, marker="o", markersize=5,
                    label=IMPL_LABEL[impl], zorder=3)

        ax.set_xscale("log")
        ax.set_title(DTYPE_LABEL[dtype], fontsize=11, color=INK_PRIMARY, fontweight="bold")
        ax.set_xlabel("M (log scale), K=N=4096", fontsize=9, color=INK_SECONDARY)
        _style_axes(ax)

    axes[0].set_ylabel("Achieved TFLOPS", fontsize=10, color=INK_PRIMARY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2,
               frameon=False, fontsize=8.5, labelcolor=INK_PRIMARY)
    fig.suptitle("Kernel 5: Compute Throughput (TFLOPS) — incl. cuBLAS reference", fontsize=13,
                 color=INK_PRIMARY, y=1.16)
    fig.tight_layout()
    _save(fig, "03_tflops")


def plot_scaling(rows: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    fig.patch.set_facecolor(SURFACE)

    for dtype in DTYPE_ORDER:
        pts = sorted(
            (_f(r, "m"), _f(r, "median_ms"))
            for r in rows
            if r["dtype"] == dtype and r["impl"] == "cuda_kernel" and r["case_name"].startswith("msweep_")
        )
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=DTYPE_COLOR[dtype], linewidth=2, marker="o", markersize=6,
                    label=DTYPE_LABEL[dtype], zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("M (log scale), K=N=4096", fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel("Median latency (ms, log scale)", fontsize=10, color=INK_PRIMARY)
    _style_axes(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_PRIMARY)
    ax.set_title("Kernel 5 (CUDA): Scaling vs. M", fontsize=13, color=INK_PRIMARY)
    fig.tight_layout()
    _save(fig, "04_scaling_m")


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
        print(f"error: {CSV_PATH} not found — run benchmarks/matmul_add_bias_bench.py first", file=sys.stderr)
        sys.exit(1)

    rows = _load(CSV_PATH)
    plot_speedup_bar(rows)
    plot_latency_vs_m(rows)
    plot_tflops(rows)
    plot_scaling(rows)


if __name__ == "__main__":
    main()
