"""Visualization pipeline for Kernel 7 (Token Scatter/Gather,
Permute-Unpermute).

Two operations, one visualization set: chart 1 is the combine op's
speedup (the headline result — fusing the weighted gather-combine beats
eager's fancy-indexing chain by 3-9x); charts 2-3 show bandwidth-vs-peak
for gather and combine separately (gather is competitive with PyTorch's
highly-optimized `index_select`, not a blowout win, so it gets its own
honest chart rather than being folded into the speedup bar); chart 4
scales both operations' CUDA kernel latency across size.
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
GATHER_CSV = REPO_ROOT / "benchmarks" / "token_permute_gather_results.csv"
COMBINE_CSV = REPO_ROOT / "benchmarks" / "token_permute_combine_results.csv"
OUT_DIR = REPO_ROOT / "visualizations" / "07_token_scatter_gather"

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


def _load(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def plot_combine_speedup_bar(rows: list[dict[str, str]]) -> None:
    shapes = ["mixtral_like", "deepseek_like", "tsweep_8192"]
    shape_labels = {
        "mixtral_like": "T=2048,k=2\nH=4096",
        "deepseek_like": "T=2048,k=6\nH=4096",
        "tsweep_8192": "T=8192,k=2\nH=4096",
    }

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
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
                ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}x", ha="center",
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
    fig.suptitle("Kernel 7: Unpermute (Weighted Combine) — Speedup vs. Eager", fontsize=13,
                 color=INK_PRIMARY, y=1.12)
    fig.tight_layout()
    _save(fig, "01_combine_speedup_bar")


def _plot_bandwidth(rows: list[dict[str, str]], sweep_prefix: str, size_key: str, title: str, name: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, dtype in zip(axes, DTYPE_ORDER):
        for impl in IMPL_ORDER:
            xs, ys = [], []
            for r in rows:
                if r["dtype"] != dtype or r["impl"] != impl or not r["case_name"].startswith(sweep_prefix):
                    continue
                xs.append(_f(r, size_key))
                ys.append(_f(r, "bandwidth_gb_s"))
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            xs, ys = [xs[i] for i in order], [ys[i] for i in order]
            ax.scatter(xs, ys, color=IMPL_COLOR[impl], s=45, label=IMPL_LABEL[impl], zorder=3)
            ax.plot(xs, ys, color=IMPL_COLOR[impl], linewidth=1, alpha=0.5, zorder=2)

        ax.axhline(PEAK_BANDWIDTH_GB_S, color=INK_MUTED, linewidth=1.25, linestyle="--", zorder=2)
        ax.set_xscale("log")
        ax.set_title(DTYPE_LABEL[dtype], fontsize=11, color=INK_PRIMARY, fontweight="bold")
        ax.set_xlabel("Size (log scale)", fontsize=9, color=INK_SECONDARY)
        _style_axes(ax)

    axes[0].set_ylabel("Achieved memory bandwidth (GB/s)", fontsize=10, color=INK_PRIMARY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=3,
               frameon=False, fontsize=9, labelcolor=INK_PRIMARY)
    fig.suptitle(title, fontsize=13, color=INK_PRIMARY, y=1.12)
    fig.tight_layout()
    _save(fig, name)


def plot_gather_bandwidth(rows: list[dict[str, str]]) -> None:
    _plot_bandwidth(
        rows, "nsweep_", "n_dst_rows", "Kernel 7: Permute (Gather) Bandwidth vs. Peak", "02_gather_bandwidth"
    )


def plot_combine_bandwidth(rows: list[dict[str, str]]) -> None:
    _plot_bandwidth(
        rows, "tsweep_", "n_tokens", "Kernel 7: Unpermute (Combine) Bandwidth vs. Peak", "03_combine_bandwidth"
    )


def plot_scaling(gather_rows: list[dict[str, str]], combine_rows: list[dict[str, str]]) -> None:
    fig, (ax_gather, ax_combine) = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.patch.set_facecolor(SURFACE)

    for dtype in DTYPE_ORDER:
        pts = sorted(
            (_f(r, "n_dst_rows"), _f(r, "median_ms"))
            for r in gather_rows
            if r["dtype"] == dtype and r["impl"] == "cuda_kernel" and r["case_name"].startswith("nsweep_")
        )
        if pts:
            xs, ys = zip(*pts)
            ax_gather.plot(xs, ys, color=DTYPE_COLOR[dtype], linewidth=2, marker="o", markersize=6,
                           label=DTYPE_LABEL[dtype], zorder=3)

    ax_gather.set_xscale("log")
    ax_gather.set_yscale("log")
    ax_gather.set_xlabel("Rows gathered (log scale), H=4096", fontsize=9, color=INK_SECONDARY)
    ax_gather.set_ylabel("Median latency (ms, log scale)", fontsize=10, color=INK_PRIMARY)
    ax_gather.set_title("Permute (gather) scaling", fontsize=10.5, color=INK_PRIMARY)
    _style_axes(ax_gather)

    for dtype in DTYPE_ORDER:
        pts = sorted(
            (_f(r, "n_tokens"), _f(r, "median_ms"))
            for r in combine_rows
            if r["dtype"] == dtype and r["impl"] == "cuda_kernel" and r["case_name"].startswith("tsweep_")
        )
        if pts:
            xs, ys = zip(*pts)
            ax_combine.plot(xs, ys, color=DTYPE_COLOR[dtype], linewidth=2, marker="o", markersize=6,
                            label=DTYPE_LABEL[dtype], zorder=3)

    ax_combine.set_xscale("log")
    ax_combine.set_yscale("log")
    ax_combine.set_xlabel("Tokens (log scale), k=2, H=4096", fontsize=9, color=INK_SECONDARY)
    ax_combine.set_title("Unpermute (combine) scaling", fontsize=10.5, color=INK_PRIMARY)
    _style_axes(ax_combine)

    handles, labels = ax_gather.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=3,
               frameon=False, fontsize=9, labelcolor=INK_PRIMARY)
    fig.suptitle("Kernel 7 (CUDA): Scaling — Gather vs. Combine", fontsize=13,
                 color=INK_PRIMARY, y=1.14)
    fig.tight_layout()
    _save(fig, "04_scaling")


def _save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{name}.png"
    svg_path = OUT_DIR / f"{name}.svg"
    fig.savefig(png_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(svg_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path.relative_to(REPO_ROOT)} and {svg_path.relative_to(REPO_ROOT)}")


def main() -> None:
    for path in (GATHER_CSV, COMBINE_CSV):
        if not path.exists():
            print(f"error: {path} not found — run benchmarks/token_permute_bench.py first", file=sys.stderr)
            sys.exit(1)

    gather_rows = _load(GATHER_CSV)
    combine_rows = _load(COMBINE_CSV)

    plot_combine_speedup_bar(combine_rows)
    plot_gather_bandwidth(gather_rows)
    plot_combine_bandwidth(combine_rows)
    plot_scaling(gather_rows, combine_rows)


if __name__ == "__main__":
    main()
