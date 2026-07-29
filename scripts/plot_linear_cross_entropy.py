"""Visualization pipeline for Kernel 4 (Fused Linear Cross Entropy Loss).

Deviates from the standard 4-chart Section 6 template (documented here,
same pattern as the benchmark script's deviations): this kernel's story is
latency *and* memory, not bandwidth, so:
  1. Speedup bar chart — same as every other kernel.
  2. Latency vs. vocab size (log-log) — vocab size is the size axis that
     matters for this kernel, not raw element count.
  3. **Memory savings** (replaces "bandwidth vs. peak"): eager vs. kernel
     peak incremental VRAM across realistic training-batch shapes — this
     is the headline result per the kernel's special instructions.
  4. **Chunk-size trade-off** (replaces "scaling across seq/batch"):
     latency and memory as a function of chunk_size, from the same sweep
     — shows why chunk_size is a tunable dial, not a fixed constant.

Run directly: `python scripts/plot_linear_cross_entropy.py` (after
benchmarks/linear_cross_entropy_bench.py has produced its three CSVs).
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
LATENCY_CSV = REPO_ROOT / "benchmarks" / "linear_cross_entropy_results.csv"
MEMORY_CSV = REPO_ROOT / "benchmarks" / "linear_cross_entropy_memory.csv"
SWEEP_CSV = REPO_ROOT / "benchmarks" / "linear_cross_entropy_chunk_sweep.csv"
OUT_DIR = REPO_ROOT / "visualizations" / "04_fused_linear_cross_entropy"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
IMPL_COLOR = {"eager": BLUE, "compiled": ORANGE, "cuda_kernel": AQUA}
IMPL_LABEL = {"eager": "PyTorch eager", "compiled": "torch.compile", "cuda_kernel": "CUDA kernel"}
IMPL_ORDER = ["eager", "compiled", "cuda_kernel"]

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
    shapes = ["small_vocab", "llama2_vocab", "llama3_vocab"]
    shape_labels = {
        "small_vocab": "V=8,000\nN=512",
        "llama2_vocab": "V=32,000\nN=512",
        "llama3_vocab": "V=128,256\nN=512",
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
                ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}x", ha="center",
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
    fig.suptitle("Kernel 4: Fused Linear Cross Entropy — Speedup vs. Eager", fontsize=13,
                 color=INK_PRIMARY, y=1.12)
    fig.tight_layout()
    _save(fig, "01_speedup_bar")


def plot_latency_vs_vocab(rows: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))
    fig.patch.set_facecolor(SURFACE)

    for impl in IMPL_ORDER:
        for dtype in DTYPE_ORDER:
            pts = sorted(
                (_f(r, "vocab_size"), _f(r, "median_ms"))
                for r in rows
                if r["impl"] == impl and r["dtype"] == dtype
            )
            if not pts:
                continue
            xs, ys = zip(*pts)
            linestyle = {"torch.float32": "-", "torch.float16": "--", "torch.bfloat16": ":"}[dtype]
            ax.plot(xs, ys, color=IMPL_COLOR[impl], linewidth=2, marker="o", markersize=5,
                    linestyle=linestyle, label=f"{IMPL_LABEL[impl]} ({DTYPE_LABEL[dtype]})", zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Vocab size (log scale)", fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel("Median latency (ms, log scale)", fontsize=10, color=INK_PRIMARY)
    _style_axes(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5, labelcolor=INK_PRIMARY, ncol=1)
    ax.set_title("Kernel 4: Latency vs. Vocab Size", fontsize=13, color=INK_PRIMARY)
    fig.tight_layout()
    _save(fig, "02_latency_vs_vocab")


def plot_memory_savings(rows: list[dict[str, str]]) -> None:
    cases = sorted({r["case_name"] for r in rows})
    case_labels = {r["case_name"]: f"N={int(_f(r,'n_tokens')):,}\nV={int(_f(r,'vocab_size')):,}" for r in rows}

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig.patch.set_facecolor(SURFACE)

    by_case_impl: dict[tuple[str, str], float] = {(r["case_name"], r["impl"]): _f(r, "peak_incremental_mb") for r in rows}

    x = range(len(cases))
    width = 0.35
    for i, impl in enumerate(["eager", "cuda_kernel"]):
        vals = [by_case_impl.get((c, impl), 0.0) for c in cases]
        offset = (i - 0.5) * width
        bars = ax.bar([xi + offset for xi in x], vals, width, color=IMPL_COLOR[impl],
                      label=IMPL_LABEL[impl], zorder=3)
        for b, v in zip(bars, vals):
            label = f"{v/1024:.2f} GB" if v >= 1024 else f"{v:.0f} MB"
            ax.text(b.get_x() + b.get_width() / 2, v, label, ha="center", va="bottom",
                    fontsize=8, color=INK_SECONDARY)

    for i, c in enumerate(cases):
        eager_v = by_case_impl.get((c, "eager"), 0.0)
        kernel_v = by_case_impl.get((c, "cuda_kernel"), 0.0)
        if kernel_v > 0:
            ax.text(i, max(eager_v, kernel_v) * 1.12, f"{eager_v/kernel_v:.1f}x less",
                    ha="center", va="bottom", fontsize=9, color=INK_PRIMARY, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels([case_labels[c] for c in cases], fontsize=8.5, color=INK_SECONDARY)
    ax.set_ylabel("Peak incremental VRAM (MB, log scale)", fontsize=10, color=INK_PRIMARY)
    ax.set_yscale("log")
    _style_axes(ax)
    ax.grid(axis="x", visible=False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2,
               frameon=False, fontsize=9, labelcolor=INK_PRIMARY)
    fig.suptitle("Kernel 4: Peak Incremental VRAM — fp16, chunk_size=4096", fontsize=13,
                 color=INK_PRIMARY, y=1.10)
    fig.tight_layout()
    _save(fig, "03_memory_savings")


def plot_chunk_sweep(rows: list[dict[str, str]]) -> None:
    fig, (ax_lat, ax_mem) = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.patch.set_facecolor(SURFACE)

    pts = sorted((int(_f(r, "chunk_size")), _f(r, "median_ms")) for r in rows)
    xs, ys = zip(*pts)
    ax_lat.plot(xs, ys, color=AQUA, linewidth=2, marker="o", markersize=6, zorder=3)
    ax_lat.set_xscale("log")
    # Zero-floored on purpose: chunk_size has only a few-percent effect on
    # latency here (see project_plan.md's Kernel 4 entry) — an
    # auto-scaled axis would visually inflate that noise-level wobble into
    # a misleadingly dramatic curve.
    ax_lat.set_ylim(0, max(ys) * 1.15)
    ax_lat.set_xlabel("chunk_size (log scale)", fontsize=9, color=INK_SECONDARY)
    ax_lat.set_ylabel("Median latency (ms)", fontsize=10, color=INK_PRIMARY)
    ax_lat.set_title("Latency vs. chunk_size", fontsize=10.5, color=INK_PRIMARY)
    _style_axes(ax_lat)

    pts_mem = sorted((int(_f(r, "chunk_size")), _f(r, "peak_incremental_mb")) for r in rows)
    xs_m, ys_m = zip(*pts_mem)
    ax_mem.plot(xs_m, ys_m, color=ORANGE, linewidth=2, marker="o", markersize=6, zorder=3)
    ax_mem.set_xscale("log")
    ax_mem.set_yscale("log")
    ax_mem.set_xlabel("chunk_size (log scale)", fontsize=9, color=INK_SECONDARY)
    ax_mem.set_ylabel("Peak incremental VRAM (MB, log scale)", fontsize=9, color=INK_PRIMARY)
    ax_mem.set_title("Memory vs. chunk_size", fontsize=10.5, color=INK_PRIMARY)
    _style_axes(ax_mem)

    fig.suptitle(
        "Kernel 4: chunk_size Trade-off (N=2048, V=128256, H=4096, fp16)", fontsize=13,
        color=INK_PRIMARY, y=1.05,
    )
    fig.tight_layout()
    _save(fig, "04_chunk_size_tradeoff")


def _save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{name}.png"
    svg_path = OUT_DIR / f"{name}.svg"
    fig.savefig(png_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(svg_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path.relative_to(REPO_ROOT)} and {svg_path.relative_to(REPO_ROOT)}")


def main() -> None:
    for path in (LATENCY_CSV, MEMORY_CSV, SWEEP_CSV):
        if not path.exists():
            print(f"error: {path} not found — run benchmarks/linear_cross_entropy_bench.py first", file=sys.stderr)
            sys.exit(1)

    latency_rows = _load(LATENCY_CSV)
    memory_rows = _load(MEMORY_CSV)
    sweep_rows = _load(SWEEP_CSV)

    plot_speedup_bar(latency_rows)
    plot_latency_vs_vocab(latency_rows)
    plot_memory_savings(memory_rows)
    plot_chunk_sweep(sweep_rows)


if __name__ == "__main__":
    main()
