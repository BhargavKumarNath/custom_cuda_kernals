"""The `custom_cuda_cli` Typer application — commands are thin
orchestration over `registry.py` (what a kernel is) and `runners.py`
(how to run pytest/benchmark/plot/profile for it); rendering lives here
since it's presentation, not execution, logic.
"""

from __future__ import annotations

import difflib
import time
from collections import defaultdict
from pathlib import Path

import typer
from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from custom_cuda import __version__
from custom_cuda.cli.console import BRAND, FAIL, OK, WARN, console, err_console, kernel_title
from custom_cuda.cli.registry import REGISTRY, KernelSpec, get_kernel, list_kernels
from custom_cuda.cli.runners import REPO_ROOT, load_csv_rows, run_pytest, run_script

app = typer.Typer(
    name="custom_cuda_cli",
    help="Developer toolkit for evaluating, benchmarking, and auditing the "
    "custom CUDA kernels in this repository.",
    add_completion=True,
    no_args_is_help=False,
    rich_markup_mode="rich",
)

COMMAND_DESCRIPTIONS: list[tuple[str, str]] = [
    ("list", "List all 12 kernels with their id, phase, and title."),
    ("doctor", "Inspect the local environment (CUDA, nvcc, PyTorch, GPU, Rust extension)."),
    ("info KERNEL", "Show a kernel's technical specification."),
    ("verify [KERNEL]", "Run the correctness test suite (all kernels if omitted)."),
    ("benchmark KERNEL", "Run the hardware benchmark harness and show latency/bandwidth/TFLOPS."),
    ("compare KERNEL", "Show eager vs. compiled vs. CUDA-kernel speedup multipliers."),
    ("visualize KERNEL", "Regenerate this kernel's performance charts."),
    ("profile KERNEL", "Capture a torch.profiler Chrome trace for bottleneck analysis."),
    ("help", "Show this command list."),
]


def _resolve_kernel(kernel_id: str) -> KernelSpec:
    spec = get_kernel(kernel_id)
    if spec is not None:
        return spec
    suggestions = difflib.get_close_matches(kernel_id, REGISTRY.keys(), n=3)
    msg = f"Unknown kernel [bold]{kernel_id!r}[/bold]."
    if suggestions:
        msg += f" Did you mean: {', '.join(suggestions)}?"
    else:
        msg += " Run [cyan]custom_cuda_cli list[/cyan] to see all kernel ids."
    err_console.print(f"[bold red]Error:[/bold red] {msg}")
    raise typer.Exit(code=1)


def _print_banner() -> None:
    console.print(
        Panel(
            f"[{BRAND}]custom_cuda_cli[/{BRAND}] v{__version__} — "
            "developer toolkit for 12 hand-written CUDA kernels\n"
            "[dim]Rust (PyO3) bindings, zero-copy PyTorch tensors, "
            "benchmarked on real hardware.[/dim]",
            box=box.ROUNDED,
            border_style="cyan",
        )
    )


def _print_help_table() -> None:
    table = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan", show_lines=False)
    table.add_column("Command", style="bold")
    table.add_column("Description")
    for cmd, desc in COMMAND_DESCRIPTIONS:
        table.add_row(cmd, desc)
    console.print(table)
    console.print(
        "\n[dim]Run `custom_cuda_cli COMMAND --help` for command-specific options. "
        "Kernel ids: " + ", ".join(sorted(REGISTRY)) + "[/dim]"
    )


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show the CLI version and exit."),
) -> None:
    if version:
        console.print(f"custom_cuda_cli {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        _print_banner()
        _print_help_table()
        raise typer.Exit()


@app.command(name="help")
def help_cmd() -> None:
    """Show all available commands."""
    _print_banner()
    _print_help_table()


@app.command(name="list")
def list_cmd() -> None:
    """List all 12 kernels with their id, phase, and title."""
    table = Table(title="Kernel Registry", box=box.SIMPLE_HEAVY, header_style="bold cyan")
    table.add_column("#", justify="right")
    table.add_column("id", style="bold cyan")
    table.add_column("Title")
    table.add_column("Phase")
    for spec in list_kernels():
        table.add_row(str(spec.number), spec.id, spec.title, spec.phase)
    console.print(table)


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


@app.command()
def doctor() -> None:
    """Inspect the local environment: CUDA toolkit, nvcc, PyTorch, GPU, and
    the Rust extension.
    """
    from custom_cuda.cli.env_checks import run_checks

    console.print(
        Panel(f"[{BRAND}]Environment Doctor[/{BRAND}]", box=box.ROUNDED, border_style="cyan")
    )
    checks = run_checks()

    table = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan")
    table.add_column("Check")
    table.add_column("Status", justify="center")
    table.add_column("Detail")
    status_badge = {"ok": OK, "fail": FAIL, "warn": WARN}
    for c in checks:
        table.add_row(c.name, status_badge[c.status], c.detail)
    console.print(table)

    n_fail = sum(1 for c in checks if c.status == "fail")
    n_warn = sum(1 for c in checks if c.status == "warn")
    if n_fail:
        console.print(
            f"\n[bold red]{n_fail} check(s) failed.[/bold red] Fix these before running kernels."
        )
        raise typer.Exit(code=1)
    if n_warn:
        console.print(
            f"\n[bold yellow]{n_warn} warning(s).[/bold yellow] The toolkit should still work."
        )
    else:
        console.print("\n[bold green]Everything looks good.[/bold green]")


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


@app.command()
def info(kernel: str = typer.Argument(..., help="Kernel id (see `list`).")) -> None:
    """Print a kernel's technical specification."""
    spec = _resolve_kernel(kernel)

    body = Text()
    body.append("Purpose\n", style="bold underline")
    body.append(spec.purpose + "\n\n")
    body.append("Memory / Compute Profile\n", style="bold underline")
    body.append(spec.memory_bound + "\n\n")
    body.append("Supported dtypes\n", style="bold underline")
    body.append(", ".join(spec.dtypes) + "\n\n")
    body.append("Optimization techniques\n", style="bold underline")
    for t in spec.optimization_techniques:
        body.append(f"  • {t}\n")
    body.append("\nTarget success criteria\n", style="bold underline")
    body.append(spec.success_criteria + "\n\n")
    body.append("Files\n", style="bold underline")
    body.append(f"  baseline:   {spec.baseline_module}\n")
    body.append(f"  kernel:     {spec.kernel_module} ({', '.join(spec.kernel_functions)})\n")
    body.append(f"  tests:      {', '.join(spec.test_files)}\n")
    body.append(f"  benchmark:  {spec.bench_script}\n")
    body.append(f"  visualize:  {spec.plot_script}\n")

    console.print(
        Panel(
            body, title=kernel_title(spec.number, spec.title), border_style="cyan",
            box=box.ROUNDED,
        )
    )


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


@app.command()
def verify(
    kernel: str | None = typer.Argument(None, help="Kernel id; omit to verify every kernel."),
    slow: bool = typer.Option(
        False, "--slow", help="Also run @pytest.mark.slow tests (torch.compile checks)."
    ),
    timeout: int = typer.Option(600, "--timeout", help="Max seconds to allow pytest to run."),
) -> None:
    """Run the correctness test suite for one kernel, or all of them."""
    if kernel is None:
        files = ["tests/"]
        title = "all 12 kernels"
    else:
        spec = _resolve_kernel(kernel)
        files = list(spec.test_files)
        title = spec.title

    extra_args = [] if slow else ["-m", "not slow"]
    with console.status(f"[cyan]Running pytest for {title}...[/cyan]"):
        try:
            summary = run_pytest(files, extra_args=extra_args, timeout=timeout)
        except Exception as e:  # subprocess.TimeoutExpired, etc.
            err_console.print(f"[bold red]Error running pytest:[/bold red] {e}")
            raise typer.Exit(code=1) from e

    table = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Passed", f"[bold green]{summary.passed}[/bold green]")
    table.add_row("Failed", f"[bold red]{summary.failed}[/bold red]" if summary.failed else "0")
    table.add_row("Errors", f"[bold red]{summary.errors}[/bold red]" if summary.errors else "0")
    table.add_row("Skipped/deselected", str(summary.skipped))
    table.add_row("Duration", f"{summary.duration_s:.2f}s")
    console.print(table)

    if summary.ok:
        console.print(f"\n[bold green]{OK}[/bold green] — all tests passed for {title}.")
    else:
        console.print(f"\n[bold red]{FAIL}[/bold red] — failures for {title}:\n")
        failure_lines = [
            line for line in summary.stdout.splitlines()
            if line.startswith("FAILED") or line.startswith("ERROR")
        ]
        for line in failure_lines[:25]:
            console.print(f"  [red]{line}[/red]")
        if len(failure_lines) > 25:
            console.print(f"  [dim]... {len(failure_lines) - 25} more[/dim]")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


def _fmt_cell(value: str) -> str:
    try:
        f = float(value)
    except ValueError:
        return value
    return f"{f:.4g}"


_LEFT_ALIGN_COLUMNS = ("impl", "case_name", "dtype", "fp8_format", "granularity")
_IDENTITY_COLUMNS = ("impl", "case_name", "dtype", "fp8_format", "granularity", "k")
_METRIC_PRIORITY = (
    "median_ms", "bandwidth_gb_s", "tflops", "tokens_per_sec", "bandwidth_pct_peak", "iqr_ms",
)


def _compact_columns(columns: list[str]) -> list[str]:
    """Identity columns (impl/case/dtype/...) plus up to 3 metric
    columns, prioritized by `_METRIC_PRIORITY` — full raw shape columns
    (m, n, rows, cols, ...) are dropped from the default view since
    `case_name` already encodes size (e.g. "small_float32"); pass
    `--full` to see every column instead.
    """
    keep = [c for c in _IDENTITY_COLUMNS if c in columns]
    metrics = [c for c in _METRIC_PRIORITY if c in columns][:3]
    return keep + metrics


def _render_csv_table(
    rows: list[dict[str, str]], title: str, max_rows: int = 40, full: bool = False
) -> None:
    if not rows:
        console.print(f"[yellow]No rows in {title}[/yellow]")
        return
    all_columns = list(rows[0].keys())
    columns = all_columns if full else _compact_columns(all_columns)
    table = Table(title=title, box=box.SIMPLE_HEAVY, header_style="bold cyan")
    for col in columns:
        table.add_column(col, justify="left" if col in _LEFT_ALIGN_COLUMNS else "right")
    for row in rows[:max_rows]:
        table.add_row(*(_fmt_cell(row[c]) for c in columns))
    console.print(table)
    if len(rows) > max_rows:
        n_more_rows = len(rows) - max_rows
        console.print(f"[dim]... {n_more_rows} more rows — see the CSV for the full sweep.[/dim]")
    if not full and len(columns) < len(all_columns):
        n_more = len(all_columns) - len(columns)
        console.print(f"[dim]... {n_more} more columns — pass --full to see them.[/dim]")


def _ensure_bench_data(spec: KernelSpec, timeout: int) -> list[list[dict[str, str]]]:
    all_rows = [load_csv_rows(p) for p in spec.bench_result_csvs]
    if all(len(r) > 0 for r in all_rows):
        return all_rows
    console.print(
        f"[cyan]No existing benchmark data for {spec.id} — running {spec.bench_script}...[/cyan]"
    )
    result = run_script(spec.bench_script, timeout=timeout)
    if not result.ok:
        err_console.print(
            f"[bold red]Benchmark script failed (exit {result.returncode}):[/bold red]"
        )
        err_console.print(result.stderr[-3000:])
        raise typer.Exit(code=1)
    return [load_csv_rows(p) for p in spec.bench_result_csvs]


@app.command()
def benchmark(
    kernel: str = typer.Argument(..., help="Kernel id (see `list`)."),
    rerun: bool = typer.Option(
        False, "--rerun", help="Re-run the benchmark even if a CSV already exists."
    ),
    full: bool = typer.Option(
        False, "--full", help="Show every recorded column, not just the key metrics."
    ),
    timeout: int = typer.Option(
        1800, "--timeout", help="Max seconds to allow the benchmark script to run."
    ),
) -> None:
    """Run the hardware-level timing harness and show latency/bandwidth/TFLOPS."""
    spec = _resolve_kernel(kernel)
    _require_cuda()

    if rerun:
        console.print(f"[cyan]Running {spec.bench_script}...[/cyan]")
        result = run_script(spec.bench_script, timeout=timeout)
        if not result.ok:
            err_console.print(
            f"[bold red]Benchmark script failed (exit {result.returncode}):[/bold red]"
        )
            err_console.print(result.stderr[-3000:])
            raise typer.Exit(code=1)
        all_rows = [load_csv_rows(p) for p in spec.bench_result_csvs]
    else:
        all_rows = _ensure_bench_data(spec, timeout=timeout)

    console.print(kernel_title(spec.number, spec.title))
    for csv_path, rows in zip(spec.bench_result_csvs, all_rows):
        _render_csv_table(rows, title=Path(csv_path).name, full=full)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def _render_speedup_table(rows: list[dict[str, str]], title: str) -> None:
    if not rows:
        console.print(f"[yellow]No data for {title}[/yellow]")
        return
    by_case: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for r in rows:
        by_case[r["case_name"]][r["impl"]] = r
    impls_seen = sorted({r["impl"] for r in rows} - {"eager"})
    if not impls_seen:
        console.print(f"[yellow]No non-eager implementations recorded for {title}[/yellow]")
        return

    table = Table(title=title, box=box.SIMPLE_HEAVY, header_style="bold cyan")
    table.add_column("case")
    table.add_column("dtype")
    table.add_column("eager (ms)", justify="right")
    for impl in impls_seen:
        table.add_column(f"{impl} speedup", justify="right")

    for case_name, impl_map in sorted(by_case.items()):
        eager_row = impl_map.get("eager")
        if eager_row is None:
            continue
        eager_ms = float(eager_row["median_ms"])
        cells = [case_name, eager_row.get("dtype", ""), f"{eager_ms:.4g}"]
        for impl in impls_seen:
            r = impl_map.get(impl)
            if r is None:
                cells.append("-")
                continue
            speedup = eager_ms / float(r["median_ms"])
            style = "bold green" if speedup >= 1.0 else "bold red"
            cells.append(f"[{style}]{speedup:.2f}x[/{style}]")
        table.add_row(*cells)
    console.print(table)


@app.command()
def compare(
    kernel: str = typer.Argument(..., help="Kernel id (see `list`)."),
    timeout: int = typer.Option(
        1800, "--timeout", help="Max seconds to allow the benchmark script to run."
    ),
) -> None:
    """Show eager vs. compiled vs. CUDA-kernel speedup multipliers."""
    spec = _resolve_kernel(kernel)
    _require_cuda()
    all_rows = _ensure_bench_data(spec, timeout=timeout)

    console.print(kernel_title(spec.number, spec.title))
    for csv_path, rows in zip(spec.bench_result_csvs, all_rows):
        _render_speedup_table(rows, title=Path(csv_path).name)


# ---------------------------------------------------------------------------
# visualize
# ---------------------------------------------------------------------------


@app.command()
def visualize(
    kernel: str = typer.Argument(..., help="Kernel id (see `list`)."),
    timeout: int = typer.Option(
        1800, "--timeout", help="Max seconds to allow the plot/benchmark scripts to run."
    ),
) -> None:
    """Regenerate this kernel's performance charts under visualizations/."""
    spec = _resolve_kernel(kernel)
    _require_cuda()
    _ensure_bench_data(spec, timeout=timeout)

    console.print(f"[cyan]Running {spec.plot_script}...[/cyan]")
    result = run_script(spec.plot_script, timeout=timeout)
    if not result.ok:
        err_console.print(f"[bold red]Plot script failed (exit {result.returncode}):[/bold red]")
        err_console.print(result.stderr[-3000:])
        raise typer.Exit(code=1)

    written = [
        line.split("wrote ", 1)[1]
        for line in result.stdout.splitlines()
        if line.startswith("wrote ")
    ]
    if written:
        table = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan")
        table.add_column("Generated file")
        for line in written:
            for path in line.split(" and "):
                table.add_row(path.strip())
        console.print(table)
    else:
        console.print(result.stdout)
    console.print(f"\n[bold green]{OK}[/bold green] visualizations updated.")


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------


@app.command()
def profile(
    kernel: str = typer.Argument(..., help="Kernel id (see `list`)."),
    warmup: int = typer.Option(5, "--warmup", help="Warmup iterations before capturing the trace."),
    iters: int = typer.Option(10, "--iters", help="Profiled iterations included in the trace."),
    output: Path | None = typer.Option(  # noqa: B008 - required typer idiom
        None, "--output", "-o", help="Chrome trace output path (.json)."
    ),
) -> None:
    """Capture a torch.profiler Chrome trace for bottleneck analysis."""
    spec = _resolve_kernel(kernel)
    _require_cuda()

    from custom_cuda.cli.runners import run_profile

    console.print(f"[cyan]Building representative inputs for {spec.id}...[/cyan]")
    try:
        target = spec.profile_builder("cuda")
    except Exception as e:
        err_console.print(f"[bold red]Failed to build profile inputs:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    trace_name = f"{spec.id}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path = output or (REPO_ROOT / "profiles" / trace_name)
    status_msg = (
        f"[cyan]Profiling {target.description} "
        f"({warmup} warmup + {iters} traced iters)...[/cyan]"
    )
    with console.status(status_msg):
        result = run_profile(target.fn, target.description, out_path, warmup=warmup, active=iters)

    # torch's own profiler table is pre-formatted, fixed-width, and
    # often wider than the terminal — a Panel would word-wrap it into
    # an unreadable mess, so print it raw (soft_wrap lets each line
    # overflow/scroll natively instead of being rewrapped mid-cell).
    console.rule("[bold cyan]Top ops by CUDA time[/bold cyan]")
    console.print(result.top_ops_table, soft_wrap=True, markup=False, highlight=False)
    console.rule()
    console.print(
        f"[bold green]{OK}[/bold green] trace written to [bold]{result.trace_path}[/bold]"
    )
    console.print("[dim]Open in chrome://tracing or https://ui.perfetto.dev to inspect.[/dim]")


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        err_console.print(
            "[bold red]Error:[/bold red] no CUDA device available. "
            "Run `custom_cuda_cli doctor` to diagnose."
        )
        raise typer.Exit(code=1)
