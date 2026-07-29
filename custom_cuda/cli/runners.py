"""Subprocess/execution helpers shared by the `verify`, `benchmark`,
`compare`, `visualize`, and `profile` commands. Kept separate from
app.py so the click/typer command bodies stay focused on
orchestration + rendering, not process/IO plumbing.
"""

from __future__ import annotations

import csv
import dataclasses
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

__all__ = [
    "REPO_ROOT",
    "PytestSummary",
    "run_pytest",
    "ScriptRun",
    "run_script",
    "load_csv_rows",
    "ProfileResult",
    "run_profile",
]


@dataclasses.dataclass(frozen=True)
class PytestSummary:
    passed: int
    failed: int
    skipped: int
    errors: int
    duration_s: float
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.skipped + self.errors


# pytest's final summary line joins segments in a fixed internal order
# (failed, passed, skipped, deselected, xfailed, xpassed, warnings,
# error) but that order is an implementation detail, not something to
# hardcode a positional regex against — match each "<N> <label>" token
# independently instead, wherever it falls in the line.
_TOKEN_RE = re.compile(r"(\d+)\s+(passed|failed|skipped|error|errors|xfailed|xpassed)")
_DURATION_RE = re.compile(r"in ([\d.]+)s")


def run_pytest(
    test_files: list[str], extra_args: list[str] | None = None, timeout: int | None = None
) -> PytestSummary:
    existing = [f for f in test_files if (REPO_ROOT / f).exists()]
    cmd = [sys.executable, "-m", "pytest", *existing, "-q", *(extra_args or [])]
    start = time.monotonic()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout)
    elapsed = time.monotonic() - start

    tail = "\n".join(proc.stdout.strip().splitlines()[-8:])
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for count_str, label in _TOKEN_RE.findall(tail):
        n = int(count_str)
        if label in ("error", "errors"):
            counts["errors"] += n
        elif label in ("xfailed",):
            counts["failed"] += n
        elif label in ("xpassed",):
            counts["passed"] += n
        else:
            counts[label] += n
    duration_match = _DURATION_RE.search(tail)
    duration = float(duration_match.group(1)) if duration_match else elapsed

    return PytestSummary(
        passed=counts["passed"], failed=counts["failed"], skipped=counts["skipped"],
        errors=counts["errors"], duration_s=duration, returncode=proc.returncode,
        stdout=proc.stdout, stderr=proc.stderr,
    )


@dataclasses.dataclass(frozen=True)
class ScriptRun:
    returncode: int
    stdout: str
    stderr: str
    duration_s: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run_script(script_path: str, timeout: int | None = None) -> ScriptRun:
    start = time.monotonic()
    proc = subprocess.run(
        [sys.executable, script_path], cwd=REPO_ROOT, capture_output=True, text=True,
        timeout=timeout,
    )
    elapsed = time.monotonic() - start
    return ScriptRun(
        returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr, duration_s=elapsed
    )


def load_csv_rows(csv_path: str) -> list[dict[str, str]]:
    full = REPO_ROOT / csv_path
    if not full.exists():
        return []
    with open(full, newline="") as f:
        return list(csv.DictReader(f))


@dataclasses.dataclass(frozen=True)
class ProfileResult:
    trace_path: Path
    description: str
    warmup_iters: int
    active_iters: int
    top_ops_table: str


def run_profile(
    fn, description: str, output_path: Path, warmup: int = 5, active: int = 10
) -> ProfileResult:
    import torch
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True, with_stack=False) as prof:
        for _ in range(active):
            fn()
        torch.cuda.synchronize()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(output_path))
    sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
    top_ops_table = prof.key_averages().table(sort_by=sort_key, row_limit=15)
    return ProfileResult(
        trace_path=output_path, description=description, warmup_iters=warmup, active_iters=active,
        top_ops_table=top_ops_table,
    )
