"""Shared rich Console + small styling helpers used across every command."""

from __future__ import annotations

from rich.console import Console

# `legacy_windows=False` forces rich's standard ANSI/UTF-8 rendering path
# instead of the legacy win32-console text API. Without this, some
# Windows shells (observed with Git Bash/MSYS specifically) misreport
# themselves as a legacy console even when output is redirected to a
# file/pipe, and rich's win32 codepath then tries to encode with the
# process's codepage (cp1252) instead of UTF-8 — raising
# UnicodeEncodeError on any non-ASCII character (em-dashes, box-drawing
# borders, spinner glyphs) the moment output isn't a live TTY. Modern
# Windows Terminal/PowerShell/cmd all support ANSI natively, so this is
# the correct setting even on Windows, not a downgrade.
console = Console(legacy_windows=False)
err_console = Console(stderr=True, legacy_windows=False)

OK = "[bold green]OK[/bold green]"
FAIL = "[bold red]FAIL[/bold red]"
WARN = "[bold yellow]WARN[/bold yellow]"

BRAND = "bold cyan"


def kernel_title(number: int, title: str) -> str:
    return f"[{BRAND}]Kernel {number:02d}[/{BRAND}] — {title}"
