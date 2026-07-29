"""`custom_cuda_cli` — the developer toolkit entry point (see
project.scripts in pyproject.toml). Import of `app` is deferred into
`main()` so `python -c "import custom_cuda"` (and anything else that
merely imports the `custom_cuda` package without wanting the CLI) never
pays for typer/rich.
"""

from __future__ import annotations

__all__ = ["main"]


def _force_utf8_streams() -> None:
    """Windows' default stdout/stderr encoding follows the system
    locale (often cp1252), not UTF-8 — this CLI's output (rich's
    box-drawing borders, em-dashes, bullets) then raises
    UnicodeEncodeError the moment output isn't a live console that
    negotiates UTF-8 itself (observed with output redirected to a file
    under Git Bash/MSYS specifically, but the underlying cause isn't
    shell-specific). `errors="replace"` means a still-unencodable
    character degrades to a placeholder glyph instead of crashing.
    """
    import sys

    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main() -> None:
    _force_utf8_streams()
    from custom_cuda.cli.app import app

    app()
