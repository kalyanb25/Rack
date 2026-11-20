"""Tiny terminal helpers: colors, tables, progress. Standard library only."""

from __future__ import annotations

import os
import sys
import time

_COLOR = None


def _use_color() -> bool:
    global _COLOR
    if _COLOR is None:
        if os.environ.get("NO_COLOR") or not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
            _COLOR = False
        else:
            if os.name == "nt":
                os.system("")  # turns on ANSI escape handling in Windows terminals
            _COLOR = True
    return _COLOR


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _use_color() else text


def bold(t: str) -> str:
    return _c("1", t)


def dim(t: str) -> str:
    return _c("2", t)


def green(t: str) -> str:
    return _c("32", t)


def yellow(t: str) -> str:
    return _c("33", t)


def red(t: str) -> str:
    return _c("31", t)


def cyan(t: str) -> str:
    return _c("36", t)


def info(msg: str) -> None:
    print(msg, flush=True)


def ok(msg: str) -> None:
    print(green("✓ ") + msg, flush=True)


def warn(msg: str) -> None:
    print(yellow("! ") + msg, flush=True)


def fail(msg: str) -> None:
    print(red("✗ ") + msg, file=sys.stderr, flush=True)


def step(msg: str) -> None:
    print(cyan("→ ") + msg, flush=True)


def table(rows: list[list[str]], header: list[str]) -> str:
    """Plain aligned table. Cells are converted to str."""
    rows = [[str(c) for c in r] for r in rows]
    widths = [len(h) for h in header]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(_strip_ansi(c)))
    def fmt(r: list[str]) -> str:
        out = []
        for i, c in enumerate(r):
            pad = widths[i] - len(_strip_ansi(c))
            out.append(c + " " * pad)
        return "  ".join(out).rstrip()
    lines = [bold(fmt(header)), dim("  ".join("-" * w for w in widths))]
    lines += [fmt(r) for r in rows]
    return "\n".join(lines)


def _strip_ansi(s: str) -> str:
    import re
    return re.sub(r"\033\[[0-9;]*m", "", s)


def pct(score: float) -> str:
    return f"{int(round(score * 100))}%"


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


class Progress:
    """Single-line progress display for byte transfers (uploads, pulls, hashing)."""

    def __init__(self, label: str, total: int | None = None):
        self.label = label
        self.total = total
        self.done = 0
        self._last = 0.0
        self._tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    def update(self, done: int, total: int | None = None, label: str | None = None) -> None:
        self.done = done
        if total:
            self.total = total
        if label:
            self.label = label
        now = time.monotonic()
        if now - self._last < 0.2 and (not self.total or done < self.total):
            return
        self._last = now
        if not self._tty:
            return
        if self.total:
            pct = min(100, int(100 * done / max(self.total, 1)))
            line = f"  {self.label}  {human_bytes(done)} / {human_bytes(self.total)}  {pct:3d}%"
        else:
            line = f"  {self.label}  {human_bytes(done)}"
        sys.stdout.write("\r" + line[:120].ljust(80))
        sys.stdout.flush()

    def finish(self, msg: str | None = None) -> None:
        if self._tty:
            sys.stdout.write("\r" + " " * 100 + "\r")
            sys.stdout.flush()
        if msg:
            ok(msg)
