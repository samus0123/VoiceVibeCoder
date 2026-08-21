"""Terminal output.

A voice tool still has a screen, and the screen's job is different from the
speaker's: it shows what was *heard* (so a bad transcript is obvious at a
glance), what was written, and what the program printed. No dependencies —
colour degrades to plain text when the output is not a terminal.
"""

from __future__ import annotations

import sys
from typing import TextIO

RESET = "\033[0m"
STYLES = {
    "heard": "\033[38;5;244m",   # grey: what the microphone got
    "user": "\033[38;5;39m",     # blue: what we understood
    "vibe": "\033[38;5;213m",    # pink: what the assistant says
    "file": "\033[38;5;114m",    # green: files written
    "warn": "\033[38;5;214m",    # amber
    "error": "\033[38;5;203m",   # red
    "dim": "\033[38;5;240m",
}

BANNER = r"""
 ___ _____ _____ _____ _____ _____ _____ _____ _____
|  V| o | i | c | e |V| i | b | e |C| o | d | e | r |
 ‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾ ‾‾‾‾‾
     say what you want. it gets written.
"""


class Console:
    def __init__(self, stream: TextIO | None = None, color: bool | None = None) -> None:
        self._stream = stream
        if color is None:
            color = hasattr(self.stream, "isatty") and self.stream.isatty()
        self.color = color

    @property
    def stream(self) -> TextIO:
        """Resolved late, so a redirected stdout is always the one written to."""
        return self._stream if self._stream is not None else sys.stdout

    # -- output ----------------------------------------------------------
    def write(self, text: str, style: str = "") -> None:
        prefix = STYLES.get(style, "") if self.color else ""
        suffix = RESET if prefix else ""
        self.stream.write(f"{prefix}{text}{suffix}\n")
        self.stream.flush()

    def chunk(self, text: str) -> None:
        """Streamed model text, printed without a trailing newline."""
        if self.color:
            self.stream.write(f"{STYLES['dim']}{text}{RESET}")
        else:
            self.stream.write(text)
        self.stream.flush()

    def banner(self) -> None:
        self.write(BANNER, "vibe")

    def heard(self, raw: str, understood: str) -> None:
        self.write(f"  🎙  {raw}", "heard")
        if understood.strip().lower() != raw.strip().lower():
            self.write(f"  ⇢  {understood}", "user")

    def vibe(self, text: str) -> None:
        self.write(f"  ◈  {text}", "vibe")

    def files(self, paths: list[str]) -> None:
        for path in paths:
            self.write(f"  ✎  {path}", "file")

    def warn(self, text: str) -> None:
        self.write(f"  !  {text}", "warn")

    def error(self, text: str) -> None:
        self.write(f"  ✖  {text}", "error")

    def detail(self, text: str) -> None:
        for line in text.splitlines():
            self.write(f"     {line}", "dim")

    # -- input -----------------------------------------------------------
    def ask(self, prompt: str, default: str = "") -> str:
        """Read a typed line (commit messages, confirmations in text mode)."""
        marker = f"  ⌨  {prompt}"
        if default:
            marker += f" [{default}]"
        self.stream.write(f"{marker}\n     > ")
        self.stream.flush()
        try:
            answer = input().strip()
        except EOFError:
            return default
        return answer or default
