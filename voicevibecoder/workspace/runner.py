"""Running the program that was just spoken into existence.

Running is part of the loop, not an afterthought: the fastest way to know
whether "make it sort by date" worked is to watch it work. Everything runs in a
child process, rooted at the workspace, with a wall-clock limit and captured
output, so a runaway ``while True`` costs a few seconds rather than the session.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Output is quoted back to the model on failure; keep it bounded.
MAX_CAPTURE_CHARS = 8000

INTERPRETERS: dict[str, list[str]] = {
    ".py": [sys.executable],
    ".js": ["node"],
    ".mjs": ["node"],
    ".ts": ["npx", "tsx"],
    ".sh": ["bash"],
    ".rb": ["ruby"],
    ".go": ["go", "run"],
}


@dataclass(frozen=True)
class RunResult:
    entrypoint: str
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def summary(self) -> str:
        """One line, suitable for speaking."""
        if self.timed_out:
            return f"{self.entrypoint} was still running after the time limit."
        if self.ok:
            first = next(
                (line for line in self.stdout.splitlines() if line.strip()), ""
            )
            tail = f" It printed: {first}" if first else ""
            return f"{self.entrypoint} ran cleanly in {self.duration_s:.1f} seconds.{tail}"
        return f"{self.entrypoint} exited with code {self.returncode}. {self.error_line()}"

    def error_line(self) -> str:
        """The most informative line of a failure — usually the last one."""
        lines = [line.strip() for line in self.stderr.splitlines() if line.strip()]
        return lines[-1] if lines else "No error output."

    def transcript(self) -> str:
        """Full-ish record handed back to the model when self-healing."""
        parts = [f"$ {self.entrypoint}", f"exit code: {self.returncode}"]
        if self.timed_out:
            parts.append("(timed out)")
        if self.stdout.strip():
            parts.append(f"--- stdout ---\n{self.stdout.strip()}")
        if self.stderr.strip():
            parts.append(f"--- stderr ---\n{self.stderr.strip()}")
        return "\n".join(parts)


class RunnerError(Exception):
    """The entry point cannot be executed at all."""


def command_for(entrypoint: str, root: Path) -> list[str]:
    """Argv for ``entrypoint``, chosen by extension then executable bit."""
    suffix = Path(entrypoint).suffix.lower()
    argv = INTERPRETERS.get(suffix)
    if argv is not None:
        binary = argv[0]
        if binary != sys.executable and shutil.which(binary) is None:
            raise RunnerError(
                f"{entrypoint} needs '{binary}', which is not installed."
            )
        return [*argv, entrypoint]
    if os.access(root / entrypoint, os.X_OK):
        return [f"./{entrypoint}"]
    raise RunnerError(f"I do not know how to run {entrypoint}.")


def run_program(
    root: Path,
    entrypoint: str,
    timeout_s: float = 30.0,
    stdin_text: str = "",
) -> RunResult:
    """Execute ``entrypoint`` inside ``root`` and capture what happened."""
    argv = command_for(entrypoint, root)
    started = time.monotonic()
    try:
        completed = subprocess.run(  # noqa: S603 — argv built from a fixed table
            argv,
            cwd=root,
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except subprocess.TimeoutExpired as expired:
        return RunResult(
            entrypoint=entrypoint,
            returncode=124,
            stdout=_clip(expired.stdout),
            stderr=_clip(expired.stderr),
            duration_s=time.monotonic() - started,
            timed_out=True,
        )
    except OSError as exc:
        raise RunnerError(f"could not start {entrypoint}: {exc}") from exc

    return RunResult(
        entrypoint=entrypoint,
        returncode=completed.returncode,
        stdout=_clip(completed.stdout),
        stderr=_clip(completed.stderr),
        duration_s=time.monotonic() - started,
    )


def _clip(raw: str | bytes | None) -> str:
    if raw is None:
        return ""
    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
    if len(text) <= MAX_CAPTURE_CHARS:
        return text
    half = MAX_CAPTURE_CHARS // 2
    return f"{text[:half]}\n... [{len(text) - MAX_CAPTURE_CHARS} chars trimmed] ...\n{text[-half:]}"
