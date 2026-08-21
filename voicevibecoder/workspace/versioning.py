"""Git as the long-term memory of a spoken session.

Snapshots (see ``project.py``) exist for "undo that" — fast, in-process, and
disposable. Git exists for the other thing you want after an hour of talking:
a readable history of what you asked for and what came back.

Commits are cheap and unlimited on purpose. Every accepted change is one
commit, so the history reads as the transcript of the session, and any state
you spoke your way into can be recovered later even after the snapshot ring
has rolled over.

Commit messages can be spoken (the model's own summary, the default) or
**typed** — dictating a prose summary is pleasant, spelling a conventional
commit subject out loud is not.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

MAX_SUBJECT = 72
AI_TRAILER = "Assisted-by: VoiceVibeCoder (AI-assisted)"


@dataclass(frozen=True)
class Commit:
    sha: str
    subject: str


class GitError(Exception):
    pass


class GitJournal:
    """A tiny, forgiving git porcelain scoped to one workspace."""

    def __init__(self, root: Path, enabled: bool = True, trailer: bool = True) -> None:
        self.root = Path(root)
        self.trailer = trailer
        self.enabled = enabled and shutil.which("git") is not None
        self.last_error: str = ""
        if self.enabled:
            self.enabled = self._ensure_repo()

    # -- public API ------------------------------------------------------
    def commit(self, subject: str, body: str = "") -> Commit | None:
        """Stage everything and commit. Returns None if there was nothing to do."""
        if not self.enabled:
            return None
        try:
            self._git("add", "-A")
            if not self._git("status", "--porcelain").strip():
                return None
            message = self.compose(subject, body)
            self._git("commit", "-m", message, "--no-verify")
            sha = self._git("rev-parse", "--short", "HEAD").strip()
            return Commit(sha, first_line(message))
        except GitError as exc:
            # Version control is a convenience here; never lose a session over it.
            self.last_error = str(exc)
            self.enabled = False
            return None

    def compose(self, subject: str, body: str = "") -> str:
        """Build the commit message, including the AI-assisted trailer."""
        parts = [truncate_subject(subject)]
        if body.strip():
            parts.append(body.strip())
        if self.trailer:
            parts.append(AI_TRAILER)
        return "\n\n".join(parts)

    def history(self, limit: int = 10) -> list[Commit]:
        if not self.enabled:
            return []
        try:
            raw = self._git("log", f"-{limit}", "--pretty=format:%h\t%s")
        except GitError:
            return []
        commits = []
        for line in raw.splitlines():
            sha, _, subject = line.partition("\t")
            if sha:
                commits.append(Commit(sha, subject))
        return commits

    def revert_last(self) -> Commit | None:
        """Undo the last commit, keeping the working tree as it now is."""
        if not self.enabled or len(self.history(2)) < 2:
            return None
        try:
            head = self.history(1)[0]
            self._git("reset", "--soft", "HEAD~1")
            return head
        except GitError:
            return None

    # -- internals -------------------------------------------------------
    def _ensure_repo(self) -> bool:
        try:
            if not (self.root / ".git").exists():
                self._git("init", "-q")
                self._git("config", "user.name", "VoiceVibeCoder")
                self._git("config", "user.email", "voicevibecoder@localhost")
            return True
        except GitError as exc:
            self.last_error = str(exc)
            return False

    def _git(self, *args: str) -> str:
        result = subprocess.run(  # noqa: S603 — fixed binary, argv never shell-parsed
            ["git", *args],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if result.returncode != 0:
            raise GitError((result.stderr or result.stdout).strip() or f"git {args[0]} failed")
        return result.stdout


def truncate_subject(subject: str) -> str:
    """One line, imperative-ish, short enough for ``git log --oneline``."""
    text = " ".join((subject or "").split()) or "Update workspace"
    text = first_line(text)
    if len(text) <= MAX_SUBJECT:
        return text
    return text[: MAX_SUBJECT - 1].rsplit(" ", 1)[0] + "…"


def first_line(text: str) -> str:
    return text.splitlines()[0].strip() if text.strip() else ""
