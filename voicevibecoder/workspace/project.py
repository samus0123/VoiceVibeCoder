"""The workspace: a directory the assistant may write to, and nothing else.

Two invariants hold everything together.

**Containment.** Every path is resolved against the workspace root and rejected
if it lands outside — absolute paths, ``..`` walks and symlinks that point out
of the tree all fail the same way. The model proposes paths; this class decides
whether they exist.

**Reversibility.** Speech is a lossy, interruptible channel: you *will* say
something that comes out wrong. Each batch of edits is wrapped in a snapshot,
so "undo that" is a directory restore rather than an apology.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from voicevibecoder.config import STATE_DIRNAME

MAX_FILE_BYTES = 1_000_000
SNAPSHOT_LIMIT = 25


class WorkspaceError(Exception):
    """A path or file operation that must not be performed."""


@dataclass
class WorkspaceState:
    """Small facts that survive between sessions."""

    entrypoint: str | None = None
    project_name: str | None = None
    undo_stack: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "entrypoint": self.entrypoint,
                "project_name": self.project_name,
                "undo_stack": self.undo_stack,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, raw: str) -> WorkspaceState:
        data = json.loads(raw)
        return cls(
            entrypoint=data.get("entrypoint"),
            project_name=data.get("project_name"),
            undo_stack=list(data.get("undo_stack") or []),
        )


class Workspace:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_dir = self.root / STATE_DIRNAME
        self.snapshot_dir = self.state_dir / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self.state_dir / "state.json"
        self.state = self._load_state()

    # -- paths -----------------------------------------------------------
    def resolve(self, relative: str) -> Path:
        """Resolve a user/model supplied path inside the workspace, or raise."""
        text = (relative or "").strip()
        if not text:
            raise WorkspaceError("no file name given")
        candidate = Path(text)
        if candidate.is_absolute() or text.startswith("~"):
            raise WorkspaceError(f"absolute paths are not allowed: {text}")

        resolved = (self.root / candidate).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise WorkspaceError(f"path escapes the workspace: {text}")
        if self.state_dir == resolved or self.state_dir in resolved.parents:
            raise WorkspaceError(f"{STATE_DIRNAME}/ is managed by VoiceVibeCoder")
        return resolved

    def relative(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.root))

    # -- files -----------------------------------------------------------
    def write(self, relative: str, content: str) -> str:
        path = self.resolve(relative)
        data = content.encode("utf-8")
        if len(data) > MAX_FILE_BYTES:
            raise WorkspaceError(
                f"{relative} is {len(data)} bytes; the limit is {MAX_FILE_BYTES}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return self.relative(path)

    def read(self, relative: str) -> str:
        path = self.resolve(relative)
        if not path.is_file():
            raise WorkspaceError(f"no such file: {relative}")
        try:
            return path.read_text("utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceError(f"{relative} is not text") from exc

    def append(self, relative: str, content: str) -> str:
        existing = self.read(relative) if self.exists(relative) else ""
        if existing and not existing.endswith("\n"):
            existing += "\n"
        return self.write(relative, existing + content)

    def delete(self, relative: str) -> str:
        path = self.resolve(relative)
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        else:
            raise WorkspaceError(f"no such file: {relative}")
        if self.state.entrypoint == self.relative(path):
            self.state.entrypoint = None
            self._save_state()
        return self.relative(path)

    def exists(self, relative: str) -> bool:
        try:
            return self.resolve(relative).exists()
        except WorkspaceError:
            return False

    def files(self) -> list[str]:
        """Every tracked file, workspace-relative, sorted."""
        found = []
        for path in self.root.rglob("*"):
            if path.is_file() and not self._is_internal(path):
                found.append(str(path.relative_to(self.root)))
        return sorted(found)

    def _is_internal(self, path: Path) -> bool:
        parts = path.relative_to(self.root).parts
        return bool(parts) and (
            parts[0] == STATE_DIRNAME
            or any(part in ("__pycache__", ".git", "node_modules") for part in parts)
        )

    # -- entry point -----------------------------------------------------
    def set_entrypoint(self, relative: str | None) -> None:
        self.state.entrypoint = (
            self.relative(self.resolve(relative)) if relative else None
        )
        self._save_state()

    def guess_entrypoint(self) -> str | None:
        """Best guess at what "run it" should mean."""
        if self.state.entrypoint and self.exists(self.state.entrypoint):
            return self.state.entrypoint
        files = self.files()
        for preferred in ("main.py", "app.py", "index.js", "main.js", "run.sh"):
            if preferred in files:
                return preferred
        scripts = [f for f in files if f.endswith((".py", ".js", ".sh"))]
        # Shallowest path wins: a top-level script beats one nested in tests/.
        scripts.sort(key=lambda f: (f.count("/"), len(f)))
        return scripts[0] if scripts else None

    # -- snapshots -------------------------------------------------------
    def snapshot(self, label: str = "") -> str:
        """Copy the current tree aside and return the snapshot id."""
        snapshot_id = f"{time.time():.6f}"
        target = self.snapshot_dir / snapshot_id
        target.mkdir(parents=True)
        for relative in self.files():
            source = self.root / relative
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        (target / "label.txt").write_text(label, "utf-8")
        self.state.undo_stack.append(snapshot_id)
        self._prune_snapshots()
        self._save_state()
        return snapshot_id

    def undo(self) -> str | None:
        """Restore the most recent snapshot. Returns its label, or None."""
        if not self.state.undo_stack:
            return None
        snapshot_id = self.state.undo_stack.pop()
        source = self.snapshot_dir / snapshot_id
        if not source.is_dir():
            self._save_state()
            return None

        for relative in self.files():  # clear the working tree first
            (self.root / relative).unlink()
        label = ""
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            if str(relative) == "label.txt":
                label = path.read_text("utf-8")
                continue
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)

        shutil.rmtree(source, ignore_errors=True)
        self._prune_empty_dirs()
        self._save_state()
        return label or snapshot_id

    def _prune_snapshots(self) -> None:
        while len(self.state.undo_stack) > SNAPSHOT_LIMIT:
            oldest = self.state.undo_stack.pop(0)
            shutil.rmtree(self.snapshot_dir / oldest, ignore_errors=True)

    def _prune_empty_dirs(self) -> None:
        for path, dirnames, filenames in os.walk(self.root, topdown=False):
            directory = Path(path)
            if directory in (self.root, self.state_dir):
                continue
            if self.state_dir in directory.parents:
                continue
            if not dirnames and not filenames:
                directory.rmdir()

    # -- persistence -----------------------------------------------------
    def _load_state(self) -> WorkspaceState:
        if self._state_file.is_file():
            try:
                return WorkspaceState.from_json(self._state_file.read_text("utf-8"))
            except (ValueError, OSError):
                pass  # a corrupt state file is not worth failing a session over
        return WorkspaceState()

    def _save_state(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._state_file.write_text(self.state.to_json(), "utf-8")

    def describe(self) -> str:
        """Compact inventory used both aloud and as model context."""
        files = self.files()
        if not files:
            return "The workspace is empty."
        lines = [f"{len(files)} file(s) in {self.root.name}:"]
        for relative in files:
            size = (self.root / relative).stat().st_size
            marker = " (entry point)" if relative == self.state.entrypoint else ""
            lines.append(f"  {relative} — {size} bytes{marker}")
        return "\n".join(lines)
