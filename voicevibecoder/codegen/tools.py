"""The tools Claude is given, and their execution against a Workspace.

The tool surface is deliberately small and total: read, write, delete, list,
finish. Everything the model can do to the machine goes through
:class:`~voicevibecoder.workspace.project.Workspace`, which enforces
containment — so "the model can only touch the workspace" is a property of one
class rather than a hope about the prompt.

There is no shell tool. Running code is the *user's* verb ("run it"), and
keeping it out of the model's hands means a spoken instruction can never turn
into an unreviewed command on the machine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from voicevibecoder.workspace.project import Workspace, WorkspaceError

TOOLS: list[dict[str, Any]] = [
    {
        "name": "write_file",
        "description": (
            "Create or overwrite a file in the workspace with its complete "
            "contents. Always write the whole file, never a fragment or a diff."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative path, e.g. 'main.py' or 'src/app.js'.",
                },
                "content": {"type": "string", "description": "The full file contents."},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_file",
        "description": "Read a file from the workspace before editing it.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_files",
        "description": "List every file in the workspace with its size.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "delete_file",
        "description": "Delete a file the user no longer wants.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "finish",
        "description": (
            "End the turn. Call this exactly once, after the files are written, "
            "with a summary written to be spoken aloud."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "One or two spoken sentences describing what changed.",
                },
                "entrypoint": {
                    "type": "string",
                    "description": (
                        "The file 'run it' should execute, or an empty string if "
                        "nothing here is runnable."
                    ),
                },
            },
            "required": ["summary", "entrypoint"],
            "additionalProperties": False,
        },
    },
]

# Tools offered when the user asked a question rather than for a change.
READ_ONLY_TOOLS = [
    tool for tool in TOOLS if tool["name"] in ("read_file", "list_files")
]


@dataclass
class ToolOutcome:
    """Result of one tool call, plus the session-level effects it implies."""

    content: str
    is_error: bool = False
    changed: str | None = None   # path written or deleted
    finished: bool = False
    summary: str = ""
    entrypoint: str = ""


class ToolBox:
    """Executes tool calls against one workspace, recording what changed."""

    def __init__(self, workspace: Workspace, read_only: bool = False) -> None:
        self.workspace = workspace
        self.read_only = read_only
        self.changed_files: list[str] = []

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return READ_ONLY_TOOLS if self.read_only else TOOLS

    def execute(self, name: str, raw_input: Any) -> ToolOutcome:
        arguments = raw_input if isinstance(raw_input, dict) else json.loads(raw_input)
        if self.read_only and name not in ("read_file", "list_files"):
            return ToolOutcome(f"{name} is not available while answering a question.", True)
        handler = getattr(self, f"_do_{name}", None)
        if handler is None:
            return ToolOutcome(f"unknown tool: {name}", True)
        try:
            return handler(arguments)
        except WorkspaceError as exc:
            # Errors go back to the model as tool results, not exceptions: a
            # mistyped path should cost one turn, not the whole build.
            return ToolOutcome(str(exc), True)

    # -- handlers --------------------------------------------------------
    def _do_write_file(self, arguments: dict[str, Any]) -> ToolOutcome:
        path = self.workspace.write(arguments["path"], arguments.get("content", ""))
        self._record(path)
        lines = arguments.get("content", "").count("\n") + 1
        return ToolOutcome(f"wrote {path} ({lines} lines)", changed=path)

    def _do_read_file(self, arguments: dict[str, Any]) -> ToolOutcome:
        return ToolOutcome(self.workspace.read(arguments["path"]))

    def _do_list_files(self, _arguments: dict[str, Any]) -> ToolOutcome:
        return ToolOutcome(self.workspace.describe())

    def _do_delete_file(self, arguments: dict[str, Any]) -> ToolOutcome:
        path = self.workspace.delete(arguments["path"])
        self._record(path)
        return ToolOutcome(f"deleted {path}", changed=path)

    def _do_finish(self, arguments: dict[str, Any]) -> ToolOutcome:
        entrypoint = (arguments.get("entrypoint") or "").strip()
        if entrypoint and not self.workspace.exists(entrypoint):
            entrypoint = ""
        return ToolOutcome(
            "acknowledged",
            finished=True,
            summary=(arguments.get("summary") or "").strip(),
            entrypoint=entrypoint,
        )

    def _record(self, path: str) -> None:
        if path not in self.changed_files:
            self.changed_files.append(path)
