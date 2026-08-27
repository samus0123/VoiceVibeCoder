"""The agentic loop: instruction in, files on disk, summary out.

The loop lives here and nowhere else. It does not know which brain it is
driving — Claude in a datacentre or a quantised model on this laptop — because
everything provider-shaped is behind
:class:`~voicevibecoder.codegen.brain.Brain`. What it does own is the part that
must be identical either way: the tool box, what counts as finished, what
changed on disk, and the summary that gets spoken when nobody said one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from voicevibecoder.codegen import prompts
from voicevibecoder.codegen.brain import Brain, ToolResult, build_brain
from voicevibecoder.codegen.ideas import (
    IDEA_SCHEMA,
    IDEA_SYSTEM,
    IMPROVE_SYSTEM,
    Idea,
    improvement_request_text,
    parse_ideas,
    request_text,
)
from voicevibecoder.codegen.tools import ToolBox
from voicevibecoder.config import Config
from voicevibecoder.workspace.project import Workspace

# A build that has not called finish after this many tool rounds is looping.
MAX_TOOL_ROUNDS = 24


@dataclass
class BuildResult:
    summary: str
    changed_files: list[str] = field(default_factory=list)
    entrypoint: str = ""
    refused: bool = False

    @property
    def wrote_anything(self) -> bool:
        return bool(self.changed_files)


class Generator(Protocol):
    """What the session needs from a code generator (real or fake)."""

    def build(self, instruction: str, workspace: Workspace) -> BuildResult: ...

    def repair(self, failure_transcript: str, workspace: Workspace) -> BuildResult: ...

    def explain(self, question: str, workspace: Workspace) -> str: ...

    def ideate(self, topic: str, workspace: Workspace) -> list[Idea]: ...

    def suggest_improvements(self, summary: str, workspace: Workspace) -> list[Idea]: ...

    def reset(self) -> None: ...


class CodeGenerator:
    """Drives a brain through the tool box until the work is done."""

    def __init__(
        self,
        config: Config,
        brain: Brain | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self._brain = brain
        self._on_text = on_text

    @property
    def brain(self) -> Brain:
        """The brain, connected on first use.

        Deliberately lazy: a session with no model reachable should still
        *start*. Listing files, reading them back, running a program, undoing,
        dictating and quitting need no model at all, and a program that refuses
        to open because a server is down is a worse program than one that says
        so when you finally ask it to build something.
        """
        if self._brain is None:
            self._brain = build_brain(self.config, on_text=self._on_text)
        return self._brain

    @property
    def brain_name(self) -> str:
        """What is connected, without connecting anything to find out."""
        return getattr(self._brain, "name", "not connected yet")

    @property
    def connected(self) -> bool:
        return self._brain is not None

    # -- public API ------------------------------------------------------
    def reset(self) -> None:
        """Forget the conversation — used when starting a new project."""
        self.brain.reset()

    def build(self, instruction: str, workspace: Workspace) -> BuildResult:
        """Carry out a spoken instruction, writing files as a side effect."""
        opening = (
            f"{prompts.workspace_context(workspace.describe())}\n\n"
            f"The person said: {instruction}"
        )
        return self._run_loop(opening, ToolBox(workspace), prompts.SYSTEM_PROMPT)

    def repair(self, failure_transcript: str, workspace: Workspace) -> BuildResult:
        """Hand a failing run straight back to the brain that wrote it."""
        return self._run_loop(
            prompts.repair_prompt(failure_transcript),
            ToolBox(workspace),
            prompts.SYSTEM_PROMPT,
        )

    def explain(self, question: str, workspace: Workspace) -> str:
        """Answer a question about the code without touching it."""
        opening = (
            f"{prompts.workspace_context(workspace.describe())}\n\n"
            f"The person asked: {question}"
        )
        result = self._run_loop(
            opening,
            ToolBox(workspace, read_only=True),
            prompts.EXPLAIN_PROMPT,
            require_finish=False,
        )
        return result.summary

    def ideate(self, topic: str, workspace: Workspace) -> list[Idea]:
        """Propose program ideas that clear the quality bar (or none at all).

        Brainstorming runs outside the build conversation on purpose: it
        should not pollute the context that the next "now add colour" needs.
        """
        payload = self.brain.structured(
            IDEA_SYSTEM, request_text(topic, workspace.describe()), IDEA_SCHEMA
        )
        return parse_ideas(payload, self.config.idea_bar)

    def suggest_improvements(self, summary: str, workspace: Workspace) -> list[Idea]:
        """Look at what was just built and ask what would make it remarkable."""
        payload = self.brain.structured(
            IMPROVE_SYSTEM,
            improvement_request_text(
                summary, workspace.describe(), read_excerpt(workspace)
            ),
            IDEA_SCHEMA,
        )
        return parse_ideas(payload, self.config.idea_bar)

    # -- the loop --------------------------------------------------------
    def _run_loop(
        self,
        opening: str,
        toolbox: ToolBox,
        system_prompt: str,
        require_finish: bool = True,
    ) -> BuildResult:
        result = BuildResult(summary="")
        spoken_text: list[str] = []
        pending: list[ToolResult] = []
        next_message: str | None = opening

        for _round in range(MAX_TOOL_ROUNDS):
            reply = self.brain.turn(
                system_prompt, next_message, pending, toolbox.definitions
            )
            next_message, pending = None, []

            if reply.refused:
                return BuildResult(
                    summary="I was not able to work on that request.", refused=True
                )
            if reply.text:
                spoken_text.append(reply.text)
            if not reply.wants_tools:
                break

            for call in reply.tool_calls:
                outcome = toolbox.execute(call.name, call.arguments)
                pending.append(
                    ToolResult(
                        call_id=call.id,
                        name=call.name,
                        content=outcome.content,
                        is_error=outcome.is_error,
                    )
                )
                if outcome.finished:
                    result.summary = outcome.summary
                    result.entrypoint = outcome.entrypoint

            if result.summary or not require_finish:
                break

        result.changed_files = toolbox.changed_files
        if not result.summary:
            joined = " ".join(text.strip() for text in spoken_text if text.strip())
            result.summary = joined or _fallback_summary(toolbox.changed_files)
        return result


def read_excerpt(workspace: Workspace, budget: int = 12000) -> str:
    """As much of the workspace source as fits in a sensible prompt budget."""
    parts: list[str] = []
    remaining = budget
    for relative in workspace.files():
        if remaining <= 0:
            parts.append("... (further files omitted)")
            break
        try:
            content = workspace.read(relative)
        except Exception:  # noqa: BLE001 — binaries and oddities are skipped
            continue
        if len(content) > remaining:
            content = content[:remaining] + "\n... (truncated)"
        remaining -= len(content)
        parts.append(f"--- {relative} ---\n{content}")
    return "\n\n".join(parts) or "(the workspace is empty)"


def _fallback_summary(changed: list[str]) -> str:
    if not changed:
        return "Nothing changed."
    if len(changed) == 1:
        return f"Updated {changed[0]}."
    return f"Updated {len(changed)} files: {', '.join(changed)}."


__all__ = ["BuildResult", "CodeGenerator", "Generator", "read_excerpt"]
