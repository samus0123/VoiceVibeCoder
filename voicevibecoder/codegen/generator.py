"""The Claude side of the loop: instruction in, files on disk, summary out.

Three choices worth knowing about:

**One conversation per session.** Follow-ups ("now make it colourful") only
work if the model still remembers what "it" is, so the message history is kept
and trimmed rather than rebuilt per utterance. The system prompt and tool list
are byte-stable across turns, which makes them a cacheable prefix.

**A manual tool loop.** The loop is a dozen lines and owning it means the
workspace can record what changed, veto a path, and stop cleanly at ``finish``.

**Streaming.** Generating three files can take a minute; streaming keeps the
request off the HTTP timeout and lets the console show progress as it happens.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from voicevibecoder.codegen import prompts
from voicevibecoder.codegen.ideas import (
    IDEA_SCHEMA,
    IDEA_SYSTEM,
    Idea,
    parse_ideas,
    request_text,
)
from voicevibecoder.codegen.tools import ToolBox
from voicevibecoder.config import Config
from voicevibecoder.workspace.project import Workspace

# A build that has not called finish after this many tool rounds is looping.
MAX_TOOL_ROUNDS = 24

FALLBACK_BETA = "server-side-fallback-2026-07-01"


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

    def explain(self, question: str, workspace: Workspace) -> str: ...

    def ideate(self, topic: str, workspace: Workspace) -> list[Idea]: ...

    def reset(self) -> None: ...


class CodeGenerator:
    """Claude, wired to a workspace through the tool box."""

    def __init__(
        self,
        config: Config,
        client: Any | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self._client = client or _make_client()
        self._on_text = on_text or (lambda _chunk: None)
        self._history: list[dict[str, Any]] = []
        self._beta_supported = config.server_side_fallback

    # -- public API ------------------------------------------------------
    def reset(self) -> None:
        """Forget the conversation — used when starting a new project."""
        self._history = []

    def build(self, instruction: str, workspace: Workspace) -> BuildResult:
        """Carry out a spoken instruction, writing files as a side effect."""
        toolbox = ToolBox(workspace)
        opening = (
            f"{prompts.workspace_context(workspace.describe())}\n\n"
            f"The person said: {instruction}"
        )
        return self._run_loop(opening, toolbox, prompts.SYSTEM_PROMPT)

    def repair(self, failure_transcript: str, workspace: Workspace) -> BuildResult:
        """Hand a failing run straight back to the model that wrote it."""
        toolbox = ToolBox(workspace)
        return self._run_loop(
            prompts.repair_prompt(failure_transcript), toolbox, prompts.SYSTEM_PROMPT
        )

    def explain(self, question: str, workspace: Workspace) -> str:
        """Answer a question about the code without touching it."""
        toolbox = ToolBox(workspace, read_only=True)
        opening = (
            f"{prompts.workspace_context(workspace.describe())}\n\n"
            f"The person asked: {question}"
        )
        result = self._run_loop(
            opening, toolbox, prompts.EXPLAIN_PROMPT, require_finish=False
        )
        return result.summary

    def ideate(self, topic: str, workspace: Workspace) -> list[Idea]:
        """Propose program ideas that clear the quality bar (or none at all).

        A single structured-output call, deliberately outside the build
        conversation: brainstorming should not pollute the context that the
        next "now add colour" depends on.
        """
        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=8000,
            system=[
                {
                    "type": "text",
                    "text": IDEA_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": request_text(topic, workspace.describe()),
                }
            ],
            thinking={"type": "adaptive"},
            output_config={
                "effort": self.config.effort,
                "format": {"type": "json_schema", "schema": IDEA_SCHEMA},
            },
        )
        if response.stop_reason == "refusal":
            return []
        payload = next(
            (block.text for block in response.content if block.type == "text"), ""
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
        messages = [*self._history, {"role": "user", "content": opening}]
        result = BuildResult(summary="")
        spoken_text: list[str] = []

        for _round in range(MAX_TOOL_ROUNDS):
            response = self._request(messages, toolbox, system_prompt)

            if response.stop_reason == "refusal":
                return BuildResult(
                    summary="I was not able to work on that request.", refused=True
                )

            messages.append({"role": "assistant", "content": response.content})
            spoken_text.extend(
                block.text for block in response.content if block.type == "text"
            )

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                break

            tool_results: list[dict[str, Any]] = []
            for call in tool_uses:
                outcome = toolbox.execute(call.name, call.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": outcome.content,
                        "is_error": outcome.is_error,
                    }
                )
                if outcome.finished:
                    result.summary = outcome.summary
                    result.entrypoint = outcome.entrypoint
            # All results from one assistant turn go back in a single user
            # message, or the model quietly stops calling tools in parallel.
            messages.append({"role": "user", "content": tool_results})

            if result.summary or (not require_finish and not tool_uses):
                break

        result.changed_files = toolbox.changed_files
        if not result.summary:
            joined = " ".join(text.strip() for text in spoken_text if text.strip())
            result.summary = joined or _fallback_summary(toolbox.changed_files)

        self._remember(messages)
        return result

    def _request(
        self, messages: list[dict[str, Any]], toolbox: ToolBox, system_prompt: str
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            # Stable prefix: prompt and tools never vary within a session, so
            # every turn after the first reads them from cache.
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": messages,
            "tools": toolbox.definitions,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.config.effort},
        }
        return self._stream(kwargs)

    def _stream(self, kwargs: dict[str, Any]) -> Any:
        """Stream one request, printing assistant text as it arrives."""
        if self._beta_supported:
            try:
                return self._consume(
                    self._client.beta.messages.stream(
                        **kwargs, betas=[FALLBACK_BETA], fallbacks="default"
                    )
                )
            except TypeError:
                # Older SDK without server-side fallbacks: degrade quietly and
                # do not try again for the rest of the session.
                self._beta_supported = False
        return self._consume(self._client.messages.stream(**kwargs))

    def _consume(self, stream_context: Any) -> Any:
        with stream_context as stream:
            for chunk in stream.text_stream:
                self._on_text(chunk)
            return stream.get_final_message()

    def _remember(self, messages: list[dict[str, Any]]) -> None:
        """Keep the tail of the conversation so follow-ups have context."""
        limit = max(2, self.config.history_turns * 2)
        trimmed = messages[-limit:]
        # Never start the history on a tool result — its tool_use is upstream.
        while trimmed and _starts_with_tool_result(trimmed[0]):
            trimmed = trimmed[1:]
        self._history = trimmed


def _starts_with_tool_result(message: dict[str, Any]) -> bool:
    content = message.get("content")
    if not isinstance(content, list) or not content:
        return False
    first = content[0]
    return isinstance(first, dict) and first.get("type") == "tool_result"


def _fallback_summary(changed: list[str]) -> str:
    if not changed:
        return "Nothing changed."
    if len(changed) == 1:
        return f"Updated {changed[0]}."
    return f"Updated {len(changed)} files: {', '.join(changed)}."


def _make_client() -> Any:
    try:
        import anthropic  # noqa: PLC0415 — optional until a build is requested
    except ImportError as exc:
        raise RuntimeError(
            "Writing code needs the Anthropic SDK:\n  pip install anthropic"
        ) from exc
    try:
        return anthropic.Anthropic()
    except Exception as exc:  # noqa: BLE001 — surface auth setup, not a stack trace
        raise RuntimeError(
            "Could not reach the Claude API. Set ANTHROPIC_API_KEY, or run "
            "'ant auth login' to store a profile.\n"
            f"({exc})"
        ) from exc
