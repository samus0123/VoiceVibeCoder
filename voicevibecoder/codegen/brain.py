"""The brain interface: where the thinking happens, whoever is doing it.

VoiceVibeCoder is not a front-end for one company's API. Everything above this
module — the NLP pipeline, the grammar, the workspace, the runner, the idea bar
— is ordinary software that works the same whether the model is a frontier
system in a datacentre or a quantised 7B on the laptop that is listening.

So the model layer is one small protocol with two operations:

``turn``        one step of an agentic conversation: here is the state, what do
                you want to do next — talk, or call a tool?
``structured``  one shot, JSON in a given schema, no conversation.

The *loop* lives in the generator, above this line; the *provider's shape* —
message format, history, tool encoding — lives in each brain, below it. That
split is what lets a local model that has never heard of tool calling still
drive the same build loop: it returns its file writes as tool calls anyway,
having parsed them out of its own prose.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from voicevibecoder.config import Config


@dataclass(frozen=True)
class ToolCall:
    """A tool the model wants run, in provider-neutral form."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    """What running it produced, on the way back."""

    call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class Reply:
    """One turn from the model."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    refused: bool = False

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@runtime_checkable
class Brain(Protocol):
    """What the generator needs from whatever is doing the thinking."""

    name: str

    def reset(self) -> None:
        """Forget the conversation so far."""

    def turn(
        self,
        system: str,
        user_text: str | None,
        tool_results: Sequence[ToolResult],
        tools: list[dict[str, Any]],
    ) -> Reply:
        """Advance the conversation by one model turn."""

    def structured(self, system: str, prompt: str, schema: dict[str, Any]) -> str:
        """One-shot JSON answer matching ``schema``, as raw text."""


class BrainUnavailable(RuntimeError):
    """No brain could be reached, with instructions for fixing that."""


def build_brain(
    config: Config,
    on_text: Callable[[str], None] | None = None,
    waiter: Callable[[], bool] | None = None,
) -> Brain:
    """Pick a brain: what was asked for, or the best that can be reached.

    ``auto`` prefers Claude when credentials exist and falls back to the local
    model when they do not — so the program still works on a plane, and still
    works on a machine that has never been given an API key.
    """
    from voicevibecoder.codegen.claude_brain import ClaudeBrain  # noqa: PLC0415
    from voicevibecoder.codegen.local_brain import LocalBrain  # noqa: PLC0415

    wanted = (config.brain or "auto").lower()

    if wanted == "claude":
        return ClaudeBrain(config, on_text=on_text)
    if wanted == "local":
        brain = LocalBrain(config, on_text=on_text, waiter=waiter)
        brain.require()
        return brain
    if wanted != "auto":
        raise ValueError(f"unknown brain: {config.brain!r}")

    try:
        return ClaudeBrain(config, on_text=on_text)
    except RuntimeError as claude_problem:
        local = LocalBrain(config, on_text=on_text, waiter=waiter)
        if local.available():
            return local
        raise BrainUnavailable(
            f"No brain available.\n"
            f"  Claude: {claude_problem}\n"
            f"  Local:  {local.setup_help()}"
        ) from claude_problem
