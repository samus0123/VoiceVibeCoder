"""The Claude brain.

Everything provider-shaped lives here: content blocks, the tool_use /
tool_result round trip, streaming, adaptive thinking, effort, the cached
system prefix, and server-side refusal fallbacks. The generator above sees
none of it — only :class:`~voicevibecoder.codegen.brain.Reply`.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from voicevibecoder.codegen.brain import Reply, ToolCall, ToolResult
from voicevibecoder.config import Config

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeBrain:
    name = "claude"

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

    # -- Brain -----------------------------------------------------------
    def reset(self) -> None:
        self._history = []

    def turn(
        self,
        system: str,
        user_text: str | None,
        tool_results: Sequence[ToolResult],
        tools: list[dict[str, Any]],
    ) -> Reply:
        if user_text is not None:
            self._history.append({"role": "user", "content": user_text})
        if tool_results:
            # All results from one assistant turn go back in a single user
            # message, or the model quietly stops calling tools in parallel.
            self._history.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": result.call_id,
                            "content": result.content,
                            "is_error": result.is_error,
                        }
                        for result in tool_results
                    ],
                }
            )

        response = self._stream(
            {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                # Stable prefix: prompt and tools never vary within a session,
                # so every turn after the first reads them from cache.
                "system": [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "messages": self._history,
                "tools": tools,
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": self.config.effort},
            }
        )

        if response.stop_reason == "refusal":
            return Reply(refused=True)

        self._history.append({"role": "assistant", "content": response.content})
        self._trim()

        return Reply(
            text=" ".join(
                block.text for block in response.content if block.type == "text"
            ).strip(),
            tool_calls=tuple(
                ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                for block in response.content
                if block.type == "tool_use"
            ),
        )

    def structured(self, system: str, prompt: str, schema: dict[str, Any]) -> str:
        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=8000,
            system=[
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ],
            messages=[{"role": "user", "content": prompt}],
            thinking={"type": "adaptive"},
            output_config={
                "effort": self.config.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        )
        if response.stop_reason == "refusal":
            return ""
        return next(
            (block.text for block in response.content if block.type == "text"), ""
        )

    # -- internals -------------------------------------------------------
    def _stream(self, kwargs: dict[str, Any]) -> Any:
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

    def _trim(self) -> None:
        """Keep the tail of the conversation so follow-ups have context."""
        limit = max(2, self.config.history_turns * 2)
        trimmed = self._history[-limit:]
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


def _make_client() -> Any:
    try:
        import anthropic  # noqa: PLC0415 — optional until a build is requested
    except ImportError as exc:
        raise RuntimeError(
            "the Anthropic SDK is not installed (pip install anthropic)"
        ) from exc
    try:
        client = anthropic.Anthropic()
    except Exception as exc:  # noqa: BLE001 — surface auth setup, not a stack trace
        raise RuntimeError(
            f"no credentials (set ANTHROPIC_API_KEY or run 'ant auth login') — {exc}"
        ) from exc

    # The SDK resolves credentials lazily, so constructing a client proves
    # nothing — without this check an auto-selected Claude brain would be
    # chosen happily and then fail on the first request, instead of falling
    # back to a local model that is right there and working.
    if not _credentials_present(client):
        raise RuntimeError(
            "no credentials (set ANTHROPIC_API_KEY or run 'ant auth login')"
        )
    return client


def _credentials_present(client: Any) -> bool:
    """Whether *some* documented credential source is configured.

    Deliberately not a network call: this runs at startup. An unset
    ANTHROPIC_API_KEY does not mean there are no credentials — a stored
    profile or workload identity federation counts too.
    """
    if getattr(client, "api_key", None) or getattr(client, "auth_token", None):
        return True
    if any(
        os.environ.get(name)
        for name in (
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_PROFILE",
            "ANTHROPIC_IDENTITY_TOKEN",
            "ANTHROPIC_IDENTITY_TOKEN_FILE",
        )
    ):
        return True
    return (Path.home() / ".config" / "anthropic").exists()
