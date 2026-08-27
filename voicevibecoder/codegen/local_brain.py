"""The local brain: a model running on this machine, no API key, no network.

Speaks to whatever local server is actually there. Two dialects cover nearly
every one people run: Ollama's native API, and the OpenAI-compatible API that
llama.cpp, LM Studio and vLLM serve. Which one is in front of it is *detected*
rather than configured — whichever model-listing endpoint answers decides —
because "install this specific server first" is a bad answer to "I just want to
use the model I already have".

Transport is the standard library: a program whose selling point is that it
works offline should not need a package index to start.

The interesting problem is that small local models are *bad at tool calling*.
A 7B coder will happily describe the file it would write instead of calling
``write_file``, and a brain that only understands tool calls would come back
empty every time. So this one accepts both dialects: native tool calls when the
model manages them, and otherwise a plain-prose convention —

    FILE: main.py
    ```python
    print("hello")
    ```

— parsed out of the reply and *returned as tool calls anyway*. The generator
above never learns which happened. That is what makes a small local model a
first-class citizen here rather than a degraded mode.
"""

from __future__ import annotations

import json
import re
import socket
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any
from urllib.parse import urlsplit

from voicevibecoder.codegen.brain import Reply, ToolCall, ToolResult
from voicevibecoder.config import Config

# Appended to the system prompt so a model with no tool support still knows
# how to hand back files.
BLOCK_PROTOCOL = """\

If you cannot call tools, write files in prose instead, exactly like this: a \
line reading `FILE: <path>` immediately followed by a fenced code block with \
the file's complete contents. Repeat for each file. Then a final line reading \
`SUMMARY: <one sentence>`. Never write a code block without a FILE: line above \
it, and never abbreviate a file with "... rest unchanged" — always the whole \
file.
"""

RUNNABLE = (".py", ".js", ".sh", ".rb", ".go", ".mjs")

# Where local model servers live by default. Ollama picks 11434, llama.cpp and
# LM Studio pick 8080 and 1234 — and someone who just started one of them has
# no reason to know which. When the configured address is on this machine and
# nothing answers there, these are tried before giving up.
DEFAULT_PORTS = (11434, 8080, 1234)
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})

# Last resort. Someone who started a server on a port of their own choosing
# should not have to tell the program which — a closed port on this machine
# refuses instantly, so trying a few costs nothing.
SCAN_PORTS = (11434, 8080, 8081, 1234, 5000, 8000, 3000, 5001, 11435)



def port_open(port: int, host: str = "127.0.0.1") -> bool:
    """Whether anything is listening, without speaking HTTP to it."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.3)
            return probe.connect_ex((host, port)) == 0
    except OSError:
        return False

_FENCE = re.compile(
    r"^[ \t]*(?P<fence>```+|~~~+)[ \t]*(?P<info>[^\n]*)\n(?P<body>.*?)^[ \t]*(?P=fence)[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
_FILE_MARKER = re.compile(r"(?:^|\n)[ \t>*#-]*FILE:[ \t]*[`\"']?(?P<path>[\w./\-]+)", re.IGNORECASE)
_SUMMARY_MARKER = re.compile(r"(?:^|\n)[ \t>*#-]*SUMMARY:[ \t]*(?P<summary>[^\n]+)", re.IGNORECASE)
_PATHISH = re.compile(r"^(?:path|file|title|name)?=?[`\"']?(?P<path>[\w\-./]+\.[\w]{1,6})[`\"']?$")


class LocalBrain:
    name = "local"

    def __init__(
        self,
        config: Config,
        on_text: Callable[[str], None] | None = None,
        request: Callable[..., Any] | None = None,
        waiter: Callable[[], bool] | None = None,
    ) -> None:
        self.config = config
        self._on_text = on_text or (lambda _chunk: None)
        self._request = request or self._http
        # Called once if nothing answers: a server this program started may
        # still be reading the model off disk.
        self._waiter = waiter
        self._waited = False
        self._history: list[dict[str, Any]] = []
        self._call_counter = 0
        self._dialect: Any = DIALECTS.get(config.local_api)  # None means "probe"
        self._served: list[str] | None = None
        self._found_url: str = ""
        self._unsupported: set[str] = set()

    # -- availability ----------------------------------------------------
    def available(self) -> bool:
        return self.installed_models() is not None

    def installed_models(self) -> list[str] | None:
        """Models the server offers, or None if no server answers.

        Doubles as dialect detection: whichever listing endpoint answers tells
        us which API this server speaks, so a person can point at Ollama or at
        llama.cpp without having to say which they are running.
        """
        dialects = [self._dialect] if self._dialect else [OllamaDialect, OpenAIDialect]
        found = self._try_urls(self.candidate_urls(), dialects)
        if found is not None:
            return found
        # Nothing where we were told to look: sweep this machine for a server
        # someone started on a port of their own choosing.
        found = self._try_urls(self.scanned_urls(), dialects)
        if found is not None:
            return found
        # Still nothing — but a server we launched at startup may just be slow
        # to load. Wait for it once, then look again.
        if self._waiter is not None and not self._waited:
            self._waited = True
            if self._waiter():
                return self._try_urls(self.candidate_urls(), dialects)
        return None

    def scanned_urls(self) -> list[str]:
        """Local addresses that have something listening on them."""
        parts = urlsplit(self.config.local_url.rstrip("/"))
        if parts.hostname not in LOCAL_HOSTS:
            return []  # a LAN address is a deliberate choice; do not sweep
        already = set(self.candidate_urls())
        return [
            url
            for port in SCAN_PORTS
            if (url := f"http://127.0.0.1:{port}") not in already and port_open(port)
        ]

    def _try_urls(self, urls: list[str], dialects: list[Any]) -> list[str] | None:
        for url in urls:
            for dialect in dialects:
                try:
                    payload = self._request(
                        # Short: a local server either answers at once or is
                        # not there, and several addresses are tried in turn.
                        f"{url}{dialect.models_path}",
                        None,
                        timeout=2,
                    )
                except (OSError, ValueError):
                    continue
                if isinstance(payload, list):  # a stub handed back chunks
                    payload = payload[-1] if payload else {}
                self._dialect = dialect
                self._found_url = url
                self._served = dialect.models_of(payload)
                return self._served
        return None

    def candidate_urls(self) -> list[str]:
        """The configured address first, then the usual local ports."""
        configured = self.config.local_url.rstrip("/")
        urls = [configured]
        parts = urlsplit(configured)
        if parts.hostname in LOCAL_HOSTS:
            host = f"{parts.scheme}://{parts.hostname}"
            urls += [f"{host}:{port}" for port in DEFAULT_PORTS]
        return list(dict.fromkeys(urls))  # ordered, deduplicated

    @property
    def dialect(self) -> Any:
        """The server's API flavour, probed once and remembered."""
        if self._dialect is None:
            self.installed_models()
        return self._dialect or OllamaDialect

    @property
    def api_name(self) -> str:
        return self.dialect.name

    def require(self) -> None:
        if not self.available():
            raise RuntimeError(self.setup_help())

    @property
    def effective_model(self) -> str:
        """The model name to actually send.

        Ollama routes by name, so the configured name matters. An
        OpenAI-compatible server like llama-server has one model loaded and
        ignores the field — so rather than refuse because the configured name
        does not match, use whatever that server says it is serving.
        """
        configured = self.config.local_model
        if self.dialect is not OpenAIDialect:
            return configured
        served = self._served or []
        if served and not _has_model(served, configured):
            return served[0]
        return configured

    def setup_help(self) -> str:
        models = self.installed_models()
        if models is None:
            return (
                "no local model server at "
                f"{', '.join(self.candidate_urls())}.\n"
                "  Ollama:     https://ollama.com  then  "
                f"ollama pull {self.config.local_model}\n"
                "  llama.cpp:  llama-server -m <model.gguf> --port 11434\n"
                "  Either works — whichever answers is the one it talks to."
            )
        if self.dialect is OpenAIDialect:
            # One loaded model, addressed by whatever name it reports.
            return f"ready — serving {self.effective_model}"
        if not _has_model(models, self.config.local_model):
            return (
                f"{self.config.local_model} is not pulled yet — "
                f"ollama pull {self.config.local_model}\n"
                f"  installed: {', '.join(models) or 'nothing yet'}"
            )
        return "ready"

    @property
    def base_url(self) -> str:
        """Where the server actually answered, or where we were told to look."""
        return self._found_url or self.config.local_url.rstrip("/")

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
        for result in tool_results:
            self._history.append(self.dialect.tool_message(result))

        message = self._chat(
            system + BLOCK_PROTOCOL,
            self._history,
            tools=[_as_function(tool) for tool in tools],
        )
        self._history.append(
            {
                "role": "assistant",
                "content": message.get("content", ""),
                **(
                    {"tool_calls": message["tool_calls"]}
                    if message.get("tool_calls")
                    else {}
                ),
            }
        )
        self._trim()

        text = (message.get("content") or "").strip()
        calls = [self._native_call(raw) for raw in message.get("tool_calls") or []]
        if not calls:
            calls = self._calls_from_prose(text)
        return Reply(text=text, tool_calls=tuple(calls))

    def warm(self, system: str) -> bool:
        """Push the system prompt through the model and ask for one token.

        Loading the weights is only half the wait: the first real request still
        has to process a thousand tokens of system prompt, which on a phone is
        another half-minute. Doing it in advance leaves that work in the
        server's cache, so the first thing the person actually asks for starts
        generating immediately.
        """
        dialect = self.dialect
        body = dialect.limit(
            dialect.body(
                self.effective_model,
                system,
                [{"role": "user", "content": "ready?"}],
                None,
                None,
                False,
            ),
            1,
        )
        try:
            self._request(
                f"{self.base_url}{dialect.chat_path}",
                body,
                timeout=self.config.local_timeout_s,
            )
        except (OSError, ValueError):
            return False
        return True

    def structured(self, system: str, prompt: str, schema: dict[str, Any]) -> str:
        message = self._chat(
            system,
            [{"role": "user", "content": prompt}],
            schema=schema,
            stream=False,
            # A schema-shaped payload is data, not something to read out.
            echo=False,
        )
        return extract_json(message.get("content", ""))

    # -- prose dialect ---------------------------------------------------
    def _calls_from_prose(self, text: str) -> list[ToolCall]:
        """Turn `FILE:` blocks into the tool calls the model did not make."""
        files = parse_file_blocks(text)
        if not files:
            return []

        calls = [
            self._call("write_file", {"path": path, "content": content})
            for path, content in files
        ]
        entrypoint = next(
            (path for path, _ in files if path.endswith(RUNNABLE)), ""
        )
        calls.append(
            self._call(
                "finish",
                {
                    "summary": spoken_summary(text, [path for path, _ in files]),
                    "entrypoint": entrypoint,
                },
            )
        )
        return calls

    def _native_call(self, raw: dict[str, Any]) -> ToolCall:
        function = raw.get("function", raw)
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):  # OpenAI-compatible servers send JSON text
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {}
        call = self._call(function.get("name", ""), dict(arguments))
        # Keep the server's own id: OpenAI-compatible servers match the tool
        # result back to the call by it.
        server_id = raw.get("id")
        return ToolCall(server_id, call.name, call.arguments) if server_id else call

    def _call(self, name: str, arguments: dict[str, Any]) -> ToolCall:
        self._call_counter += 1
        return ToolCall(id=f"local_{self._call_counter}", name=name, arguments=arguments)

    # -- transport -------------------------------------------------------
    def _chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        schema: dict[str, Any] | None = None,
        stream: bool = True,
        echo: bool = True,
    ) -> dict[str, Any]:
        dialect = self.dialect
        body = dialect.body(
            self.effective_model,
            system,
            messages,
            None if "tools" in self._unsupported else tools,
            None if "schema" in self._unsupported else schema,
            stream,
        )

        url = f"{self.base_url}{dialect.chat_path}"
        try:
            payload = self._request(url, body, timeout=self.config.local_timeout_s)
        except ServerRejected as rejection:
            # Plenty of local builds refuse tools, or JSON schemas, or both.
            # The prose protocol needs neither, so drop them and try once more
            # rather than failing at something we can do without.
            retry = rejection.without_unsupported(body)
            if retry is None:
                raise RuntimeError(f"the model server refused: {rejection}") from rejection
            self._unsupported.update(retry.pop("_dropped"))
            payload = self._request(url, retry, timeout=self.config.local_timeout_s)
        except urllib.error.URLError as exc:
            raise RuntimeError(f"local model unreachable: {self.setup_help()}") from exc
        except OSError as exc:
            raise RuntimeError(f"local model failed: {exc}") from exc

        return self._collect(payload, echo=echo)

    def _collect(self, payload: Any, echo: bool = True) -> dict[str, Any]:
        """Fold a streamed response (or a single one) into one message."""
        chunks = payload if isinstance(payload, list) else [payload]
        message_of = self.dialect.message_of
        content: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for chunk in chunks:
            message = message_of(chunk)
            piece = message.get("content") or ""
            if piece:
                content.append(piece)
                if echo:
                    self._on_text(piece)
            _merge_tool_calls(tool_calls, message.get("tool_calls") or [])
        return {"content": "".join(content), "tool_calls": tool_calls}

    def _http(self, url: str, body: Any, timeout: float) -> Any:
        """POST JSON (or GET when body is None); returns dict or NDJSON list.

        A rejected request carries the reason in its *body* — "this model does
        not support tools", a grammar error, a context-length complaint. urllib
        throws that away and leaves you with "HTTP Error 400: Bad Request", so
        it is read back out and put in the exception where it is of some use.
        """
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(  # noqa: S310 — http(s) URL from config
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if data else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                raw = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            raise ServerRejected(exc.code, _error_detail(exc)) from exc
        return _parse_body(raw)

    def _trim(self) -> None:
        limit = max(2, self.config.history_turns * 2)
        trimmed = self._history[-limit:]
        while trimmed and trimmed[0].get("role") == "tool":
            trimmed = trimmed[1:]  # a tool result with no call above it
        self._history = trimmed



# ---------------------------------------------------------------------------
# Server dialects
# ---------------------------------------------------------------------------
class OllamaDialect:
    """Ollama's native API: NDJSON streaming, tools by name."""

    name = "ollama"
    chat_path = "/api/chat"
    models_path = "/api/tags"

    @staticmethod
    def body(model, system, messages, tools, schema, stream):
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": stream,
            # Low but not zero: deterministic enough to follow the protocol,
            # loose enough to not loop on a phrase.
            "options": {"temperature": 0.2},
        }
        if tools:
            body["tools"] = tools
        if schema:
            body["format"] = schema
        return body

    @staticmethod
    def limit(body, tokens):
        body.setdefault("options", {})["num_predict"] = tokens
        return body

    @staticmethod
    def message_of(chunk):
        return chunk.get("message") or {}

    @staticmethod
    def models_of(payload):
        return [model.get("name", "") for model in payload.get("models", [])]

    @staticmethod
    def tool_message(result):
        return {
            "role": "tool",
            "tool_name": result.name,
            "content": ("ERROR: " if result.is_error else "") + result.content,
        }


class OpenAIDialect:
    """The OpenAI-compatible API that llama.cpp, LM Studio and vLLM serve."""

    name = "openai"
    chat_path = "/v1/chat/completions"
    models_path = "/v1/models"

    @staticmethod
    def body(model, system, messages, tools, schema, stream):
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": stream,
            "temperature": 0.2,
        }
        if tools:
            body["tools"] = tools
        if schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": schema},
            }
        return body

    @staticmethod
    def limit(body, tokens):
        body["max_tokens"] = tokens
        return body

    @staticmethod
    def message_of(chunk):
        choices = chunk.get("choices") or [{}]
        choice = choices[0] if choices else {}
        # Streaming sends deltas; a single response sends a whole message.
        return choice.get("delta") or choice.get("message") or {}

    @staticmethod
    def models_of(payload):
        return [model.get("id", "") for model in payload.get("data", [])]

    @staticmethod
    def tool_message(result):
        return {
            "role": "tool",
            "tool_call_id": result.call_id,
            "content": ("ERROR: " if result.is_error else "") + result.content,
        }


DIALECTS = {"ollama": OllamaDialect, "openai": OpenAIDialect}

# ---------------------------------------------------------------------------
# Parsing the prose dialect
# ---------------------------------------------------------------------------
def parse_file_blocks(text: str) -> list[tuple[str, str]]:
    """Extract ``(path, content)`` pairs from a reply written as prose.

    A path comes from a ``FILE:`` line just above the fence, or from the
    fence's info string (```` ```python main.py ````). A fenced block with no
    path is ignored — a nameless file is not a file.
    """
    files: list[tuple[str, str]] = []
    for match in _FENCE.finditer(text):
        path = _path_before(text, match.start()) or _path_from_info(match.group("info"))
        if not path:
            continue
        body = match.group("body")
        if body.endswith("\n\n"):
            body = body[:-1]
        files.append((path, body))
    return files


def _path_before(text: str, fence_start: int) -> str:
    """The nearest FILE: marker in the few lines above a fence."""
    window = text[max(0, fence_start - 400) : fence_start]
    markers = list(_FILE_MARKER.finditer(window))
    if not markers:
        return ""
    # Only if nothing but blank space and the marker line separate them.
    tail = window[markers[-1].end() :]
    return markers[-1].group("path") if tail.strip(" \t\r\n`\"'") == "" else ""


def _path_from_info(info: str) -> str:
    for token in info.replace(":", " ").split():
        candidate = _PATHISH.match(token)
        if candidate:
            return candidate.group("path")
    return ""


def spoken_summary(text: str, paths: list[str]) -> str:
    """The sentence to say aloud, from a SUMMARY: line or the surrounding prose."""
    marked = _SUMMARY_MARKER.search(text)
    if marked:
        return marked.group("summary").strip()
    prose = _FENCE.sub(" ", text)
    prose = _FILE_MARKER.sub(" ", prose)
    sentences = [line.strip() for line in prose.split(".") if len(line.strip()) > 15]
    if sentences:
        return sentences[0].strip() + "."
    return f"Wrote {', '.join(paths)}." if paths else "Nothing changed."


def extract_json(text: str) -> str:
    """Pull the first JSON object out of a reply that may be wrapped in prose."""
    text = (text or "").strip()
    if text.startswith("{"):
        return text
    for match in _FENCE.finditer(text):
        body = match.group("body").strip()
        if body.startswith("{"):
            return body
    start = text.find("{")
    end = text.rfind("}")
    return text[start : end + 1] if 0 <= start < end else ""


def _as_function(tool: dict[str, Any]) -> dict[str, Any]:
    """Anthropic-shaped tool definition -> the function-calling shape."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {}),
        },
    }


def _has_model(models: list[str], wanted: str) -> bool:
    # "qwen2.5-coder:7b" should match an installed "qwen2.5-coder:7b" exactly,
    # and a bare "qwen2.5-coder" should match its default tag.
    return any(name == wanted or name.split(":")[0] == wanted for name in models)


def _parse_body(raw: str) -> Any:
    """Read one JSON object, NDJSON (Ollama) or server-sent events (OpenAI)."""
    chunks: list[Any] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("data:"):  # server-sent events
            line = line[5:].strip()
            if line == "[DONE]":
                continue
        try:
            chunks.append(json.loads(line))
        except ValueError:
            continue
    if not chunks:
        raise ValueError("no JSON in response")
    return chunks[0] if len(chunks) == 1 else chunks


def _merge_tool_calls(
    collected: list[dict[str, Any]], incoming: list[dict[str, Any]]
) -> None:
    """Accumulate tool calls that a streaming server sends in fragments.

    Streamed OpenAI-compatible calls arrive as indexed pieces — the name in one
    chunk, the arguments a few characters at a time in the next — so they are
    joined by index rather than appended.
    """
    for raw in incoming:
        index = raw.get("index")
        existing = None
        if index is not None:
            existing = next(
                (call for call in collected if call.get("index") == index), None
            )
        if existing is None:
            collected.append(json.loads(json.dumps(raw)))  # a private copy
            continue
        function = existing.setdefault("function", {})
        piece = raw.get("function") or {}
        if piece.get("name"):
            function["name"] = piece["name"]
        if piece.get("arguments"):
            function["arguments"] = (function.get("arguments") or "") + piece["arguments"]
        if raw.get("id"):
            existing["id"] = raw["id"]


class ServerRejected(OSError):
    """The server answered, and the answer was no — with a reason.

    An OSError on purpose: during dialect probing a 404 means "not this API",
    which every caller already handles as "nothing here". Only the chat path
    cares about the distinction, and it catches this class first.
    """

    # Words a server uses when it is refusing a *feature* rather than the
    # request itself. Each maps to the request key worth dropping.
    FEATURE_HINTS = {
        "tools": ("tool", "function call", "tool_choice"),
        "schema": ("response_format", "json_schema", "grammar", "structured"),
    }

    def __init__(self, status: int, detail: str) -> None:
        self.status = status
        self.detail = detail
        super().__init__(f"HTTP {status}: {detail}" if detail else f"HTTP {status}")

    def without_unsupported(self, body: dict[str, Any]) -> dict[str, Any] | None:
        """A retry with the refused feature removed, or None if that is not it."""
        haystack = self.detail.lower()
        dropped = {
            feature
            for feature, hints in self.FEATURE_HINTS.items()
            if any(hint in haystack for hint in hints)
        }
        # A 400 with no clue in it is still most likely the tools payload,
        # which is the only exotic thing in an otherwise ordinary request.
        if not dropped and self.status == 400 and body.get("tools"):
            dropped = {"tools"}
        if not dropped:
            return None

        retry = {key: value for key, value in body.items() if key not in ("tools",)} \
            if "tools" in dropped else dict(body)
        if "schema" in dropped:
            retry.pop("format", None)
            retry.pop("response_format", None)
        if not (set(body) - set(retry)):
            return None  # nothing actually came off; do not loop
        retry["_dropped"] = dropped
        return retry


def _error_detail(exc: urllib.error.HTTPError) -> str:
    """The server's own explanation, dug out of the error body."""
    try:
        raw = exc.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 — a body we cannot read is no body
        return ""
    try:
        parsed = json.loads(raw)
    except ValueError:
        return " ".join(raw.split())[:300]
    if isinstance(parsed, dict):
        error = parsed.get("error", parsed)
        if isinstance(error, dict):
            return str(error.get("message") or error)[:300]
        return str(error)[:300]
    return str(parsed)[:300]
