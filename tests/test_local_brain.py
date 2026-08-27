"""The local brain: transport, the prose dialect, and brain selection."""

from __future__ import annotations

import pytest

from voicevibecoder.codegen.brain import ToolResult
from voicevibecoder.codegen.local_brain import (
    LocalBrain,
    extract_json,
    parse_file_blocks,
    spoken_summary,
)

REPLY_IN_PROSE = """\
Here is a counter.

FILE: main.py
```python
count = 0
print(count)
```

FILE: util/helpers.py
```python
def bump(n):
    return n + 1
```

SUMMARY: Wrote a counter and a helper that increments it.
"""


class FakeServer:
    """Stands in for Ollama: canned responses, recorded requests."""

    def __init__(self, *responses, repeat: bool = False):
        self.responses = list(responses)
        self.repeat = repeat
        self.requests: list[tuple[str, object]] = []

    def __call__(self, url, body, timeout=None):
        self.requests.append((url, body))
        if not self.responses:
            raise AssertionError("no canned response left")
        response = self.responses[0] if self.repeat else self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def chat(content: str = "", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"message": message, "done": True}


def ollama(config, **overrides):
    """Config with the dialect pinned, so no detection request is made."""
    return config.merged(local_api="ollama", **overrides)


def tools():
    return [
        {
            "name": "write_file",
            "description": "write it",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]


# -- the prose dialect ------------------------------------------------------
def test_files_are_parsed_from_a_reply_with_no_tool_calls():
    assert parse_file_blocks(REPLY_IN_PROSE) == [
        ("main.py", "count = 0\nprint(count)\n"),
        ("util/helpers.py", "def bump(n):\n    return n + 1\n"),
    ]


def test_a_path_in_the_fence_info_is_accepted():
    text = "```python main.py\nprint(1)\n```"
    assert parse_file_blocks(text) == [("main.py", "print(1)\n")]


def test_a_nameless_code_block_is_not_a_file():
    text = "For example:\n```python\nprint(1)\n```\n"
    assert parse_file_blocks(text) == []


def test_a_file_marker_far_above_a_fence_does_not_claim_it():
    text = "FILE: main.py\n\nSome long explanation here instead.\n\n```python\nx = 1\n```"
    assert parse_file_blocks(text) == []


def test_the_summary_line_is_preferred_for_speech():
    assert spoken_summary(REPLY_IN_PROSE, ["main.py"]) == (
        "Wrote a counter and a helper that increments it."
    )


def test_prose_is_used_when_no_summary_line_exists():
    text = "This program totals the numbers you give it.\n\nFILE: a.py\n```py\nx=1\n```"
    assert spoken_summary(text, ["a.py"]).startswith("This program totals")


def test_summary_falls_back_to_the_file_list():
    assert spoken_summary("```\nx\n```", ["a.py"]) == "Wrote a.py."


# -- the brain itself -------------------------------------------------------
def test_a_prose_reply_becomes_tool_calls(config):
    server = FakeServer(chat(REPLY_IN_PROSE))
    brain = LocalBrain(ollama(config), request=server)
    reply = brain.turn("system", "make a counter", [], tools())

    names = [call.name for call in reply.tool_calls]
    assert names == ["write_file", "write_file", "finish"]
    assert reply.tool_calls[0].arguments["path"] == "main.py"
    assert reply.tool_calls[-1].arguments["entrypoint"] == "main.py"
    assert reply.tool_calls[-1].arguments["summary"].startswith("Wrote a counter")


def test_native_tool_calls_are_used_when_the_model_makes_them(config):
    server = FakeServer(
        chat(
            "",
            [{"function": {"name": "write_file", "arguments": {"path": "a.py", "content": "x"}}}],
        )
    )
    reply = LocalBrain(ollama(config), request=server).turn("system", "go", [], tools())
    assert [call.name for call in reply.tool_calls] == ["write_file"]
    assert reply.tool_calls[0].arguments["content"] == "x"


def test_arguments_arriving_as_a_json_string_are_parsed(config):
    server = FakeServer(
        chat("", [{"function": {"name": "write_file", "arguments": '{"path": "a.py"}'}}])
    )
    reply = LocalBrain(ollama(config), request=server).turn("system", "go", [], tools())
    assert reply.tool_calls[0].arguments == {"path": "a.py"}


def test_a_streamed_response_is_folded_into_one_message(config):
    chunks = [
        {"message": {"content": "Here is "}},
        {"message": {"content": "the plan."}},
        {"message": {"content": ""}, "done": True},
    ]
    streamed: list[str] = []
    brain = LocalBrain(ollama(config), on_text=streamed.append, request=FakeServer(chunks))
    reply = brain.turn("system", "go", [], tools())

    assert reply.text == "Here is the plan."
    assert streamed == ["Here is ", "the plan."]


def test_the_request_carries_the_model_the_system_prompt_and_the_tools(config):
    server = FakeServer(chat("nothing to do"))
    LocalBrain(ollama(config, local_model="tiny:1b"), request=server).turn(
        "SYSTEM TEXT", "go", [], tools()
    )
    url, body = server.requests[0]

    assert url.endswith("/api/chat")
    assert body["model"] == "tiny:1b"
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"].startswith("SYSTEM TEXT")
    assert body["tools"][0]["function"]["name"] == "write_file"


def test_tool_results_are_sent_back_as_tool_messages(config):
    server = FakeServer(chat("ok"), chat("done"))
    brain = LocalBrain(ollama(config), request=server)
    brain.turn("system", "go", [], tools())
    brain.turn(
        "system",
        None,
        [ToolResult(call_id="1", name="write_file", content="wrote a.py")],
        tools(),
    )
    _url, body = server.requests[1]
    tool_message = [m for m in body["messages"] if m["role"] == "tool"][0]
    assert tool_message["content"] == "wrote a.py"
    assert tool_message["tool_name"] == "write_file"


def test_an_error_result_is_marked_for_the_model(config):
    server = FakeServer(chat("ok"), chat("done"))
    brain = LocalBrain(ollama(config), request=server)
    brain.turn("system", "go", [], tools())
    brain.turn(
        "system",
        None,
        [ToolResult("1", "write_file", "path escapes the workspace", is_error=True)],
        tools(),
    )
    _url, body = server.requests[1]
    assert "ERROR:" in [m for m in body["messages"] if m["role"] == "tool"][0]["content"]


def test_structured_output_asks_for_the_schema_and_unwraps_json(config):
    schema = {"type": "object", "properties": {}}
    server = FakeServer(chat('```json\n{"ideas": []}\n```'))
    payload = LocalBrain(ollama(config), request=server).structured("sys", "prompt", schema)

    assert payload == '{"ideas": []}'
    _url, body = server.requests[0]
    assert body["format"] == schema
    assert body["stream"] is False


def test_reset_forgets_the_conversation(config):
    server = FakeServer(chat("one"), chat("two"))
    brain = LocalBrain(ollama(config), request=server)
    brain.turn("system", "first", [], tools())
    brain.reset()
    brain.turn("system", "second", [], tools())

    _url, body = server.requests[1]
    assert not any("first" in str(message) for message in body["messages"])


def test_an_unreachable_server_explains_the_setup(config):
    brain = LocalBrain(ollama(config), request=FakeServer(OSError("refused"), repeat=True))
    assert not brain.available()
    assert "ollama pull" in brain.setup_help()

    with pytest.raises(RuntimeError, match="ollama"):
        brain.require()


def test_a_missing_model_is_named_in_the_setup_help(config):
    server = FakeServer({"models": [{"name": "llama3:8b"}]})
    brain = LocalBrain(config.merged(local_model="qwen2.5-coder:7b"), request=server)
    assert "not pulled yet" in brain.setup_help()


def test_a_pulled_model_reports_ready(config):
    server = FakeServer({"models": [{"name": "qwen2.5-coder:7b"}]})
    brain = LocalBrain(config.merged(local_model="qwen2.5-coder:7b"), request=server)
    assert brain.setup_help() == "ready"


def test_json_extraction_handles_bare_prose():
    assert extract_json('the answer is {"a": 1} I think') == '{"a": 1}'
    assert extract_json("no json here") == ""


# -- brain selection --------------------------------------------------------
def test_auto_falls_back_to_local_when_claude_has_no_credentials(config, monkeypatch):
    from voicevibecoder.codegen import brain as brain_module
    from voicevibecoder.codegen import claude_brain, local_brain

    monkeypatch.setattr(
        claude_brain, "_make_client", lambda: (_ for _ in ()).throw(RuntimeError("no key"))
    )
    monkeypatch.setattr(local_brain.LocalBrain, "available", lambda self: True)

    chosen = brain_module.build_brain(config.merged(brain="auto"))
    assert chosen.name == "local"


def test_auto_reports_both_problems_when_nothing_is_reachable(config, monkeypatch):
    from voicevibecoder.codegen import brain as brain_module
    from voicevibecoder.codegen import claude_brain, local_brain

    monkeypatch.setattr(
        claude_brain, "_make_client", lambda: (_ for _ in ()).throw(RuntimeError("no key"))
    )
    monkeypatch.setattr(local_brain.LocalBrain, "available", lambda self: False)
    monkeypatch.setattr(local_brain.LocalBrain, "setup_help", lambda self: "no server")

    with pytest.raises(brain_module.BrainUnavailable) as failure:
        brain_module.build_brain(config.merged(brain="auto"))
    assert "no key" in str(failure.value)
    assert "no server" in str(failure.value)


def test_asking_for_local_never_silently_uses_claude(config, monkeypatch):
    from voicevibecoder.codegen import brain as brain_module
    from voicevibecoder.codegen import local_brain

    monkeypatch.setattr(local_brain.LocalBrain, "available", lambda self: False)
    monkeypatch.setattr(local_brain.LocalBrain, "setup_help", lambda self: "no server")

    with pytest.raises(RuntimeError, match="no server"):
        brain_module.build_brain(config.merged(brain="local"))


def test_an_unknown_brain_is_rejected(config):
    from voicevibecoder.codegen import brain as brain_module

    with pytest.raises(ValueError, match="banana"):
        brain_module.build_brain(config.merged(brain="banana"))


# -- speaking to llama.cpp / LM Studio / vLLM -------------------------------
def openai_chat(content="", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message, "finish_reason": "stop"}]}


def test_the_dialect_is_detected_when_ollama_answers(config):
    server = FakeServer({"models": [{"name": "qwen2.5-coder:7b"}]})
    brain = LocalBrain(config, request=server)

    assert brain.installed_models() == ["qwen2.5-coder:7b"]
    assert brain.api_name == "ollama"


def test_the_dialect_falls_through_to_openai_when_ollama_does_not_answer(config):
    class Server:
        def __init__(self):
            self.urls = []

        def __call__(self, url, body, timeout=None):
            self.urls.append(url)
            if url.endswith("/api/tags"):
                raise OSError("404")
            return {"data": [{"id": "llama-3.1-8b-instruct"}]}

    server = Server()
    brain = LocalBrain(config, request=server)

    assert brain.installed_models() == ["llama-3.1-8b-instruct"]
    assert brain.api_name == "openai"
    assert server.urls == [
        "http://localhost:11434/api/tags",
        "http://localhost:11434/v1/models",
    ]


def test_an_openai_server_gets_an_openai_shaped_request(config):
    server = FakeServer(openai_chat("nothing to do"))
    brain = LocalBrain(config.merged(local_api="openai"), request=server)
    brain.turn("SYSTEM", "go", [], tools())

    url, body = server.requests[0]
    assert url.endswith("/v1/chat/completions")
    assert body["temperature"] == 0.2
    assert "options" not in body           # that is Ollama's spelling
    assert body["messages"][0]["role"] == "system"


def test_an_openai_reply_in_prose_still_becomes_tool_calls(config):
    server = FakeServer(openai_chat(REPLY_IN_PROSE))
    brain = LocalBrain(config.merged(local_api="openai"), request=server)
    reply = brain.turn("system", "make a counter", [], tools())

    assert [call.name for call in reply.tool_calls] == [
        "write_file",
        "write_file",
        "finish",
    ]


def test_an_openai_tool_call_keeps_the_servers_id_for_the_result(config):
    server = FakeServer(
        openai_chat(
            tool_calls=[
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": '{"path": "a.py"}'},
                }
            ]
        ),
        openai_chat("done"),
    )
    brain = LocalBrain(config.merged(local_api="openai"), request=server)
    reply = brain.turn("system", "go", [], tools())
    assert reply.tool_calls[0].id == "call_abc"

    brain.turn(
        "system",
        None,
        [ToolResult(call_id="call_abc", name="write_file", content="wrote a.py")],
        tools(),
    )
    _url, body = server.requests[1]
    tool_message = [m for m in body["messages"] if m["role"] == "tool"][0]
    assert tool_message["tool_call_id"] == "call_abc"


def test_openai_structured_output_uses_response_format(config):
    schema = {"type": "object", "properties": {}}
    server = FakeServer(openai_chat('{"ideas": []}'))
    payload = LocalBrain(config.merged(local_api="openai"), request=server).structured(
        "sys", "prompt", schema
    )

    assert payload == '{"ideas": []}'
    _url, body = server.requests[0]
    assert body["response_format"]["json_schema"]["schema"] == schema


def test_server_sent_events_are_parsed_and_streamed(config):
    from voicevibecoder.codegen.local_brain import _parse_body

    raw = (
        'data: {"choices":[{"delta":{"content":"Here "}}]}\n'
        'data: {"choices":[{"delta":{"content":"we go"}}]}\n'
        "data: [DONE]\n"
    )
    chunks = _parse_body(raw)
    streamed: list[str] = []
    brain = LocalBrain(
        config.merged(local_api="openai"),
        on_text=streamed.append,
        request=lambda *_a, **_k: chunks,
    )
    reply = brain.turn("system", "go", [], tools())

    assert reply.text == "Here we go"
    assert streamed == ["Here ", "we go"]


def test_tool_calls_streamed_in_fragments_are_reassembled(config):
    chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "write_file", "arguments": '{"pa'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": 'th": "a.py"}'}}
                        ]
                    }
                }
            ]
        },
    ]
    brain = LocalBrain(
        config.merged(local_api="openai"), request=lambda *_a, **_k: chunks
    )
    reply = brain.turn("system", "go", [], tools())

    assert len(reply.tool_calls) == 1
    assert reply.tool_calls[0].name == "write_file"
    assert reply.tool_calls[0].arguments == {"path": "a.py"}


def test_the_setup_help_names_both_servers(config):
    brain = LocalBrain(config, request=FakeServer(OSError("refused"), repeat=True))
    help_text = brain.setup_help()
    assert "ollama" in help_text.lower()
    assert "llama-server" in help_text


def test_structured_payloads_are_not_echoed_to_the_console(config):
    streamed: list[str] = []
    server = FakeServer(chat('{"ideas": []}'))
    brain = LocalBrain(ollama(config), on_text=streamed.append, request=server)
    brain.structured("sys", "prompt", {"type": "object"})

    assert streamed == []  # JSON is data, not narration
