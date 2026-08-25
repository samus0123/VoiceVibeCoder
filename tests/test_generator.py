"""The Claude tool loop, driven with a stub client (no network)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from voicevibecoder.codegen.generator import CodeGenerator


def text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def tool_block(name: str, arguments: dict, call_id: str = "call_1"):
    return SimpleNamespace(type="tool_use", name=name, input=arguments, id=call_id)


def message(*blocks, stop_reason: str = "tool_use"):
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason)


class StubStream:
    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    @property
    def text_stream(self):
        return iter(
            block.text for block in self._response.content if block.type == "text"
        )

    def get_final_message(self):
        return self._response


class StubMessages:
    def __init__(self, responses, reject_beta_kwargs: bool = False):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.reject_beta_kwargs = reject_beta_kwargs

    def stream(self, **kwargs):
        if self.reject_beta_kwargs and ("fallbacks" in kwargs or "betas" in kwargs):
            raise TypeError("unexpected keyword argument 'fallbacks'")
        self.calls.append(kwargs)
        return StubStream(self.responses.pop(0))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class StubClient:
    def __init__(self, responses, reject_beta_kwargs: bool = False):
        self.messages = StubMessages(responses, reject_beta_kwargs)
        self.beta = SimpleNamespace(messages=self.messages)


@pytest.fixture
def build_responses():
    return [
        message(
            text_block("Writing it now."),
            tool_block("write_file", {"path": "main.py", "content": "print(1)\n"}),
        ),
        message(
            tool_block(
                "finish",
                {"summary": "Wrote main.py, which prints one.", "entrypoint": "main.py"},
                call_id="call_2",
            )
        ),
        message(text_block("done"), stop_reason="end_turn"),
    ]


def test_a_build_writes_files_and_returns_the_spoken_summary(
    config, workspace, build_responses
):
    client = StubClient(build_responses)
    result = CodeGenerator(config, client=client).build("make a thing", workspace)

    assert workspace.read("main.py") == "print(1)\n"
    assert result.summary == "Wrote main.py, which prints one."
    assert result.entrypoint == "main.py"
    assert result.changed_files == ["main.py"]


def test_the_request_carries_the_current_model_and_effort(
    config, workspace, build_responses
):
    client = StubClient(build_responses)
    CodeGenerator(config.merged(effort="max"), client=client).build("x", workspace)

    first = client.messages.calls[0]
    assert first["model"] == "claude-opus-5"
    assert first["output_config"]["effort"] == "max"
    assert first["thinking"] == {"type": "adaptive"}
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_an_sdk_without_server_side_fallbacks_degrades_quietly(
    config, workspace, build_responses
):
    client = StubClient(build_responses, reject_beta_kwargs=True)
    result = CodeGenerator(config, client=client).build("make a thing", workspace)

    assert result.summary  # it still completed
    assert all("fallbacks" not in call for call in client.messages.calls)


def test_a_refusal_is_reported_rather_than_raised(config, workspace):
    client = StubClient([message(text_block(""), stop_reason="refusal")])
    result = CodeGenerator(config, client=client).build("...", workspace)

    assert result.refused
    assert not result.changed_files


def test_a_tool_error_is_returned_to_the_model_not_raised(config, workspace):
    client = StubClient(
        [
            message(tool_block("write_file", {"path": "../escape.py", "content": "x"})),
            message(
                tool_block(
                    "finish", {"summary": "Wrote nothing.", "entrypoint": ""}, "c2"
                )
            ),
        ]
    )
    result = CodeGenerator(config, client=client).build("escape", workspace)

    # The conversation carries the failure back as a tool result, not an
    # exception, so the model gets a chance to correct the path.
    blocks = [
        block
        for entry in client.messages.calls[-1]["messages"]
        if isinstance(entry["content"], list)
        for block in entry["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert any(block["is_error"] for block in blocks)
    assert "escapes the workspace" in str(blocks)
    assert result.changed_files == []


def test_follow_ups_keep_the_earlier_conversation(config, workspace, build_responses):
    client = StubClient(
        build_responses
        + [
            message(
                tool_block("finish", {"summary": "Added colour.", "entrypoint": "main.py"}, "c3")
            )
        ]
    )
    generator = CodeGenerator(config, client=client)
    generator.build("make a thing", workspace)
    generator.build("now add colour", workspace)

    latest = client.messages.calls[-1]["messages"]
    assert any("make a thing" in str(entry) for entry in latest)


def test_reset_forgets_the_conversation(config, workspace, build_responses):
    client = StubClient(
        build_responses
        + [message(tool_block("finish", {"summary": "New.", "entrypoint": ""}, "c3"))]
    )
    generator = CodeGenerator(config, client=client)
    generator.build("make a thing", workspace)
    generator.reset()
    generator.build("something else", workspace)

    latest = client.messages.calls[-1]["messages"]
    assert not any("make a thing" in str(entry) for entry in latest)


def test_explain_is_given_only_read_only_tools(config, workspace):
    workspace.write("main.py", "print(1)\n")
    client = StubClient([message(text_block("It prints one."), stop_reason="end_turn")])
    answer = CodeGenerator(config, client=client).explain("what does it do", workspace)

    tools = {tool["name"] for tool in client.messages.calls[0]["tools"]}
    assert tools == {"read_file", "list_files"}
    assert answer == "It prints one."


def test_a_finish_naming_a_missing_entrypoint_is_ignored(config, workspace):
    client = StubClient(
        [message(tool_block("finish", {"summary": "Done.", "entrypoint": "ghost.py"}))]
    )
    result = CodeGenerator(config, client=client).build("x", workspace)
    assert result.entrypoint == ""


def test_ideation_uses_structured_output_and_the_local_bar(config, workspace):
    payload = json.dumps(
        {
            "ideas": [
                {
                    "name": "Drift Detector",
                    "pitch": "It notices when the numbers change character.",
                    "mechanism": "a sliding two-sample test",
                    "build_instruction": "build a drift detector",
                    "score": 93,
                },
                {
                    "name": "Todo List",
                    "pitch": "Tasks, with tags.",
                    "mechanism": "none really",
                    "build_instruction": "build a todo list",
                    "score": 40,
                },
            ]
        }
    )
    client = StubClient([message(text_block(payload), stop_reason="end_turn")])
    ideas = CodeGenerator(config, client=client).ideate("numbers", workspace)

    assert [idea.name for idea in ideas] == ["Drift Detector"]
    schema = client.messages.calls[0]["output_config"]["format"]
    assert schema["type"] == "json_schema"


def test_improvements_are_given_the_code_that_exists(config, workspace):
    workspace.write("main.py", "print('unique-marker')\n")
    client = StubClient([message(text_block('{"ideas": []}'), stop_reason="end_turn")])
    CodeGenerator(config, client=client).suggest_improvements("a printer", workspace)

    request = client.messages.calls[0]["messages"][0]["content"]
    assert "unique-marker" in request
