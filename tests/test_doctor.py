"""The diagnostic, and the promise that a session starts without a brain."""

from __future__ import annotations

from voicevibecoder.codegen.generator import CodeGenerator
from voicevibecoder.doctor import report


def test_the_report_covers_every_way_it_can_be_half_set_up(config):
    text = report(config)
    for heading in ("runtime", "packages", "brains", "listening", "workspace"):
        assert f"{heading}:" in text
    assert "python" in text
    assert "VERDICT" in text


def test_the_verdict_says_what_to_do_when_no_brain_is_reachable(config, monkeypatch):
    from voicevibecoder.codegen import local_brain

    monkeypatch.setattr(local_brain.LocalBrain, "available", lambda self: False)
    monkeypatch.setattr(local_brain.LocalBrain, "installed_models", lambda self: None)
    monkeypatch.setattr("voicevibecoder.doctor._claude_ready", lambda: False)

    text = report(config)
    assert "no brain is reachable" in text
    assert "ANTHROPIC_API_KEY" in text
    assert "11434" in text


def test_the_verdict_names_the_brain_that_will_be_used(config, monkeypatch):
    from voicevibecoder.codegen import local_brain

    monkeypatch.setattr(local_brain.LocalBrain, "available", lambda self: True)
    monkeypatch.setattr(
        local_brain.LocalBrain, "installed_models", lambda self: ["qwen2.5-coder:7b"]
    )
    monkeypatch.setattr("voicevibecoder.doctor._claude_ready", lambda: False)

    assert f"ready to build, using {config.local_model}" in report(config)


def test_the_report_flags_an_unwritable_workspace(config, tmp_path, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr("pathlib.Path.mkdir", refuse)
    assert "NOT WRITABLE" in report(config)


def test_a_generator_connects_to_nothing_until_it_must(config):
    # No server, no key: constructing must not raise, or the session cannot
    # even open to tell you what is wrong.
    generator = CodeGenerator(config.merged(local_url="http://127.0.0.1:1"))
    assert not generator.connected
    assert generator.brain_name == "not connected yet"


def test_the_session_survives_a_build_with_no_brain(config, workspace, speaker):
    from voicevibecoder.console import Console
    from voicevibecoder.session import Session

    generator = CodeGenerator(config.merged(local_url="http://127.0.0.1:1"))
    session = Session(
        config=config,
        workspace=workspace,
        generator=generator,
        speaker=speaker,
        console=Console(color=False),
    )
    session.greet()
    assert session.handle("list files") is True       # works with no brain
    assert session.handle("make me a snake game") is True  # reports, does not crash
    assert "brain" in " ".join(speaker.said).lower()
    assert session.handle("quit") is False


def test_android_is_warned_off_the_rust_build(monkeypatch):
    from voicevibecoder.codegen import claude_brain

    monkeypatch.setenv("TERMUX_VERSION", "0.118.0")
    message = claude_brain._sdk_missing_help()
    assert "rust" in message.lower()
    assert "setup-llama" in message


def test_elsewhere_the_hint_stays_simple(monkeypatch):
    from voicevibecoder.codegen import claude_brain

    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.setenv("PREFIX", "/usr")
    assert claude_brain._sdk_missing_help() == (
        "the Anthropic SDK is not installed (pip install anthropic)"
    )
