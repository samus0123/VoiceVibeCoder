"""The Termux/Android backend, driven with a fake command runner."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from voicevibecoder.speech import android


def completed(stdout: str = "", returncode: int = 0, stderr: str = ""):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


class FakeRunner:
    """Stands in for the termux-* commands."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls: list[list[str]] = []

    def __call__(self, argv):
        self.calls.append(argv)
        if not self.results:
            raise StopIteration  # ends the generator under test
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def available(monkeypatch):
    monkeypatch.setattr(android, "has_termux_api", lambda: True)


def take(iterator, count):
    out = []
    try:
        for _ in range(count):
            out.append(next(iterator))
    except (StopIteration, RuntimeError):
        pass
    return out


def test_recognised_utterances_are_yielded(config, available):
    runner = FakeRunner(completed("make a snake game"), completed("run it"))
    assert take(android.utterances(config, run=runner), 2) == [
        "make a snake game",
        "run it",
    ]
    assert runner.calls[0] == ["termux-speech-to-text"]


def test_a_silent_session_just_listens_again(config, available):
    runner = FakeRunner(completed("   "), completed("run it"))
    assert take(android.utterances(config, run=runner), 1) == ["run it"]


def test_a_timeout_is_survivable(config, available):
    runner = FakeRunner(
        subprocess.TimeoutExpired("termux-speech-to-text", 120),
        completed("undo that"),
    )
    status: list[str] = []
    stream = android.utterances(config, run=runner, on_status=status.append)
    assert take(stream, 1) == ["undo that"]
    assert "timed out" in status[0]


def test_one_failure_is_retried(config, available):
    runner = FakeRunner(completed(returncode=1, stderr="busy"), completed("run it"))
    assert take(android.utterances(config, run=runner), 1) == ["run it"]


def test_repeated_failures_stop_with_a_setup_hint(config, available):
    runner = FakeRunner(*[completed(returncode=1, stderr="permission denied")] * 3)
    with pytest.raises(RuntimeError, match="permission"):
        list(android.utterances(config, run=runner))


def test_a_missing_termux_api_explains_the_setup(config, monkeypatch):
    monkeypatch.setattr(android, "has_termux_api", lambda: False)
    with pytest.raises(RuntimeError, match="F-Droid"):
        next(android.utterances(config))


def test_the_speaker_shortens_before_speaking():
    runner = FakeRunner(completed(), completed())
    android.TermuxSpeaker(run=runner).say("x" * 500)
    argv = runner.calls[0]
    assert argv[0] == "termux-tts-speak"
    assert len(argv[1]) <= 243  # shorten()'s limit plus an ellipsis


def test_the_speaker_never_raises():
    def explode(_argv):
        raise OSError("no tts engine")

    android.TermuxSpeaker(run=explode).say("hello")  # must not raise


def test_an_empty_response_is_not_spoken():
    runner = FakeRunner(completed())
    android.TermuxSpeaker(run=runner).say("   ")
    assert runner.calls == []


def test_termux_is_detected_from_the_environment(monkeypatch):
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.setenv("PREFIX", "/usr")
    assert not android.is_termux()

    monkeypatch.setenv("TERMUX_VERSION", "0.118.0")
    assert android.is_termux()

    monkeypatch.delenv("TERMUX_VERSION")
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
    assert android.is_termux()
