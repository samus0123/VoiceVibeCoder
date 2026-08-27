"""Starting the model server ourselves, so its load time overlaps with typing."""

from __future__ import annotations

from pathlib import Path

from voicevibecoder import model_server
from voicevibecoder.model_server import ModelServer, find_model, port_of


def gguf(path: Path, size: int = 100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + b"\0" * size)
    return path


def test_the_port_comes_from_the_configured_url():
    assert port_of("http://127.0.0.1:8081") == 8081
    assert port_of("http://localhost") == 11434
    assert port_of("http://localhost/") == 11434


def test_a_named_model_file_is_used_as_given(config, tmp_path):
    named = gguf(tmp_path / "chosen.gguf")
    found = find_model(config.merged(local_model_file=str(named)))
    assert found == named


def test_a_named_model_file_that_does_not_exist_is_not_invented(config):
    assert find_model(config.merged(local_model_file="/nope/missing.gguf")) is None


def test_the_largest_model_lying_around_wins(config, tmp_path, monkeypatch):
    """A directory usually has one real model and a couple of experiments."""
    gguf(tmp_path / "tiny.gguf", 10)
    big = gguf(tmp_path / "real.gguf", 5000)
    monkeypatch.setattr(model_server, "MODEL_DIRS", (str(tmp_path),))
    assert find_model(config) == big


def test_no_model_anywhere_is_not_an_error(config, tmp_path, monkeypatch):
    monkeypatch.setattr(model_server, "MODEL_DIRS", (str(tmp_path),))
    assert find_model(config) is None


def test_discovery_needs_both_a_binary_and_a_model(config, tmp_path, monkeypatch):
    monkeypatch.setattr(model_server, "MODEL_DIRS", (str(tmp_path),))
    monkeypatch.setattr(model_server.shutil, "which", lambda _name: None)
    assert ModelServer.discover(config) is None  # no binary

    gguf(tmp_path / "m.gguf")
    monkeypatch.setattr(
        model_server.shutil, "which", lambda name: "/usr/bin/llama-server"
    )
    server = ModelServer.discover(config)
    assert server is not None
    assert server.model_path.name == "m.gguf"
    assert server.port == 11434


def test_a_taken_port_is_never_raced(tmp_path, monkeypatch):
    """Something already owns it: starting a second one only fails to bind."""
    monkeypatch.setattr(model_server, "port_open", lambda *_a, **_k: True)
    server = ModelServer(
        binary="llama-server",
        model_path=gguf(tmp_path / "m.gguf"),
        port=11434,
        log_path=tmp_path / "log",
    )
    assert server.start() is False
    assert server.process is None


def test_waiting_returns_when_the_port_answers(tmp_path, monkeypatch):
    answers = iter([False, False, True])
    monkeypatch.setattr(model_server, "port_open", lambda *_a, **_k: next(answers))
    server = ModelServer("llama-server", tmp_path / "m.gguf", 11434, tmp_path / "log")

    assert server.wait(timeout_s=5, interval_s=0.01) is True


def test_waiting_gives_up_when_the_process_dies(tmp_path, monkeypatch):
    monkeypatch.setattr(model_server, "port_open", lambda *_a, **_k: False)

    class Dead:
        def poll(self):
            return 1

    server = ModelServer("llama-server", tmp_path / "m.gguf", 11434, tmp_path / "log")
    server.process = Dead()
    assert server.wait(timeout_s=5, interval_s=0.01) is False


def test_the_log_tail_is_available_for_the_error_message(tmp_path):
    log = tmp_path / "log"
    log.write_text("\n".join(f"line {n}" for n in range(30)), "utf-8")
    server = ModelServer("llama-server", tmp_path / "m.gguf", 11434, log)
    assert server.tail(3) == "line 27\nline 28\nline 29"
    assert ModelServer("x", tmp_path / "m", 1, tmp_path / "absent").tail() == ""


def test_the_brain_waits_once_for_a_loading_server(config):
    """The whole point: a model still loading is waited for, not failed on."""
    from voicevibecoder.codegen.local_brain import LocalBrain

    state = {"ready": False, "waits": 0}

    def server(url, body, timeout=None):
        if not state["ready"]:
            raise OSError("connection refused")
        return {"models": [{"name": "llama-3.2-1b"}]}

    def waiter():
        state["waits"] += 1
        state["ready"] = True
        return True

    brain = LocalBrain(config, request=server, waiter=waiter)
    assert brain.installed_models() == ["llama-3.2-1b"]
    assert state["waits"] == 1


def test_the_brain_does_not_wait_twice(config):
    from voicevibecoder.codegen.local_brain import LocalBrain

    waits = []

    def server(url, body, timeout=None):
        raise OSError("connection refused")

    brain = LocalBrain(config, request=server, waiter=lambda: (waits.append(1), False)[1])
    assert brain.installed_models() is None
    assert brain.installed_models() is None
    assert len(waits) == 1  # a server that never came up is not waited for again


def test_warming_sends_the_system_prompt_and_asks_for_one_token(config):
    """Loading the weights is half the wait; the prompt is the other half."""
    from voicevibecoder.codegen.local_brain import LocalBrain

    sent = []

    def server(url, body, timeout=None):
        if url.endswith("/api/tags"):
            return {"models": [{"name": "llama-3.2-1b"}]}
        sent.append(body)
        return {"message": {"content": "ready"}}

    brain = LocalBrain(config, request=server)
    assert brain.warm("SYSTEM PROMPT TEXT") is True

    body = sent[0]
    assert body["messages"][0]["content"] == "SYSTEM PROMPT TEXT"
    assert body["options"]["num_predict"] == 1  # one token, not a paragraph
    assert not body.get("tools")


def test_warming_an_openai_server_caps_max_tokens(config):
    from voicevibecoder.codegen.local_brain import LocalBrain

    sent = []

    def server(url, body, timeout=None):
        if url.endswith("/v1/models"):
            return {"data": [{"id": "m"}]}
        sent.append(body)
        return {"choices": [{"message": {"content": "ok"}}]}

    brain = LocalBrain(config.merged(local_api="openai"), request=server)
    assert brain.warm("SYSTEM") is True
    assert sent[0]["max_tokens"] == 1


def test_warming_leaves_the_conversation_untouched(config):
    """A warm-up is not a turn; the first real instruction must start clean."""
    from voicevibecoder.codegen.local_brain import LocalBrain

    def server(url, body, timeout=None):
        if url.endswith("/api/tags"):
            return {"models": [{"name": "m"}]}
        return {"message": {"content": "ready"}}

    brain = LocalBrain(config, request=server)
    brain.warm("SYSTEM")
    assert brain._history == []


def test_a_failed_warm_up_is_harmless(config):
    from voicevibecoder.codegen.local_brain import LocalBrain

    def server(url, body, timeout=None):
        if url.endswith("/api/tags"):
            return {"models": [{"name": "m"}]}
        raise OSError("server busy loading")

    assert LocalBrain(config, request=server).warm("SYSTEM") is False
