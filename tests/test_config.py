"""Configuration layering."""

from __future__ import annotations

from voicevibecoder import config as config_module


def test_defaults_are_sane():
    config = config_module.Config()
    assert config.model == "claude-opus-5"
    assert config.frame_samples == 480  # 30 ms at 16 kHz


def test_file_then_env_then_flags(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "voicevibe.toml").write_text(
        '[voicevibe]\neffort = "low"\nidea_bar = 50\nauto_run = false\n'
    )
    monkeypatch.setenv("VVC_IDEA_BAR", "70")

    config = config_module.load(workspace=workspace, effort="max")
    assert config.effort == "max"      # flag beats file
    assert config.idea_bar == 70       # env beats file
    assert config.auto_run is False    # file beats default


def test_unknown_keys_in_the_file_are_ignored(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "voicevibe.toml").write_text('nonsense = 1\nmodel = "claude-sonnet-5"\n')
    assert config_module.load(workspace=workspace).model == "claude-sonnet-5"


def test_merged_rejects_unknown_overrides():
    try:
        config_module.Config().merged(banana=True)
    except TypeError as exc:
        assert "banana" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected a TypeError")


def test_none_overrides_are_ignored():
    config = config_module.Config().merged(model=None, effort="low")
    assert config.model == "claude-opus-5"
    assert config.effort == "low"
