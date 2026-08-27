"""Configuration for a VoiceVibeCoder session.

Precedence, lowest to highest: dataclass defaults -> ``voicevibe.toml`` in the
workspace -> ``VVC_*`` environment variables -> command line flags.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

CONFIG_FILENAME = "voicevibe.toml"
STATE_DIRNAME = ".voicevibe"


@dataclass(frozen=True)
class Config:
    """Everything a session needs to know, in one immutable record."""

    # --- workspace -------------------------------------------------------
    workspace: Path = Path("./vibe-workspace")

    # --- brain -----------------------------------------------------------
    brain: str = "auto"  # auto | claude | local
    model: str = "claude-opus-5"
    effort: str = "high"  # low | medium | high | xhigh | max
    max_tokens: int = 32000
    server_side_fallback: bool = True  # rescue a policy refusal on the API side

    # --- local brain (Ollama-compatible, no key and no network) ----------
    local_model: str = "qwen2.5-coder:7b"
    local_api: str = "auto"  # auto | ollama | openai (llama.cpp, LM Studio, vLLM)
    local_url: str = "http://localhost:11434"
    local_timeout_s: float = 300.0

    # --- audio capture ---------------------------------------------------
    sample_rate: int = 16000
    frame_ms: int = 30
    input_device: int | None = None
    # An utterance ends after this much trailing silence.
    hangover_ms: int = 900
    # Ignore blips shorter than this; they are almost always door slams.
    min_utterance_ms: int = 350
    # Hard ceiling so a stuck stream cannot buffer forever.
    max_utterance_ms: int = 30000
    # Speech starts when frame energy exceeds noise_floor * this ratio.
    speech_ratio: float = 3.5

    # --- speech ----------------------------------------------------------
    asr_backend: str = "auto"  # auto | whisper | termux | text
    whisper_model: str = "base.en"
    whisper_compute_type: str = "int8"
    language: str = "en"
    tts_backend: str = "auto"  # auto | pyttsx3 | say | espeak | termux | off

    # --- behaviour -------------------------------------------------------
    wake_phrase: str | None = None  # e.g. "hey vibe"; None = always listening
    auto_run: bool = True  # run the program after a successful build
    self_heal_attempts: int = 2  # feed tracebacks back to Claude this many times
    run_timeout_s: float = 30.0
    confirm_destructive: bool = True  # "delete X" needs a spoken "yes"
    idea_bar: int = 80  # ideas scoring below this are never spoken aloud
    suggest_improvements: bool = True  # volunteer one idea after each build
    git: bool = True  # commit every accepted change
    commit_mode: str = "auto"  # auto | typed | off
    ai_trailer: bool = True  # mark generated commits as AI-assisted
    history_turns: int = 12  # build turns kept in the model conversation

    # ------------------------------------------------------------------
    @property
    def frame_samples(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)

    @property
    def state_dir(self) -> Path:
        return self.workspace / STATE_DIRNAME

    def merged(self, **overrides: Any) -> Config:
        """Return a copy with the non-``None`` overrides applied."""
        clean = {k: v for k, v in overrides.items() if v is not None}
        unknown = set(clean) - {f.name for f in fields(self)}
        if unknown:
            raise TypeError(f"unknown config keys: {', '.join(sorted(unknown))}")
        return replace(self, **clean)


def _coerce(raw: Any, target: Any) -> Any:
    """Coerce a TOML/env scalar to the type of the current field value."""
    if target is None or isinstance(target, str):
        return None if raw in ("", "none", "None") else str(raw)
    if isinstance(target, Path):
        return Path(str(raw)).expanduser()
    if isinstance(target, bool):
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(target, int):
        return int(raw)
    if isinstance(target, float):
        return float(raw)
    return raw


def load(workspace: Path | str | None = None, **overrides: Any) -> Config:
    """Build a Config from file + environment + explicit overrides."""
    cfg = Config()
    if workspace is not None:
        cfg = cfg.merged(workspace=Path(workspace).expanduser())

    file_values = _from_file(cfg.workspace / CONFIG_FILENAME)
    if file_values:
        cfg = cfg.merged(**file_values)

    env_values = _from_env(cfg)
    if env_values:
        cfg = cfg.merged(**env_values)

    return cfg.merged(**overrides)


def _from_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    # Accept both a flat table and a [voicevibe] section.
    data = data.get("voicevibe", data)
    known = {f.name: getattr(Config(), f.name) for f in fields(Config)}
    return {
        key: _coerce(value, known[key])
        for key, value in data.items()
        if key in known
    }


def _from_env(cfg: Config) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for field in fields(cfg):
        raw = os.environ.get(f"VVC_{field.name.upper()}")
        if raw is not None:
            values[field.name] = _coerce(raw, getattr(cfg, field.name))
    return values
