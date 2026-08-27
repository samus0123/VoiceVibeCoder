"""``voicevibe --doctor``: one command that says what is and is not working.

Written after one too many "it did not start". A program that runs on phones,
laptops and locked-down boxes, with two possible brains and three possible ways
of listening, has a lot of ways to be *almost* set up — and the failure modes
all look identical from the outside. This prints every one of them at once, so
a single paste answers the question.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

from voicevibecoder.config import Config

OK = "ok"
MISSING = "--"


def report(config: Config) -> str:
    """The whole diagnostic, as text meant to be pasted somewhere."""
    lines = ["VoiceVibeCoder — diagnostic", ""]
    for heading, rows in (
        ("runtime", _runtime()),
        ("packages", _packages()),
        ("brains", _brains(config)),
        ("listening", _listening(config)),
        ("workspace", _workspace(config)),
    ):
        lines.append(f"{heading}:")
        lines.extend(f"  {label:<22} {value}" for label, value in rows)
        lines.append("")
    lines.append(_verdict(config))
    return "\n".join(lines)


def _runtime() -> list[tuple[str, str]]:
    version = ".".join(str(part) for part in sys.version_info[:3])
    supported = "ok" if sys.version_info >= (3, 11) else "TOO OLD — needs 3.11+"
    return [
        ("python", f"{version}  {supported}"),
        ("executable", sys.executable),
        ("platform", sys.platform),
        ("termux", "yes" if _is_termux() else "no"),
        ("git", shutil.which("git") or MISSING),
    ]


def _packages() -> list[tuple[str, str]]:
    return [
        (name, OK if importlib.util.find_spec(name) else f"{MISSING}  ({why})")
        for name, why in (
            ("anthropic", "needed for --brain claude"),
            ("numpy", "needed for microphone capture"),
            ("sounddevice", "needed for microphone capture"),
            ("faster_whisper", "needed for local transcription"),
            ("pyttsx3", "optional spoken replies"),
        )
    ]


def _brains(config: Config) -> list[tuple[str, str]]:
    rows = [("configured", config.brain)]

    if importlib.util.find_spec("anthropic"):
        try:
            from voicevibecoder.codegen.claude_brain import _make_client  # noqa: PLC0415

            _make_client()
            rows.append((config.model, "ok — credentials found"))
        except Exception as exc:  # noqa: BLE001 — this is the diagnosis
            rows.append((config.model, f"{MISSING}  {_short(exc)}"))
    else:
        rows.append((config.model, f"{MISSING}  pip install -e '.[claude]'"))

    from voicevibecoder.codegen.local_brain import LocalBrain  # noqa: PLC0415

    local = LocalBrain(config)
    installed = local.installed_models()
    if installed is None:
        rows.append((config.local_url, f"{MISSING}  no server reachable"))
    else:
        rows.append((config.local_url, f"ok — {len(installed)} model(s)"))
        rows.append((config.local_model, local.setup_help()))
    return rows


def _listening(config: Config) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if _is_termux():
        for command in ("termux-speech-to-text", "termux-tts-speak"):
            found = shutil.which(command)
            rows.append((command, found or f"{MISSING}  pkg install termux-api"))
    else:
        rows.append(("microphone", "sounddevice" if importlib.util.find_spec("sounddevice") else f"{MISSING}  pip install -e '.[voice]'"))
        rows.append(("transcription", config.whisper_model if importlib.util.find_spec("faster_whisper") else f"{MISSING}  pip install -e '.[voice]'"))
    rows.append(("typed input", "ok — always available (--text)"))
    return rows


def _workspace(config: Config) -> list[tuple[str, str]]:
    root = Path(config.workspace).expanduser()
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".voicevibe-write-test"
        probe.write_text("x", "utf-8")
        probe.unlink()
        writable = "ok — writable"
    except OSError as exc:
        writable = f"NOT WRITABLE — {exc}"
    return [("path", str(root)), ("access", writable)]


def _verdict(config: Config) -> str:
    """The one line that answers "so can I use it or not"."""
    from voicevibecoder.codegen.local_brain import LocalBrain  # noqa: PLC0415

    has_claude = bool(importlib.util.find_spec("anthropic")) and _claude_ready()
    has_local = LocalBrain(config).available()

    if has_claude or has_local:
        which = config.model if has_claude else config.local_model
        return f"VERDICT: ready to build, using {which}."
    return (
        "VERDICT: it will start, and everything except building works — but no "
        "brain is reachable yet.\n"
        "  Either:  pip install -e '.[claude]'  and set ANTHROPIC_API_KEY\n"
        "  Or:      run Ollama and point at it:  --local-url http://<host>:11434"
    )


def _claude_ready() -> bool:
    try:
        from voicevibecoder.codegen.claude_brain import _make_client  # noqa: PLC0415

        _make_client()
    except Exception:  # noqa: BLE001
        return False
    return True


def _is_termux() -> bool:
    return bool(os.environ.get("TERMUX_VERSION")) or "com.termux" in os.environ.get(
        "PREFIX", ""
    )


def _short(exc: Exception) -> str:
    text = str(exc).strip().splitlines()
    return text[0][:120] if text else exc.__class__.__name__
