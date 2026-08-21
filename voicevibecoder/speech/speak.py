"""Spoken feedback.

Voice in, voice out: if you are looking at the whiteboard rather than the
screen, a build that finished silently may as well not have finished. Every
backend is optional and the ``NullSpeaker`` is always available, so nothing
here can ever be the reason a session fails to start.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Protocol, runtime_checkable

from voicevibecoder.config import Config

# Long summaries are for the eyes; the ear wants one sentence.
MAX_SPOKEN_CHARS = 240


@runtime_checkable
class Speaker(Protocol):
    def say(self, text: str) -> None:
        """Speak ``text``. Must never raise — feedback is not critical path."""


class NullSpeaker:
    """Says nothing, successfully."""

    def say(self, text: str) -> None:
        pass


class CommandSpeaker:
    """Speaks via an external binary (`say` on macOS, `espeak` on Linux)."""

    def __init__(self, argv: list[str]) -> None:
        self._argv = argv

    def say(self, text: str) -> None:
        try:
            subprocess.run(  # noqa: S603 — argv is built from a fixed table
                [*self._argv, shorten(text)],
                check=False,
                timeout=30,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError):
            pass


class Pyttsx3Speaker:
    """Cross-platform offline TTS."""

    def __init__(self) -> None:
        import pyttsx3  # noqa: PLC0415

        self._engine = pyttsx3.init()

    def say(self, text: str) -> None:
        try:
            self._engine.say(shorten(text))
            self._engine.runAndWait()
        except Exception:  # noqa: BLE001 — a mute assistant beats a crashed one
            pass


def shorten(text: str, limit: int = MAX_SPOKEN_CHARS) -> str:
    """Trim to a listenable length, preferring a sentence boundary."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    head = text[:limit]
    cut = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
    if cut > limit // 2:
        return head[: cut + 1]
    return head.rsplit(" ", 1)[0] + "..."


def build_speaker(config: Config) -> Speaker:
    """Best available backend for this machine, or silence."""
    wanted = config.tts_backend
    if wanted == "off":
        return NullSpeaker()

    if wanted in ("auto", "pyttsx3"):
        try:
            return Pyttsx3Speaker()
        except Exception:  # noqa: BLE001 — fall through to a CLI backend
            if wanted != "auto":
                return NullSpeaker()

    for name, argv in (("say", ["say"]), ("espeak", ["espeak", "-s", "165"])):
        if wanted in ("auto", name) and shutil.which(name):
            return CommandSpeaker(argv)

    return NullSpeaker()
