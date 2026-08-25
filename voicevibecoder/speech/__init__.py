"""Speech in (ASR) and speech out (TTS).

Imported lazily: the transcription backends need numpy, and the Android path
never touches them — Termux:API hands over finished text. Nothing should have
to install a compiled dependency for a code path it will not run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from voicevibecoder.speech.listen import Transcriber, build_transcriber
    from voicevibecoder.speech.speak import Speaker, build_speaker

__all__ = ["Speaker", "Transcriber", "build_speaker", "build_transcriber"]

_LAZY = {
    "Transcriber": "voicevibecoder.speech.listen",
    "build_transcriber": "voicevibecoder.speech.listen",
    "Speaker": "voicevibecoder.speech.speak",
    "build_speaker": "voicevibecoder.speech.speak",
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(_LAZY[name]), name)
