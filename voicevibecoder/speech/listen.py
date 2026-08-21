"""Speech-to-text backends.

``Transcriber`` is a one-method protocol, which is the whole point: the local
Whisper backend, a canned test double, and any future cloud backend are
interchangeable, and nothing upstream knows which one is running.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol, runtime_checkable

import numpy as np

from voicevibecoder.config import Config


@runtime_checkable
class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Return the text spoken in ``audio`` (empty string if nothing was)."""


class WhisperTranscriber:
    """Local, offline transcription via ``faster-whisper``.

    Local by default is a deliberate choice: dictating source code means
    dictating credentials, customer names and unreleased ideas sooner or
    later, and none of that needs to leave the machine to become text.
    """

    def __init__(self, config: Config) -> None:
        try:
            from faster_whisper import WhisperModel  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "Local transcription needs faster-whisper:\n"
                "  pip install 'voicevibecoder[voice]'"
            ) from exc
        self._config = config
        self._model = WhisperModel(
            config.whisper_model, compute_type=config.whisper_compute_type
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if sample_rate != 16000:
            audio = resample_to_16k(audio, sample_rate)
        segments, _info = self._model.transcribe(
            audio.astype(np.float32),
            language=self._config.language or None,
            vad_filter=False,  # we already endpointed the utterance ourselves
            condition_on_previous_text=False,  # commands are independent
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


class ScriptedTranscriber:
    """Returns pre-baked lines in order — the test/demo double."""

    def __init__(self, lines: Iterable[str]) -> None:
        self._lines: Iterator[str] = iter(list(lines))

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:  # noqa: ARG002
        return next(self._lines, "")


def resample_to_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Linear resample to 16 kHz — plenty for speech, and dependency free."""
    if sample_rate == 16000:
        return audio
    duration = len(audio) / sample_rate
    target_len = max(1, int(duration * 16000))
    source_grid = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    target_grid = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(target_grid, source_grid, audio).astype(np.float32)


def build_transcriber(config: Config) -> Transcriber:
    """Pick a backend from config, failing with an actionable message."""
    if config.asr_backend in ("auto", "whisper"):
        return WhisperTranscriber(config)
    if config.asr_backend == "text":
        raise RuntimeError("the 'text' backend reads the keyboard; use --text")
    raise ValueError(f"unknown asr_backend: {config.asr_backend!r}")
