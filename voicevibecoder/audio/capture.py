"""Microphone capture built on ``sounddevice``.

Imported lazily and behind a clear error message, so the rest of the package
(and the whole test suite) runs on a machine with no audio hardware at all.
"""

from __future__ import annotations

import queue
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

from voicevibecoder.audio.vad import Endpointer, Segment
from voicevibecoder.config import Config

_MISSING = (
    "Microphone capture needs the 'sounddevice' package and PortAudio.\n"
    "  pip install 'voicevibecoder[voice]'      # sounddevice + faster-whisper\n"
    "  # Debian/Ubuntu also needs:  sudo apt install libportaudio2\n"
    "Without a microphone you can still drive VoiceVibeCoder from the keyboard:\n"
    "  voicevibe --text"
)


def _sounddevice():
    try:
        import sounddevice  # noqa: PLC0415 — optional dependency, imported on demand
    except (ImportError, OSError) as exc:  # OSError: PortAudio not installed
        raise RuntimeError(_MISSING) from exc
    return sounddevice


def list_devices() -> str:
    """Human-readable table of input devices, for ``--list-devices``."""
    return str(_sounddevice().query_devices())


@contextmanager
def open_stream(config: Config) -> Iterator[queue.Queue[np.ndarray]]:
    """Yield a queue that fills with mono float32 frames from the mic."""
    sd = _sounddevice()
    frames: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, _frame_count, _time_info, status) -> None:
        if status:  # overflow/underflow — keep going, the audio is still usable
            pass
        frames.put(indata[:, 0].copy())

    stream = sd.InputStream(
        samplerate=config.sample_rate,
        blocksize=config.frame_samples,
        device=config.input_device,
        channels=1,
        dtype="float32",
        callback=callback,
    )
    with stream:
        yield frames


def utterances(config: Config) -> Iterator[Segment]:
    """Endless stream of endpointed utterances from the default microphone."""
    endpointer = Endpointer(
        sample_rate=config.sample_rate,
        frame_ms=config.frame_ms,
        hangover_ms=config.hangover_ms,
        min_utterance_ms=config.min_utterance_ms,
        max_utterance_ms=config.max_utterance_ms,
        speech_ratio=config.speech_ratio,
    )
    with open_stream(config) as frames:
        while True:
            segment = endpointer.push(frames.get())
            if segment is not None:
                yield segment
