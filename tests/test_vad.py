"""Endpointing, driven with synthetic audio."""

from __future__ import annotations

import numpy as np

from voicevibecoder.audio.vad import Endpointer, frame_energy

RATE = 16000
FRAME_MS = 30
FRAME = RATE * FRAME_MS // 1000


def frames(seconds: float, amplitude: float, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    count = int(seconds * 1000 / FRAME_MS)
    return [rng.normal(0, amplitude, FRAME).astype(np.float32) for _ in range(count)]


def feed(endpointer: Endpointer, batch: list[np.ndarray]) -> list:
    return [seg for frame in batch if (seg := endpointer.push(frame)) is not None]


def test_silence_alone_produces_no_utterance():
    endpointer = Endpointer(hangover_ms=300)
    assert feed(endpointer, frames(2.0, 0.0005)) == []


def test_speech_between_silences_is_one_utterance():
    endpointer = Endpointer(hangover_ms=300)
    segments = feed(endpointer, frames(0.6, 0.0005))
    segments += feed(endpointer, frames(1.0, 0.2, seed=1))
    segments += feed(endpointer, frames(0.9, 0.0005, seed=2))

    assert len(segments) == 1
    # Pre-roll means the segment is a little longer than the speech itself.
    assert segments[0].duration_s >= 1.0


def test_a_pause_between_words_does_not_split_the_utterance():
    endpointer = Endpointer(hangover_ms=600)
    segments = feed(endpointer, frames(0.5, 0.0005))
    segments += feed(endpointer, frames(0.5, 0.2, seed=1))
    segments += feed(endpointer, frames(0.3, 0.0005, seed=2))  # a breath
    segments += feed(endpointer, frames(0.5, 0.2, seed=3))
    segments += feed(endpointer, frames(0.9, 0.0005, seed=4))

    assert len(segments) == 1


def test_two_sentences_are_two_utterances():
    endpointer = Endpointer(hangover_ms=300)
    segments = feed(endpointer, frames(0.5, 0.0005))
    segments += feed(endpointer, frames(0.6, 0.2, seed=1))
    segments += feed(endpointer, frames(1.0, 0.0005, seed=2))
    segments += feed(endpointer, frames(0.6, 0.2, seed=3))
    segments += feed(endpointer, frames(1.0, 0.0005, seed=4))

    assert len(segments) == 2


def test_a_click_is_too_short_to_count():
    endpointer = Endpointer(hangover_ms=300, min_utterance_ms=400)
    segments = feed(endpointer, frames(0.5, 0.0005))
    segments += feed(endpointer, frames(0.06, 0.3, seed=1))  # door slam
    segments += feed(endpointer, frames(0.9, 0.0005, seed=2))

    assert segments == []


def test_the_noise_floor_adapts_to_a_louder_room():
    quiet = Endpointer()
    feed(quiet, frames(1.0, 0.0005))
    loud = Endpointer()
    feed(loud, frames(1.0, 0.01))

    assert loud.noise_floor > quiet.noise_floor
    assert loud.open_threshold > quiet.open_threshold


def test_speech_that_never_stops_is_cut_at_the_ceiling():
    endpointer = Endpointer(max_utterance_ms=600, hangover_ms=3000)
    segments = feed(endpointer, frames(2.0, 0.2, seed=1))
    assert len(segments) >= 2


def test_flush_emits_what_is_buffered():
    endpointer = Endpointer(hangover_ms=3000)
    feed(endpointer, frames(0.4, 0.0005))
    feed(endpointer, frames(0.8, 0.2, seed=1))
    assert endpointer.speaking
    assert endpointer.flush() is not None
    assert endpointer.flush() is None


def test_frame_energy_of_silence_is_zero():
    assert frame_energy(np.zeros(FRAME, dtype=np.float32)) == 0.0
    assert frame_energy(np.array([], dtype=np.float32)) == 0.0
