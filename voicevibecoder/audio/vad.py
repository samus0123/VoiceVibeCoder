"""Energy-based voice activity detection with adaptive noise tracking.

Why not a neural VAD? Because the job here is *endpointing* — deciding when a
sentence started and stopped — not classifying speech in noise. A calibrated
energy gate with hysteresis and a pre-roll buffer does that in microseconds,
with no model to download, and it is trivially testable with synthetic audio.

The three ideas that make it behave well in a real room:

1. **Adaptive noise floor.** The floor is re-estimated from silent frames only,
   with a fast attack / slow release EMA, so a laptop fan spinning up does not
   permanently trip the gate.
2. **Hysteresis.** Speech must exceed ``floor * speech_ratio`` to open the gate
   but only fall below a lower release threshold to close it, so the natural
   dip between words does not chop an utterance in half.
3. **Pre-roll.** The frames immediately before the trigger are kept, so the
   plosive at the start of "print this" is never clipped.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Segment:
    """One endpointed utterance."""

    audio: np.ndarray
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.sample_rate


def frame_energy(frame: np.ndarray) -> float:
    """Root-mean-square energy of a float32 frame in [-1, 1]."""
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))


class Endpointer:
    """Feed it fixed-size frames; it hands back complete utterances.

    ``push()`` returns a :class:`Segment` on the frame that closes an
    utterance and ``None`` otherwise, which keeps the caller a simple loop.
    """

    # The gate closes at a lower threshold than it opens (hysteresis).
    RELEASE_FACTOR = 0.6
    # Absolute floor so a pin-drop-silent room does not make the gate hair
    # triggered: 16-bit dither sits around 3e-4 RMS.
    MIN_NOISE_FLOOR = 3e-4

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        hangover_ms: int = 900,
        min_utterance_ms: int = 350,
        max_utterance_ms: int = 30000,
        speech_ratio: float = 3.5,
        preroll_ms: int = 240,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.speech_ratio = speech_ratio
        self._hangover_frames = max(1, hangover_ms // frame_ms)
        self._min_frames = max(1, min_utterance_ms // frame_ms)
        self._max_frames = max(self._min_frames, max_utterance_ms // frame_ms)
        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, preroll_ms // frame_ms))

        self._noise_floor = self.MIN_NOISE_FLOOR
        self._speaking = False
        self._silence_run = 0
        self._voiced: list[np.ndarray] = []

    # -- introspection, useful for meters and tests ----------------------
    @property
    def noise_floor(self) -> float:
        return self._noise_floor

    @property
    def speaking(self) -> bool:
        return self._speaking

    @property
    def open_threshold(self) -> float:
        return self._noise_floor * self.speech_ratio

    # -- main loop -------------------------------------------------------
    def push(self, frame: np.ndarray) -> Segment | None:
        """Consume one frame of audio, maybe emitting a finished utterance."""
        energy = frame_energy(frame)

        if not self._speaking:
            self._track_noise(energy)
            self._preroll.append(frame)
            if energy > self.open_threshold:
                self._speaking = True
                self._silence_run = 0
                self._voiced = list(self._preroll)
                self._preroll.clear()
            return None

        self._voiced.append(frame)
        if energy > self.open_threshold * self.RELEASE_FACTOR:
            self._silence_run = 0
        else:
            self._silence_run += 1

        if self._silence_run >= self._hangover_frames:
            return self._close()
        if len(self._voiced) >= self._max_frames:
            # Hard cap: emit what we have and start a fresh utterance so a
            # monologue is still transcribed instead of buffered forever.
            return self._close()
        return None

    def flush(self) -> Segment | None:
        """Close the current utterance, if any (end of stream)."""
        return self._close() if self._speaking else None

    # -- internals -------------------------------------------------------
    def _track_noise(self, energy: float) -> None:
        # Fast attack toward quieter rooms, slow release toward louder ones:
        # a genuinely quiet moment should lower the floor quickly, while a
        # single loud frame must not raise it.
        alpha = 0.25 if energy < self._noise_floor else 0.02
        self._noise_floor = max(
            self.MIN_NOISE_FLOOR,
            (1 - alpha) * self._noise_floor + alpha * energy,
        )

    def _close(self) -> Segment | None:
        voiced, self._voiced = self._voiced, []
        self._speaking = False
        self._silence_run = 0
        self._preroll.clear()
        if len(voiced) < self._min_frames:
            return None  # a cough, a chair, a keyboard clack
        return Segment(np.concatenate(voiced), self.sample_rate)
