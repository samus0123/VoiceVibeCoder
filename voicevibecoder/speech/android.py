"""Android support, via Termux and the Termux:API companion app.

A phone has no PortAudio and no CTranslate2, so the desktop chain — capture,
endpoint, transcribe — does not port. It does not need to: Android already
does all three behind one system service, and Termux:API exposes it as a
command that blocks until you stop talking and prints what you said.

That changes where this plugs in. On a laptop the microphone is a *source of
audio* and Whisper is a *transcriber*; on Android the platform hands over
finished text, so this module is a sibling of ``cli.spoken_lines`` — an
utterance source — rather than another :class:`Transcriber`. Everything
downstream (NLP, grammar, workspace, runner) is unchanged and unaware.

One honest trade-off: ``termux-speech-to-text`` uses Android's recognition
service, which on most phones is Google's and may do the recognition in the
cloud. The desktop path keeps dictated code on the machine; this one does not,
unless you have installed offline recognition for your language.

Requires:
    the Termux app *and* the Termux:API app, both from F-Droid (the Play Store
    builds are stale and cannot talk to each other), then:
        pkg install termux-api
        termux-setup-storage      # prompts for permissions
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
from collections.abc import Callable, Iterator

from voicevibecoder.config import Config
from voicevibecoder.speech.speak import shorten

LISTEN_COMMAND = "termux-speech-to-text"
SPEAK_COMMAND = "termux-tts-speak"

# Recognition returns as soon as Android decides the utterance ended; the
# timeout is only a backstop against a wedged service.
LISTEN_TIMEOUT_S = 120

SETUP_HELP = (
    f"{LISTEN_COMMAND} was not found.\n"
    "On Android, VoiceVibeCoder listens through Termux:API:\n"
    "  1. install Termux *and* Termux:API from F-Droid (not the Play Store)\n"
    "  2. pkg install termux-api\n"
    "  3. grant the microphone permission when first prompted\n"
    "Or run without a microphone:  voicevibe --text"
)

Runner = Callable[[list[str]], subprocess.CompletedProcess]


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603 — argv is a fixed command plus its text
        argv, capture_output=True, text=True, check=False, timeout=LISTEN_TIMEOUT_S
    )


def is_termux() -> bool:
    """True when running inside Termux on Android."""
    import os  # noqa: PLC0415 — cheap, and keeps the import surface local

    return bool(os.environ.get("TERMUX_VERSION")) or "com.termux" in os.environ.get(
        "PREFIX", ""
    )


def has_termux_api() -> bool:
    return shutil.which(LISTEN_COMMAND) is not None


class TermuxSpeaker:
    """Spoken responses through Android's text-to-speech engine."""

    def __init__(self, run: Runner = _run) -> None:
        self._run = run

    def say(self, text: str) -> None:
        spoken = shorten(text)
        if not spoken:
            return
        # A mute assistant beats a crashed one.
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            self._run([SPEAK_COMMAND, spoken])


def utterances(
    config: Config,
    run: Runner = _run,
    on_status: Callable[[str], None] | None = None,
) -> Iterator[str]:
    """Yield recognised utterances, one Android recognition session at a time.

    Android listens once per invocation, so this is a loop of blocking calls
    rather than a stream. A silent session yields nothing and simply listens
    again, which is what makes it usable hands-free.
    """
    if not has_termux_api():
        raise RuntimeError(SETUP_HELP)

    say = on_status or (lambda _message: None)
    consecutive_failures = 0

    while True:
        try:
            result = run([LISTEN_COMMAND])
        except subprocess.TimeoutExpired:
            say("recognition timed out; listening again")
            continue
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"{LISTEN_COMMAND} failed: {exc}") from exc

        if result.returncode != 0:
            consecutive_failures += 1
            message = (result.stderr or "").strip() or f"exit code {result.returncode}"
            # Three in a row is a broken setup, not a bad utterance: stop
            # rather than spin on a missing permission forever.
            if consecutive_failures >= 3:
                raise RuntimeError(
                    f"{LISTEN_COMMAND} keeps failing ({message}).\n"
                    "Check that Termux:API is installed and the microphone "
                    "permission is granted."
                )
            say(f"recognition failed ({message}); listening again")
            continue

        consecutive_failures = 0
        text = (result.stdout or "").strip()
        if text:
            yield text
