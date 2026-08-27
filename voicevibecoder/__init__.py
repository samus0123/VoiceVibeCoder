"""VoiceVibeCoder — speak a program into existence.

The package is deliberately layered so that every stage of the pipeline can be
swapped or unit tested in isolation:

    microphone -> endpointer -> transcriber -> normalizer -> intent grammar
              -> (deterministic command)  or  (Claude codegen -> workspace)
              -> runner -> spoken feedback

Only the leaf modules import optional heavyweight dependencies (sounddevice,
faster-whisper, anthropic), and they do so lazily, so importing
``voicevibecoder`` never requires a microphone or an API key.
"""

import sys

if sys.version_info < (3, 11):  # noqa: UP036 — this is the check, not a shim
    # Said plainly here because the alternative is a ModuleNotFoundError for
    # tomllib, which tells a person nothing about what to do next.
    raise SystemExit(
        "VoiceVibeCoder needs Python 3.11 or newer; this is "
        f"{'.'.join(str(part) for part in sys.version_info[:3])}.\n"
        "  Termux:  pkg upgrade python\n"
        "  Debian:  sudo apt install python3.11"
    )

from voicevibecoder.config import Config  # noqa: E402 — must follow the guard

__version__ = "0.1.0"
__all__ = ["Config", "__version__"]
