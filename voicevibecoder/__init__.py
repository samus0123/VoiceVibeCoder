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

from voicevibecoder.config import Config

__version__ = "0.1.0"
__all__ = ["Config", "__version__"]
