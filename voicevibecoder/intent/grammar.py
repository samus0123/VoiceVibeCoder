"""A deterministic command grammar, tried before the model is ever called.

The central design decision of VoiceVibeCoder: **control is a grammar, code is
a conversation.** "undo that" must undo, every single time, in 200 microseconds
and for zero cents — inferring it with a language model would be slower, more
expensive and less reliable. So a small ordered table of anchored patterns
claims the closed set of control phrases, and *everything else* falls through
to :data:`Kind.BUILD`, which is the open-ended "describe the program you want"
channel.

Anchoring is what keeps the two apart. ``run it`` is a command; ``run the
simulation ten thousand times`` is a specification, and the ``^run (it|that)$``
pattern correctly refuses the second one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Kind(Enum):
    """What the user wants to happen."""

    BUILD = "build"          # open-ended: hand to Claude, it writes files
    EXPLAIN = "explain"      # ask about the code, do not modify it
    RUN = "run"              # execute the current entry point
    STOP = "stop"            # interrupt whatever is running
    UNDO = "undo"            # restore the previous snapshot
    LIST = "list"            # what files exist
    SHOW = "show"            # read a file back aloud
    DELETE = "delete"        # remove a file (confirmed)
    ENTRYPOINT = "entrypoint"  # declare what "run it" means
    NEW_PROJECT = "new_project"
    DICTATE_START = "dictate_start"  # literal text mode
    DICTATE_END = "dictate_end"
    REPEAT = "repeat"        # say the last response again
    YES = "yes"
    NO = "no"
    HELP = "help"
    QUIT = "quit"
    SILENCE = "silence"      # nothing was said


@dataclass(frozen=True)
class Intent:
    kind: Kind
    text: str = ""                       # the utterance, post-normalisation
    slots: dict[str, str] = field(default_factory=dict)

    def slot(self, name: str, default: str = "") -> str:
        return self.slots.get(name, default)


# A file reference spoken out loud: at most six tokens, and validated by
# ``looks_like_file_ref`` so that prose never sneaks into a file slot.
_FILE_REF = r"(?P<target>[\w.\-/]+(?:\s+[\w.\-/]+){0,5})"

_SYMBOL_WORDS = frozenset({"dot", "point", "slash", "dash", "hyphen", "underscore"})


def looks_like_file_ref(text: str) -> bool:
    """True if ``text`` plausibly names a file rather than describing work.

    "main dot pie", "app.py", "utils" are file references; "the simulation ten
    thousand times" is not, and must fall through to a build instruction.
    """
    text = text.strip()
    if not text:
        return False
    tokens = text.split()
    if any(token.lower().strip(".,") in _SYMBOL_WORDS for token in tokens):
        return True
    if any(char in text for char in "./"):
        return True
    return len(tokens) == 1


# (pattern, kind, needs_file_ref). Order matters: the first match wins, so narrow patterns
# come before broad ones. Every pattern is fully anchored — a control phrase
# is the *entire* utterance, never a prefix of a build instruction.
_RULES: tuple[tuple[str, Kind, bool], ...] = (
    (r"(?:quit|exit|goodbye|good bye|stop listening|we(?:'re| are) done|that's all)", Kind.QUIT, False),
    (r"(?:help|what can (?:you|i) (?:do|say)|(?:list|show) (?:the )?commands)", Kind.HELP, False),

    (r"(?:end|stop|finish) dictation", Kind.DICTATE_END, False),
    (rf"(?:start )?dictat(?:e|ion)(?: (?:in)?to (?:the )?(?:file )?{_FILE_REF})?", Kind.DICTATE_START, True),

    (r"(?:run|execute|start|try) (?:it|that|this|the (?:program|code|app|script|thing))", Kind.RUN, False),
    (rf"(?:run|execute) (?:the (?:file )?)?{_FILE_REF}", Kind.RUN, True),
    (r"(?:stop|halt|kill|abort|cancel)(?: (?:it|that|the (?:program|run|script)))?", Kind.STOP, False),

    (r"(?:undo|revert|roll ?back|scratch that|never ?mind|take that back|go back)"
     r"(?: (?:that|it|the last (?:change|thing|edit)))?", Kind.UNDO, False),

    (r"(?:list|show me|what(?:'s| is) in) (?:the )?files?", Kind.LIST, False),
    (r"what files (?:do we have|are there|exist)", Kind.LIST, False),
    (rf"(?:show|open|read(?: back)?|display|print) (?:me )?(?:the )?(?:file )?{_FILE_REF}", Kind.SHOW, True),
    (rf"(?:delete|remove|trash|throw away) (?:the )?(?:file )?{_FILE_REF}", Kind.DELETE, True),

    (rf"(?:the )?(?:main|entry ?point)(?: file)? is {_FILE_REF}", Kind.ENTRYPOINT, True),
    (r"(?:new|start|create|begin)(?: a)?(?: new)? (?:project|app|program|thing)"
     r"(?: (?:called|named))? (?P<name>[\w .\-/]{1,60})", Kind.NEW_PROJECT, False),

    (r"(?:explain|describe|what does|how does|why does|walk me through)\b(?P<question>.*)", Kind.EXPLAIN, False),

    (r"(?:repeat(?: that)?|say (?:that )?again|what did you say)", Kind.REPEAT, False),
    (r"(?:yes|yeah|yep|yup|sure|do it|confirm|go ahead|affirmative)", Kind.YES, False),
    (r"(?:no|nope|don't|do not|negative|forget it)", Kind.NO, False),
)

_COMPILED = tuple(
    (re.compile(rf"^\s*{pattern}\s*[.!?]?\s*$", re.IGNORECASE), kind, needs_file)
    for pattern, kind, needs_file in _RULES
)


def parse(text: str, wake_phrase: str | None = None) -> Intent:
    """Classify one utterance.

    ``wake_phrase``, when set, is stripped from the front of the utterance;
    gating on it is the caller's job (see :mod:`voicevibecoder.session`).
    """
    text = (text or "").strip()
    if wake_phrase:
        text = strip_wake_phrase(text, wake_phrase)
    if not text:
        return Intent(Kind.SILENCE)

    for pattern, kind, needs_file in _COMPILED:
        match = pattern.match(text)
        if not match:
            continue
        slots = {k: v.strip() for k, v in match.groupdict().items() if v and v.strip()}
        if needs_file and "target" in slots and not looks_like_file_ref(slots["target"]):
            continue  # prose in a file slot: this was a build instruction
        if kind is Kind.EXPLAIN and not slots.get("question"):
            continue  # a bare "explain" is not a question yet
        return Intent(kind, text, slots)

    return Intent(Kind.BUILD, text)


def strip_wake_phrase(text: str, wake_phrase: str) -> str:
    """Remove a leading wake phrase, tolerating ASR punctuation around it."""
    pattern = r"\s+".join(re.escape(word) for word in wake_phrase.split())
    return re.sub(rf"^\s*{pattern}\b[\s,.:!-]*", "", text, count=1, flags=re.IGNORECASE)


def heard_wake_phrase(text: str, wake_phrase: str) -> bool:
    pattern = r"\s+".join(re.escape(word) for word in wake_phrase.split())
    return re.match(rf"^\s*{pattern}\b", text or "", flags=re.IGNORECASE) is not None


HELP_TEXT = """\
Say what you want built — "make a command line to-do list that saves to JSON" —
and I will write the files. Follow-ups keep the context: "now add due dates".

Control phrases (these never reach the model):
  run it / stop it            run or interrupt the program
  undo that                   restore the previous snapshot
  list files                  what exists in the workspace
  show me main.py             read a file back
  delete main.py              remove a file (asks first)
  the main file is app.py     set what "run it" means
  new project called radar    start a fresh workspace
  dictate into notes.md       type literally until "end dictation"
  explain the parser          ask about the code without changing it
  repeat that                 say the last answer again
  help / quit
"""
