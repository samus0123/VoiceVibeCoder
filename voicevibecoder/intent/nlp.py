"""Natural language processing for dictated code instructions.

A speech recogniser is trained on prose, and prose is not how people talk to a
compiler. Between the transcript and the grammar sits this pipeline, which
turns what was *heard* into what was *meant*:

    raw transcript
      -> self-repair resolution   "add a timer, no wait, a countdown"
      -> disfluency removal       "um", "you know", "like"
      -> domain lexicon           "numb pie" -> "numpy", "a sink" -> "async"
      -> number normalisation     "ten thousand" -> "10000"
      -> sentence typing          question vs. instruction
    Utterance

Every stage is a pure function over tokens, which makes the whole pipeline
inspectable: :class:`Utterance` carries the corrections it applied, so the
console can show "heard X, understood Y" and a person can tell the difference
between a bad idea and a bad transcript.

The lexicon is small and curated on purpose. A large fuzzy-matching table
would "fix" words the user actually meant; these entries are ones where the
recogniser's output is not a plausible thing to say to a programming assistant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Disfluencies
# ---------------------------------------------------------------------------

FILLERS = frozenset(
    {"um", "uh", "erm", "uhh", "ah", "hmm", "mmm", "er", "eh"}
)

# Multi-word verbal tics. Removed only as whole phrases.
FILLER_PHRASES = (
    "you know",
    "i mean like",
    "sort of",
    "kind of like",
    "if that makes sense",
)

# ---------------------------------------------------------------------------
# Self-repair
# ---------------------------------------------------------------------------

# When a speaker corrects themselves, everything before the marker is dead.
# "make it red, actually no, make it blue" -> "make it blue"
REPAIR_MARKERS = (
    "scratch that",
    "strike that",
    "no wait",
    "wait no",
    "actually no",
    "sorry i meant",
    "sorry, i meant",
    "i meant",
    "i mean",
    "let me rephrase",
    "rather",
)

_REPAIR_RE = re.compile(
    r"[,;]?\s*\b(" + "|".join(re.escape(m) for m in REPAIR_MARKERS) + r")\b[,:]?\s*",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Domain lexicon
# ---------------------------------------------------------------------------

# Unconditional n-gram rewrites: the left side is not something a person says
# to a coding assistant, and the right side is what they said.
LEXICON: dict[str, str] = {
    # language and libraries
    "pie thon": "python",
    "pi thon": "python",
    "numb pie": "numpy",
    "num pie": "numpy",
    "pie test": "pytest",
    "pie torch": "pytorch",
    "pie game": "pygame",
    "matt plot lib": "matplotlib",
    "j query": "jquery",
    "note js": "node.js",
    "no js": "node.js",
    "type script": "typescript",
    "java script": "javascript",
    # syntax
    "deaf": "def",
    "a sink": "async",
    "a sync": "async",
    "a wait": "await",
    "dicked": "dict",
    "for each": "foreach",
    "l if": "elif",
    "else if": "elif",
    "try accept": "try except",
    "accept block": "except block",
    "in it": "__init__",
    "dunder in it": "__init__",
    "self dot": "self.",
    "reg ex": "regex",
    "regular expression": "regex",
    # formats and protocols
    "jay son": "JSON",
    "jason": "JSON",
    "why a mel": "YAML",
    "yam l": "YAML",
    "see s v": "CSV",
    "sequel": "SQL",
    "my sequel": "MySQL",
    "post gres": "postgres",
    "post grass": "postgres",
    "http s": "https",
    "you are l": "URL",
    "u r l": "URL",
    "a p i": "API",
    "rest api": "REST API",
    "web socket": "websocket",
    "command line": "command-line",
    # tools
    "get hub": "github",
    "git hub": "github",
    "read me": "README",
    "dot in v": ".env",
    "make file": "Makefile",
    "doc string": "docstring",
    "code base": "codebase",
    "test case": "test case",
}

# Rewrites that only apply near a related word, because the left side is an
# ordinary English word: {wrong: (right, {neighbours})}.
CONTEXTUAL: dict[str, tuple[str, frozenset[str]]] = {
    "four": ("for", frozenset({"loop", "each", "every"})),
    "cue": ("queue", frozenset({"a", "the", "priority", "message", "job", "into", "from"})),
    "wile": ("while", frozenset({"loop", "true"})),
    "brake": ("break", frozenset({"loop", "out", "the", "statement"})),
    "flout": ("float", frozenset({"a", "to", "as", "value", "number"})),
    "in put": ("input", frozenset({"user", "the", "an", "read"})),
    "out put": ("output", frozenset({"the", "an", "print", "to"})),
    "bass": ("base", frozenset({"class", "case", "sixty-four", "64"})),
    "chron": ("cron", frozenset({"job", "tab", "schedule"})),
    "colonel": ("kernel", frozenset({"the", "linux", "panic"})),
}

_MAX_LEXICON_SPAN = max(len(phrase.split()) for phrase in LEXICON)

# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000}
NUMBER_WORDS = frozenset(UNITS) | frozenset(TENS) | frozenset(SCALES) | {"and"}

QUESTION_OPENERS = frozenset(
    {"what", "why", "how", "when", "where", "who", "which", "does", "do",
     "is", "are", "can", "could", "should", "would", "will", "did"}
)


@dataclass(frozen=True)
class Utterance:
    """One transcript, after processing, with a record of what changed."""

    raw: str
    text: str
    tokens: list[str] = field(default_factory=list)
    is_question: bool = False
    corrections: list[tuple[str, str]] = field(default_factory=list)
    retracted: str = ""

    @property
    def changed(self) -> bool:
        return self.text.strip().lower() != self.raw.strip().lower()

    def diff(self) -> str:
        """Short "heard -> understood" line for the console."""
        if not self.changed:
            return self.text
        edits = ", ".join(f"{was} -> {now}" for was, now in self.corrections[:4])
        return f"{self.text}   [{edits}]" if edits else self.text


def analyze(raw: str) -> Utterance:
    """Run the full pipeline over one transcript."""
    text = (raw or "").strip()
    if not text:
        return Utterance(raw=raw or "", text="")

    kept, retracted = resolve_self_repair(text)
    kept = strip_filler_phrases(kept)

    tokens = _tokenize(kept)
    tokens = [t for t in tokens if _bare(t) not in FILLERS] or tokens
    tokens, lexicon_edits = apply_lexicon(tokens)
    tokens, number_edits = normalize_numbers(tokens)

    text_out = _detokenize(tokens)
    return Utterance(
        raw=raw,
        text=text_out,
        tokens=tokens,
        is_question=is_question(text_out),
        corrections=lexicon_edits + number_edits,
        retracted=retracted,
    )


def resolve_self_repair(text: str) -> tuple[str, str]:
    """Split a self-corrected utterance into (what stands, what was retracted).

    >>> resolve_self_repair("make it red, actually no, make it blue")
    ('make it blue', 'make it red')
    """
    matches = list(_REPAIR_RE.finditer(text))
    for match in reversed(matches):
        tail = text[match.end():].strip(" ,.;:")
        if len(tail.split()) >= 2:  # a marker with nothing after it is a tic
            return tail, text[: match.start()].strip(" ,.;:")
    return text.strip(), ""


def strip_filler_phrases(text: str) -> str:
    for phrase in FILLER_PHRASES:
        text = re.sub(rf"\b{re.escape(phrase)}\b[,]?\s*", "", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def apply_lexicon(tokens: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """Rewrite mis-recognised technical vocabulary, longest span first."""
    out: list[str] = []
    edits: list[tuple[str, str]] = []
    index = 0
    while index < len(tokens):
        span_used = 0
        for span in range(min(_MAX_LEXICON_SPAN, len(tokens) - index), 0, -1):
            phrase = " ".join(_bare(t) for t in tokens[index : index + span])
            if phrase in LEXICON:
                replacement = LEXICON[phrase]
                out.append(replacement)
                edits.append((phrase, replacement))
                span_used = span
                break
        if span_used:
            index += span_used
            continue

        word = _bare(tokens[index])
        if word in CONTEXTUAL:
            replacement, neighbours = CONTEXTUAL[word]
            window = {
                _bare(t)
                for t in tokens[max(0, index - 2) : index + 3]
                if _bare(t) != word
            }
            if window & neighbours:
                out.append(replacement)
                edits.append((word, replacement))
                index += 1
                continue

        out.append(tokens[index])
        index += 1
    return out, edits


def normalize_numbers(tokens: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """Fold spoken number runs into digits: "ten thousand" -> "10000"."""
    out: list[str] = []
    edits: list[tuple[str, str]] = []
    index = 0
    while index < len(tokens):
        if _bare(tokens[index]) not in NUMBER_WORDS or _bare(tokens[index]) == "and":
            out.append(tokens[index])
            index += 1
            continue

        end = index
        while end < len(tokens) and _bare(tokens[end]) in NUMBER_WORDS:
            end += 1
        # A trailing "and" belongs to the sentence, not the number.
        while end > index and _bare(tokens[end - 1]) == "and":
            end -= 1

        run = [_bare(t) for t in tokens[index:end]]
        value = words_to_number(run)
        # Single small words ("three files") read better as words; anything
        # ten or larger, or built from several words, becomes a numeral.
        if value is not None and (len(run) > 1 or value >= 10):
            spoken = " ".join(run)
            out.append(str(value))
            edits.append((spoken, str(value)))
        else:
            out.extend(tokens[index:end])
        index = end
    return out, edits


def words_to_number(words: list[str]) -> int | None:
    """Parse a spoken cardinal: ["twenty", "five", "thousand"] -> 25000."""
    if not words:
        return None
    total = 0
    current = 0
    seen = False
    for word in words:
        if word == "and":
            continue
        if word in UNITS:
            current += UNITS[word]
        elif word in TENS:
            current += TENS[word]
        elif word == "hundred":
            current = (current or 1) * 100
        elif word in SCALES:
            total += (current or 1) * SCALES[word]
            current = 0
        else:
            return None
        seen = True
    return total + current if seen else None


def is_question(text: str) -> bool:
    """Cheap sentence typing — good enough to route explain vs. build."""
    stripped = text.strip()
    if stripped.endswith("?"):
        return True
    first = _bare(stripped.split()[0]) if stripped.split() else ""
    return first in QUESTION_OPENERS


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"\s+", text.strip()) if t]


def _detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    # Removing a filler can leave orphaned punctuation: "make a, , script".
    text = re.sub(r"(?:\s*,)+(\s*,)", r"\1", text)
    text = re.sub(r",\s*(?=[,.;:!?])", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" ,;:")


def _bare(word: str) -> str:
    return re.sub(r"[^\w+#.]", "", word.lower())
