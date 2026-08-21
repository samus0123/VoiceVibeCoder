"""Cleaning up what the recogniser heard.

Speech recognisers are trained on prose, and prose has no ``.py``. Two
transforms bridge the gap:

``normalize``   removes disfluencies and dictated punctuation from any
                utterance, so "um, make it, uh, faster period" becomes
                "make it faster".
``to_filename`` rewrites the *spoken form* of a path into the real thing:
                "main dot pie" -> "main.py", "source slash web app dot j s"
                -> "source/web_app.js".

Deliberately not done here: rewriting the body of an instruction into code.
That is Claude's job, and it is much better at it with the raw wording intact.
"""

from __future__ import annotations

import re

FILLERS = frozenset(
    {"um", "uh", "erm", "ah", "hmm", "like", "basically", "actually", "okay", "ok"}
)

# Punctuation that people dictate out loud at the end of a sentence.
TRAILING_PUNCTUATION = re.compile(
    r"[\s,]*\b(period|full stop|question mark|exclamation (mark|point))\s*$",
    re.IGNORECASE,
)

# Spoken symbols that appear inside file paths.
PATH_SYMBOLS = {
    "dot": ".",
    "point": ".",
    "slash": "/",
    "dash": "-",
    "hyphen": "-",
    "underscore": "_",
}

# Spoken forms of file extensions. Keys are matched after a "dot", longest
# first, so "jay son" wins over "jay".
EXTENSIONS = {
    "pie": "py",
    "py": "py",
    "python": "py",
    "jay son": "json",
    "jason": "json",
    "json": "json",
    "j s": "js",
    "js": "js",
    "javascript": "js",
    "t s": "ts",
    "ts": "ts",
    "typescript": "ts",
    "t s x": "tsx",
    "h t m l": "html",
    "html": "html",
    "c s s": "css",
    "css": "css",
    "m d": "md",
    "md": "md",
    "markdown": "md",
    "text": "txt",
    "txt": "txt",
    "yaml": "yml",
    "yml": "yml",
    "toml": "toml",
    "s h": "sh",
    "sh": "sh",
    "shell": "sh",
    "bash": "sh",
    "rust": "rs",
    "rs": "rs",
    "go": "go",
    "java": "java",
    "c": "c",
    "see": "c",
    "c plus plus": "cpp",
    "cpp": "cpp",
    "s q l": "sql",
    "sql": "sql",
    "c s v": "csv",
    "csv": "csv",
    "env": "env",
    "lock": "lock",
    "cfg": "cfg",
    "ini": "ini",
    "log": "log",
}

# Longest spoken extension, in words: "h t m l" is four.
_MAX_EXTENSION_SPAN = max(len(key.split()) for key in EXTENSIONS)

_WORD_NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


def normalize(text: str) -> str:
    """Strip disfluencies and dictated end punctuation; collapse whitespace."""
    text = TRAILING_PUNCTUATION.sub("", text.strip())
    words = [w for w in re.split(r"\s+", text) if w]
    kept = [w for w in words if _bare(w) not in FILLERS]
    # An utterance of nothing but fillers is still an utterance of nothing;
    # but never delete the whole thing if the user genuinely said "okay".
    cleaned = " ".join(kept) if kept else " ".join(words)
    cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
    return cleaned.strip(" ,.;:")


def to_filename(spoken: str) -> str:
    """Rewrite the spoken form of a path into a real relative path.

    >>> to_filename("main dot pie")
    'main.py'
    >>> to_filename("source slash web app dot j s")
    'source/web_app.js'
    >>> to_filename("README")
    'README'
    """
    spoken = normalize(spoken).strip()
    if not spoken:
        return ""
    # A single token that is already path-shaped is left exactly as it is,
    # case and all: "README", "main.py", "src/app.js".
    if re.fullmatch(r"[\w./\-]+", spoken):
        return spoken

    tokens = re.split(r"\s+", spoken.lower())
    out: list[str] = []      # finished path pieces, symbols included
    pending: list[str] = []  # word run that will become one path segment

    def flush() -> None:
        if pending:
            out.append("_".join(pending))
            pending.clear()

    index = 0
    while index < len(tokens):
        token = _bare(tokens[index])
        if not token:
            index += 1
            continue
        if token in PATH_SYMBOLS:
            flush()
            symbol = PATH_SYMBOLS[token]
            out.append(symbol)
            if symbol == ".":
                extension, consumed = _match_extension(tokens, index + 1)
                if extension:
                    out.append(extension)
                    index += 1 + consumed
                    continue
            index += 1
            continue
        pending.append(_WORD_NUMBERS.get(token, token))
        index += 1

    flush()
    path = "".join(
        piece if piece in (".", "/", "-", "_") else piece for piece in out
    )
    return re.sub(r"_+", "_", path).strip("_")


def _match_extension(tokens: list[str], start: int) -> tuple[str, int]:
    """Longest spoken extension starting at ``tokens[start]``."""
    remaining = [_bare(t) for t in tokens[start:]]
    for span in range(min(_MAX_EXTENSION_SPAN, len(remaining)), 0, -1):
        candidate = " ".join(remaining[:span])
        if candidate in EXTENSIONS:
            return EXTENSIONS[candidate], span
    return "", 0


def _bare(word: str) -> str:
    return re.sub(r"[^\w+]", "", word.lower())
