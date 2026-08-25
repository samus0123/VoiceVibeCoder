"""Genius-only idea generation.

Say "give me ideas for something with my camera" and you get back three
proposals, ranked, spoken aloud, each one buildable in a single session — then
say "build the second one" and it exists.

The interesting part is the filter. Ask any model for app ideas and it will
happily produce a to-do list with tags, a weather dashboard and a recipe
organiser. So the request is structured rather than open: every idea must name
the *mechanism* that makes it work — the specific trick, constraint or
inversion a competent engineer would not have thought of in the first minute —
and score itself against a rubric. Ideas below the bar are dropped locally,
not argued with. If nothing clears the bar, the honest answer is "nothing good
enough yet, tell me more", which is a better outcome than a mediocre idea
delivered confidently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

IDEA_SYSTEM = """\
You generate program ideas for someone who will build the one they pick, out \
loud, in the next twenty minutes.

The bar is the whole point. An idea qualifies only if it has a *mechanism*: a \
specific technical trick, inversion, or constraint that makes it work and that \
is not obvious in the first minute of thinking about the topic. State that \
mechanism plainly.

Automatically disqualified, no matter how well presented: to-do lists, note \
apps, weather dashboards, recipe managers, habit trackers, URL shorteners, \
chat wrappers around a language model, CRUD over a table, and anything whose \
description would be unchanged if you swapped the domain for another one.

Every idea must be:
- buildable as a handful of files that run on this machine, offline, with the \
standard library plus at most one common package;
- demonstrable in one run — there is something to watch happen;
- describable out loud in two sentences, because it will be read aloud.

Score each idea 0-100 on this rubric, and be a harsh grader: mechanism \
non-obviousness (40), demonstrability in one run (25), how well it fits what \
the person actually asked for (25), buildability in twenty minutes (10). \
Scores above 90 should be rare. Return fewer ideas rather than padding, and \
return none at all if nothing clears the bar.
"""

IDEA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ideas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Two or three words, easy to say out loud.",
                    },
                    "pitch": {
                        "type": "string",
                        "description": "Two spoken sentences: what it does and what you watch happen.",
                    },
                    "mechanism": {
                        "type": "string",
                        "description": "The non-obvious trick that makes it work.",
                    },
                    "build_instruction": {
                        "type": "string",
                        "description": (
                            "A complete instruction to build it, as if spoken: "
                            "what files, what behaviour, what the program prints."
                        ),
                    },
                    "score": {"type": "integer", "description": "0-100 against the rubric."},
                },
                "required": ["name", "pitch", "mechanism", "build_instruction", "score"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["ideas"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Idea:
    name: str
    pitch: str
    mechanism: str
    build_instruction: str
    score: int

    def spoken(self, position: int) -> str:
        return f"{position}. {self.name}. {self.pitch}"

    def printed(self, position: int) -> str:
        return (
            f"{position}. {self.name}  [{self.score}]\n"
            f"   {self.pitch}\n"
            f"   mechanism: {self.mechanism}"
        )


def parse_ideas(payload: str, bar: int) -> list[Idea]:
    """Parse the structured response and apply the quality bar locally.

    The model scores; the client decides. Keeping the threshold out of the
    prompt means it can be tightened without re-tuning anything.
    """
    try:
        data = json.loads(payload)
    except (TypeError, ValueError):
        return []
    ideas = [
        Idea(
            name=str(raw.get("name", "")).strip(),
            pitch=str(raw.get("pitch", "")).strip(),
            mechanism=str(raw.get("mechanism", "")).strip(),
            build_instruction=str(raw.get("build_instruction", "")).strip(),
            score=int(raw.get("score", 0) or 0),
        )
        for raw in (data.get("ideas") or [])
        if isinstance(raw, dict)
    ]
    qualified = [idea for idea in ideas if idea.name and idea.score >= bar]
    qualified.sort(key=lambda idea: idea.score, reverse=True)
    return qualified[:3]


def request_text(topic: str, existing: str) -> str:
    subject = topic.strip() or "anything at all — surprise me"
    return (
        f"Topic: {subject}\n\n"
        f"For context, the workspace currently contains:\n{existing}\n\n"
        "Propose at most three ideas that clear the bar. If none do, return an "
        "empty list."
    )


ORDINALS: dict[str, int] = {
    "first": 1, "one": 1, "1": 1, "1st": 1,
    "second": 2, "two": 2, "2": 2, "2nd": 2,
    "third": 3, "three": 3, "3": 3, "3rd": 3,
    "last": -1,
}


def resolve_choice(spoken: str, count: int) -> int | None:
    """Map "the second one" / "number two" / "that last one" to an index."""
    words = [w.strip(".,") for w in (spoken or "").lower().split()]
    for word in words:
        if word in ORDINALS:
            index = ORDINALS[word]
            index = count if index == -1 else index
            return index if 1 <= index <= count else None
    return 1 if count == 1 else None


IMPROVE_SYSTEM = """\
You have just helped someone build a program by voice, and you are now looking \
at what exists and asking one question: what would make this *remarkable* \
rather than merely working?

Propose improvements to the program in front of you — specific to this code, \
not to software in general. Each one must name the mechanism: the concrete \
change, where it goes, and why it makes the program better in a way the person \
will notice the moment they run it again.

Automatically disqualified, however sensible they sound: add tests, add error \
handling, add type hints, add a README, add logging, add a config file, split \
this into modules, add a database, "make it more robust", or any suggestion \
whose wording would be identical for a different program. Those are chores; \
the person can ask for them by name. You are looking for the idea they did not \
think of.

Good improvements usually come from one of these:
- a property of the data or domain the program is ignoring;
- an interaction that would make the output legible at a glance instead of \
after reading it;
- a constraint that, if enforced, would make a whole class of bugs impossible;
- doing something the person assumed was hard, cheaply, because of how this \
particular program is already structured.

Score each 0-100: mechanism non-obviousness (40), noticeable improvement on \
the next run (25), fit to this specific code (25), small enough to add in a \
few minutes (10). Be a harsh grader; most suggestions deserve less than 70. \
Return an empty list if the program is genuinely fine as it is — saying \
nothing is a valid and often correct answer.
"""


def improvement_request_text(summary: str, inventory: str, excerpt: str) -> str:
    return (
        f"The person just asked for: {summary or 'the program in the workspace'}\n\n"
        f"Workspace:\n{inventory}\n\n"
        f"The code as it stands:\n{excerpt}\n\n"
        "What would make this remarkable? At most three ideas, or none."
    )
