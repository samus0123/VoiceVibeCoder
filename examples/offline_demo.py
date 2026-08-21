#!/usr/bin/env python3
"""Drive the whole pipeline without a microphone, an API key, or a network.

    python examples/offline_demo.py

A scripted generator stands in for Claude, so what you are watching is the
real thing everywhere else: the NLP layer, the command grammar, the workspace,
snapshots and undo, the runner, and the session state machine — including a
program that fails, is repaired, and runs clean.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from voicevibecoder.codegen.generator import BuildResult
from voicevibecoder.codegen.ideas import Idea
from voicevibecoder.config import Config
from voicevibecoder.console import Console
from voicevibecoder.session import Session
from voicevibecoder.workspace.project import Workspace

BROKEN = """\
import sys

numbers = [int(line) for line in sys.stdin.read().split()]
print("total:", sum(numbrs))          # deliberate typo
"""

FIXED = """\
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
print("total:", sum(numbers))
print("median:", sorted(numbers)[len(numbers) // 2])
"""


class ScriptedGenerator:
    """Claude's stand-in: canned responses, real file writes."""

    def __init__(self) -> None:
        self.built = False

    def build(self, instruction: str, workspace: Workspace) -> BuildResult:
        path = workspace.write("main.py", BROKEN)
        self.built = True
        return BuildResult(
            summary="Wrote main.py, which totals the numbers it is given.",
            changed_files=[path],
            entrypoint=path,
        )

    def repair(self, transcript: str, workspace: Workspace) -> BuildResult:
        path = workspace.write("main.py", FIXED)
        return BuildResult(
            summary="Fixed the misspelled variable and gave it real input.",
            changed_files=[path],
            entrypoint=path,
        )

    def explain(self, question: str, workspace: Workspace) -> str:
        return "It sums a list of numbers and prints the total and the median."

    def ideate(self, topic: str, workspace: Workspace) -> list[Idea]:
        return [
            Idea(
                name="Drift Detector",
                pitch="It watches the numbers as they arrive and says the moment "
                "the distribution stops looking like the one it started with.",
                mechanism="a running two-sample test over a sliding window, so it "
                "needs no training data and no second pass",
                build_instruction="build a drift detector over stdin numbers",
                score=93,
            )
        ]

    def suggest_improvements(self, summary: str, workspace: Workspace) -> list[Idea]:
        if not self.built:
            return []
        return [
            Idea(
                name="Sparkline Output",
                pitch="Print the numbers as a one-line sparkline above the total, "
                "so the shape of the data is visible before the arithmetic.",
                mechanism="map each value onto eight block characters — the whole "
                "chart is one line of output and no dependencies",
                build_instruction="add a sparkline of the input above the total",
                score=88,
            )
        ]

    def reset(self) -> None:
        self.built = False


SCRIPT = [
    "um, make me a, uh, program that totals the numbers i give it",
    "run it",
    "yes",
    "how could this be better",
    "what does the main file do",
    "undo that",
    "list files",
    "quit",
]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "demo"
        config = Config(
            workspace=root,
            tts_backend="off",
            git=False,
            auto_run=False,  # so "run it" below is the thing that finds the bug
            suggest_improvements=False,  # asked for explicitly in the script
        )
        console = Console()
        session = Session(
            config=config,
            workspace=Workspace(root),
            generator=ScriptedGenerator(),
            console=console,
        )
        console.banner()
        for utterance in SCRIPT:
            if not session.handle(utterance):
                break


if __name__ == "__main__":
    main()
