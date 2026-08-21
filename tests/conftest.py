from __future__ import annotations

import pytest

from voicevibecoder.codegen.generator import BuildResult
from voicevibecoder.codegen.ideas import Idea
from voicevibecoder.config import Config
from voicevibecoder.console import Console
from voicevibecoder.session import Session
from voicevibecoder.workspace.project import Workspace


class FakeGenerator:
    """A generator whose behaviour is a script, not a network call.

    ``files`` maps an instruction substring to the files that instruction
    should produce, which is enough to drive the whole session state machine.
    """

    def __init__(self, files: dict[str, dict[str, str]] | None = None) -> None:
        self.files = files or {}
        self.calls: list[tuple[str, str]] = []
        self.ideas: list[Idea] = []
        self.improvements: list[Idea] = []
        self.answer = "It parses the file and prints a total."
        self.raises: Exception | None = None
        self.resets = 0

    def _emit(self, instruction: str, workspace: Workspace) -> BuildResult:
        if self.raises is not None:
            raise self.raises
        written = []
        for key, files in self.files.items():
            if key in instruction:
                for path, content in files.items():
                    written.append(workspace.write(path, content))
        entrypoint = next((p for p in written if p.endswith(".py")), "")
        return BuildResult(
            summary=f"Wrote {', '.join(written)}." if written else "Nothing to do.",
            changed_files=written,
            entrypoint=entrypoint,
        )

    def build(self, instruction: str, workspace: Workspace) -> BuildResult:
        self.calls.append(("build", instruction))
        return self._emit(instruction, workspace)

    def repair(self, transcript: str, workspace: Workspace) -> BuildResult:
        self.calls.append(("repair", transcript))
        return self._emit("repair", workspace)

    def explain(self, question: str, workspace: Workspace) -> str:
        self.calls.append(("explain", question))
        return self.answer

    def ideate(self, topic: str, workspace: Workspace) -> list[Idea]:
        self.calls.append(("ideate", topic))
        return self.ideas

    def suggest_improvements(self, summary: str, workspace: Workspace) -> list[Idea]:
        self.calls.append(("improve", summary))
        return self.improvements

    def reset(self) -> None:
        self.resets += 1


class RecordingSpeaker:
    def __init__(self) -> None:
        self.said: list[str] = []

    def say(self, text: str) -> None:
        self.said.append(text)


def idea(name: str, score: int = 95) -> Idea:
    return Idea(
        name=name,
        pitch=f"{name} does something worth watching.",
        mechanism="an inversion of the usual approach",
        build_instruction=f"build {name}",
        score=score,
    )


@pytest.fixture
def workspace(tmp_path):
    return Workspace(tmp_path / "ws")


@pytest.fixture
def config(tmp_path):
    return Config(
        workspace=tmp_path / "ws",
        auto_run=False,
        suggest_improvements=False,
        git=False,
        tts_backend="off",
        run_timeout_s=10.0,
    )


@pytest.fixture
def generator():
    return FakeGenerator()


@pytest.fixture
def speaker():
    return RecordingSpeaker()


@pytest.fixture
def session(config, workspace, generator, speaker):
    return Session(
        config=config,
        workspace=workspace,
        generator=generator,
        speaker=speaker,
        console=Console(color=False),
    )
