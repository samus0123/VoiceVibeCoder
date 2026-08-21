"""The session loop — where a sentence becomes a program.

Everything else in the package is a component; this is the state machine that
orders them. Its contract with the outside world is one method,
``handle(transcript) -> bool``, which makes the whole application testable
without a microphone, an API key, or a terminal: feed it strings, assert on
the workspace.

The state it keeps is small and each piece exists for a reason speech creates:

``pending_confirmation``  a destructive verb heard once should not be acted on
                          until it is heard twice — "delete main.py", "yes".
``pending_ideas``         so "build the second one" has a second one to mean.
``dictation_target``      literal-text mode, because sometimes you want the
                          words themselves, not a program about them.
``last_response``         "what did you say?" is a normal thing to ask a voice.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from voicevibecoder.codegen.generator import BuildResult, Generator
from voicevibecoder.codegen.ideas import Idea, resolve_choice
from voicevibecoder.config import Config
from voicevibecoder.console import Console
from voicevibecoder.intent import nlp
from voicevibecoder.intent.grammar import HELP_TEXT, Intent, Kind, heard_wake_phrase, parse
from voicevibecoder.intent.normalize import to_filename
from voicevibecoder.speech.speak import NullSpeaker, Speaker
from voicevibecoder.workspace.project import Workspace, WorkspaceError
from voicevibecoder.workspace.runner import RunnerError, RunResult, run_program
from voicevibecoder.workspace.versioning import GitJournal


@dataclass
class PendingAction:
    """Something dangerous, waiting for a spoken "yes"."""

    question: str
    perform: Callable[[], None]


class Session:
    def __init__(
        self,
        config: Config,
        workspace: Workspace,
        generator: Generator,
        speaker: Speaker | None = None,
        console: Console | None = None,
        journal: GitJournal | None = None,
    ) -> None:
        self.config = config
        self.workspace = workspace
        self.generator = generator
        self.speaker = speaker or NullSpeaker()
        self.console = console or Console()
        self.journal = journal or GitJournal(
            workspace.root, enabled=config.git, trailer=config.ai_trailer
        )

        self.pending_confirmation: PendingAction | None = None
        self.pending_ideas: list[Idea] = []
        self.dictation_target: str | None = None
        self.last_response: str = ""
        self.last_build_summary: str = ""
        self.running = True

    # ------------------------------------------------------------------
    # entry points
    # ------------------------------------------------------------------
    def run(self, transcripts: Iterable[str]) -> None:
        """Consume utterances until one of them says stop."""
        for transcript in transcripts:
            if not self.handle(transcript):
                break

    def handle(self, transcript: str) -> bool:
        """Process one utterance. Returns False when the session should end.

        Nothing said to this session is ever silently dropped: an unrecognised
        utterance becomes a build instruction, an utterance that raises is
        reported and the loop continues, and every one of them is written to
        the transcript journal with the intent it resolved to.
        """
        try:
            return self._handle(transcript)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 — one bad utterance is not fatal
            self._journal_line(transcript, "", "error", _short(exc))
            self._fail("Something went wrong handling that", exc)
            return True

    def _handle(self, transcript: str) -> bool:
        raw = (transcript or "").strip()
        if not raw:
            return True

        if self.config.wake_phrase and not self.dictation_target:
            if not heard_wake_phrase(raw, self.config.wake_phrase):
                self.console.write(f"  ·  (not addressed to me: {raw})", "dim")
                self._journal_line(raw, "", "unaddressed", "")
                return True

        # Dictation is literal: the words are the payload, so the NLP layer
        # and the grammar are bypassed except for the phrase that ends it.
        if self.dictation_target and not _is_dictation_end(raw):
            self._journal_line(raw, raw, "dictation", self.dictation_target)
            return self._dictate(raw)

        utterance = nlp.analyze(raw)
        self.console.heard(raw, utterance.text)
        if utterance.retracted:
            self.console.write(f"  ·  (dropped: {utterance.retracted})", "dim")

        intent = parse(utterance.text, self.config.wake_phrase)
        self._journal_line(raw, utterance.text, intent.kind.value, utterance.retracted)
        return self._dispatch(intent, utterance)

    # ------------------------------------------------------------------
    # dispatch
    # ------------------------------------------------------------------
    def _dispatch(self, intent: Intent, utterance: nlp.Utterance) -> bool:
        if self.pending_confirmation is not None:
            return self._resolve_confirmation(intent, utterance)

        handlers: dict[Kind, Callable[[Intent, nlp.Utterance], bool]] = {
            Kind.SILENCE: lambda *_: True,
            Kind.QUIT: self._on_quit,
            Kind.HELP: self._on_help,
            Kind.BUILD: self._on_build,
            Kind.IDEAS: self._on_ideas,
            Kind.BUILD_IDEA: self._on_build_idea,
            Kind.IMPROVE: self._on_improve,
            Kind.EXPLAIN: self._on_explain,
            Kind.RUN: self._on_run,
            Kind.STOP: self._on_stop,
            Kind.UNDO: self._on_undo,
            Kind.LIST: self._on_list,
            Kind.SHOW: self._on_show,
            Kind.DELETE: self._on_delete,
            Kind.ENTRYPOINT: self._on_entrypoint,
            Kind.NEW_PROJECT: self._on_new_project,
            Kind.DICTATE_START: self._on_dictate_start,
            Kind.DICTATE_END: self._on_dictate_end,
            Kind.REPEAT: self._on_repeat,
            Kind.YES: self._on_stray_yes_no,
            Kind.NO: self._on_stray_yes_no,
        }
        return handlers.get(intent.kind, self._on_build)(intent, utterance)

    def _resolve_confirmation(self, intent: Intent, utterance: nlp.Utterance) -> bool:
        pending, self.pending_confirmation = self.pending_confirmation, None
        assert pending is not None
        if intent.kind is Kind.YES:
            pending.perform()
            return True
        if intent.kind in (Kind.NO, Kind.STOP):
            self._respond("Cancelled.")
            return True
        # Anything else means they moved on; drop the question and obey.
        self.console.write("  ·  (question dropped)", "dim")
        return self._dispatch(intent, utterance)

    # ------------------------------------------------------------------
    # handlers
    # ------------------------------------------------------------------
    def _on_quit(self, *_args) -> bool:
        self._respond("Session ended. Everything is saved.")
        self.running = False
        return False

    def _on_help(self, *_args) -> bool:
        self.console.detail(HELP_TEXT)
        self._respond("Say what you want built, or say run it, undo that, or give me ideas.")
        return True

    def _on_build(self, intent: Intent, utterance: nlp.Utterance) -> bool:
        instruction = intent.text or utterance.text
        # A question that reached the build handler is still a question.
        if utterance.is_question and not _sounds_imperative(instruction):
            return self._answer(instruction)
        return self._build(instruction, spoken=instruction)

    def _on_ideas(self, intent: Intent, _utterance: nlp.Utterance) -> bool:
        topic = intent.slot("topic")
        self.console.write("  ⋯  thinking of something worth building", "dim")
        try:
            ideas = self.generator.ideate(topic, self.workspace)
        except Exception as exc:  # noqa: BLE001 — surfaced, never fatal
            return self._fail("Idea generation failed", exc)

        self.pending_ideas = ideas
        if not ideas:
            self._respond(
                "Nothing cleared the bar. Give me a constraint to work with — "
                "a subject, a file you have, or something you find annoying."
            )
            return True
        for position, idea in enumerate(ideas, start=1):
            self.console.detail(idea.printed(position))
        spoken = " ".join(idea.spoken(i) for i, idea in enumerate(ideas, start=1))
        self._respond(f"{spoken} Say build the first one, or ask for more.")
        return True

    def _on_build_idea(self, intent: Intent, utterance: nlp.Utterance) -> bool:
        if not self.pending_ideas:
            return self._on_build(intent, utterance)  # "build one" with no list
        index = resolve_choice(intent.slot("choice"), len(self.pending_ideas))
        if index is None:
            self._respond("Which one — first, second or third?")
            return True
        idea = self.pending_ideas[index - 1]
        self.pending_ideas = []
        self.console.vibe(f"Building: {idea.name}")
        return self._build(idea.build_instruction, spoken=f"build {idea.name}")

    def _on_improve(self, _intent: Intent, _utterance: nlp.Utterance) -> bool:
        """"How could this be better?" — asked out loud, or after every build."""
        if not self.workspace.files():
            self._respond("There is nothing built yet to improve.")
            return True
        self.console.write("  ⋯  looking for what would make this remarkable", "dim")
        ideas = self._improvements(self.last_build_summary)
        if not ideas:
            self._respond(
                "Nothing worth doing to it. The obvious additions would not "
                "make it better, only bigger."
            )
            return True
        self._offer(ideas, lead="Two things worth doing." if len(ideas) > 1 else "One thing worth doing.")
        return True

    def _on_explain(self, intent: Intent, _utterance: nlp.Utterance) -> bool:
        return self._answer(intent.slot("question") or intent.text)

    def _on_run(self, intent: Intent, _utterance: nlp.Utterance) -> bool:
        target = to_filename(intent.slot("target")) if intent.slot("target") else None
        entrypoint = target or self.workspace.guess_entrypoint()
        if not entrypoint:
            self._respond("There is nothing to run yet.")
            return True
        if not self.workspace.exists(entrypoint):
            self._respond(f"I cannot find {entrypoint}.")
            return True
        result = self._execute(entrypoint)
        if result is not None and not result.ok:
            self._offer_repair(result)
        return True

    def _on_stop(self, *_args) -> bool:
        # Programs run synchronously under a wall-clock limit, so by the time
        # anyone can say "stop" there is nothing left to interrupt.
        self._respond("Nothing is running right now.")
        return True

    def _on_undo(self, *_args) -> bool:
        label = self.workspace.undo()
        if label is None:
            self._respond("There is nothing to undo.")
            return True
        self.journal.commit(f"Undo: {label}", "Restored the previous snapshot.")
        self._respond(f"Undone: {label}. {_count_files(self.workspace)}")
        return True

    def _on_list(self, *_args) -> bool:
        inventory = self.workspace.describe()
        self.console.detail(inventory)
        self._respond(_count_files(self.workspace))
        return True

    def _on_show(self, intent: Intent, utterance: nlp.Utterance) -> bool:
        path = to_filename(intent.slot("target"))
        if not self.workspace.exists(path):
            # The grammar guessed wrong: this was an instruction, not a file.
            return self._on_build(Intent(Kind.BUILD, utterance.text), utterance)
        content = self.workspace.read(path)
        self.console.detail(content)
        lines = content.count("\n") + 1
        self._respond(f"{path} is {lines} lines. It is on screen.")
        return True

    def _on_delete(self, intent: Intent, utterance: nlp.Utterance) -> bool:
        path = to_filename(intent.slot("target"))
        if not self.workspace.exists(path):
            return self._on_build(Intent(Kind.BUILD, utterance.text), utterance)

        def perform() -> None:
            self.workspace.snapshot(f"before deleting {path}")
            self.workspace.delete(path)
            self.journal.commit(f"Delete {path}", "Requested out loud.")
            self._respond(f"Deleted {path}. Say undo that to bring it back.")

        if not self.config.confirm_destructive:
            perform()
            return True
        self.pending_confirmation = PendingAction(f"Delete {path}?", perform)
        self._respond(f"Delete {path}? Say yes or no.")
        return True

    def _on_entrypoint(self, intent: Intent, _utterance: nlp.Utterance) -> bool:
        path = to_filename(intent.slot("target"))
        if not self.workspace.exists(path):
            self._respond(f"There is no {path} yet.")
            return True
        self.workspace.set_entrypoint(path)
        self._respond(f"Run it now means {path}.")
        return True

    def _on_new_project(self, intent: Intent, _utterance: nlp.Utterance) -> bool:
        name = to_filename(intent.slot("name")) or "project"
        root = self.workspace.root.parent / name
        self.workspace = Workspace(root)
        self.workspace.state.project_name = name
        self.generator.reset()
        self.pending_ideas = []
        self.journal = GitJournal(
            self.workspace.root, enabled=self.config.git, trailer=self.config.ai_trailer
        )
        self._respond(f"Started {name}. The workspace is empty and I have forgotten the last project.")
        return True

    def _on_dictate_start(self, intent: Intent, _utterance: nlp.Utterance) -> bool:
        target = to_filename(intent.slot("target")) or "notes.md"
        self.dictation_target = target
        self._respond(f"Dictating into {target}. Say end dictation when you are done.")
        return True

    def _on_dictate_end(self, *_args) -> bool:
        target, self.dictation_target = self.dictation_target, None
        if target is None:
            self._respond("I was not dictating.")
            return True
        self.journal.commit(f"Dictate into {target}", "Literal transcription.")
        self._respond(f"Finished dictating into {target}.")
        return True

    def _on_repeat(self, *_args) -> bool:
        self._respond(self.last_response or "I have not said anything yet.")
        return True

    def _on_stray_yes_no(self, *_args) -> bool:
        self._respond("There is nothing waiting on a yes or no.")
        return True

    # ------------------------------------------------------------------
    # the work
    # ------------------------------------------------------------------
    def _build(self, instruction: str, spoken: str) -> bool:
        self.workspace.snapshot(spoken)
        try:
            result = self.generator.build(instruction, self.workspace)
        except Exception as exc:  # noqa: BLE001 — one bad turn is not fatal
            return self._fail("That build did not go through", exc)

        self._report(result)
        self.last_build_summary = result.summary
        if result.entrypoint:
            self.workspace.set_entrypoint(result.entrypoint)
        self._commit(result, spoken)

        if self.config.auto_run and result.wrote_anything:
            self._run_and_heal(spoken)
        if self.config.suggest_improvements and result.wrote_anything:
            self._suggest_next(result.summary)
        return True

    def _suggest_next(self, summary: str) -> None:
        """After a build, volunteer the one idea worth hearing — if there is one.

        Only the best idea is spoken; the rest are printed. An assistant that
        reads three suggestions aloud after every single change is an
        assistant people turn off.
        """
        ideas = self._improvements(summary)
        if not ideas:
            return
        self._offer(ideas[:2], lead="One idea:")

    def _improvements(self, summary: str) -> list[Idea]:
        try:
            return self.generator.suggest_improvements(summary, self.workspace)
        except Exception as exc:  # noqa: BLE001 — a suggestion is never worth a crash
            self.console.write(f"  ·  (no suggestions: {_short(exc)})", "dim")
            return []

    def _offer(self, ideas: list[Idea], lead: str) -> None:
        """Print the ideas, speak the best one, and arm "build the first one"."""
        self.pending_ideas = ideas
        for position, idea in enumerate(ideas, start=1):
            self.console.detail(idea.printed(position))
        best = ideas[0]
        self._respond(f"{lead} {best.name}. {best.pitch} Say build the first one to do it.")

    def _report(self, result: BuildResult) -> None:
        self.console.files(result.changed_files)
        self._respond(result.summary)

    def _run_and_heal(self, spoken: str) -> None:
        """Run what was just written, and let the model fix its own mistakes."""
        entrypoint = self.workspace.guess_entrypoint()
        if not entrypoint:
            return
        result = self._execute(entrypoint)

        for attempt in range(self.config.self_heal_attempts):
            if result is None or result.ok:
                return
            self._respond("Fixing it.")  # the error itself was just spoken
            try:
                repair = self.generator.repair(result.transcript(), self.workspace)
            except Exception as exc:  # noqa: BLE001
                self._fail("The repair attempt failed", exc)
                return
            self._report(repair)
            self._commit(repair, f"fix after failed run ({attempt + 1})")
            result = self._execute(self.workspace.guess_entrypoint() or entrypoint)

        if result is not None and not result.ok:
            self._respond(
                f"It still fails: {result.error_line()} "
                "Tell me what it should do differently, or say undo that."
            )
        _ = spoken

    def _execute(self, entrypoint: str) -> RunResult | None:
        self.console.write(f"  ▶  {entrypoint}", "dim")
        try:
            result = run_program(
                self.workspace.root, entrypoint, timeout_s=self.config.run_timeout_s
            )
        except RunnerError as exc:
            self._respond(str(exc))
            return None
        if result.stdout.strip():
            self.console.detail(result.stdout.rstrip())
        if not result.ok and result.stderr.strip():
            self.console.error(result.error_line())
        self._respond(result.summary())
        return result

    def _offer_repair(self, result: RunResult) -> None:
        def perform() -> None:
            try:
                repair = self.generator.repair(result.transcript(), self.workspace)
            except Exception as exc:  # noqa: BLE001
                self._fail("The repair attempt failed", exc)
                return
            self._report(repair)
            self._commit(repair, "fix after failed run")
            self._execute(self.workspace.guess_entrypoint() or result.entrypoint)

        self.pending_confirmation = PendingAction("Fix it?", perform)
        self._respond("Want me to fix it? Say yes or no.")

    def _answer(self, question: str) -> bool:
        try:
            answer = self.generator.explain(question, self.workspace)
        except Exception as exc:  # noqa: BLE001
            return self._fail("I could not answer that", exc)
        self._respond(answer or "I do not have an answer for that.")
        return True

    def _commit(self, result: BuildResult, spoken: str) -> None:
        """Record the change in git — automatically, or with a typed subject."""
        if not result.wrote_anything or self.config.commit_mode == "off":
            return
        subject = result.summary
        if self.config.commit_mode == "typed":
            subject = self.console.ask(
                "commit message (enter for the spoken summary):", result.summary
            )
        body = f"Spoken instruction: {spoken}"
        commit = self.journal.commit(subject, body)
        if commit:
            self.console.write(f"  ⑂  {commit.sha}  {commit.subject}", "dim")

    def _dictate(self, raw: str) -> bool:
        assert self.dictation_target is not None
        try:
            self.workspace.append(self.dictation_target, raw)
        except WorkspaceError as exc:
            return self._fail("Could not write that down", exc)
        self.console.write(f"  ✎  {self.dictation_target}: {raw}", "file")
        return True

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _journal_line(self, raw: str, understood: str, kind: str, note: str) -> None:
        """Append one line to the session transcript.

        The journal is what makes "process every instruction" checkable rather
        than aspirational: after a long session you can read back exactly what
        was heard, what it was understood as, and what it was treated as.
        """
        entry = {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "heard": raw,
            "understood": understood,
            "intent": kind,
            "note": note,
        }
        try:
            self.workspace.state_dir.mkdir(parents=True, exist_ok=True)
            with (self.workspace.state_dir / "transcript.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(entry) + "\n")
        except OSError:
            pass  # journalling is a convenience, never a reason to stop

    def _respond(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self.last_response = text
        self.console.vibe(text)
        self.speaker.say(text)

    def _fail(self, context: str, exc: Exception) -> bool:
        self.console.error(f"{context}: {exc}")
        self._respond(f"{context}. {_short(exc)}")
        return True


def _is_dictation_end(raw: str) -> bool:
    return parse(raw).kind is Kind.DICTATE_END


def _sounds_imperative(text: str) -> bool:
    """"How about a dark mode" is a request; "how does it parse" is a question."""
    lowered = text.lower().strip()
    return lowered.startswith(("how about", "what if", "can you", "could you", "would you"))


def _count_files(workspace: Workspace) -> str:
    count = len(workspace.files())
    if count == 0:
        return "The workspace is empty."
    return f"{count} file{'s' if count != 1 else ''} in the workspace."


def _short(exc: Exception) -> str:
    text = str(exc).strip().splitlines()
    return text[0][:200] if text else exc.__class__.__name__


def open_workspace(config: Config) -> Workspace:
    return Workspace(Path(config.workspace))
