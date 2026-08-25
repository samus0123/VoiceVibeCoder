"""The session state machine, driven end to end with a fake generator."""

from __future__ import annotations

import json

from tests.conftest import idea


def test_a_spoken_instruction_writes_files(session, generator, workspace, speaker):
    generator.files = {"counter": {"main.py": "print('hi')\n"}}
    session.handle("make me a counter")

    assert workspace.files() == ["main.py"]
    assert generator.calls[0][0] == "build"
    assert speaker.said  # the summary was spoken


def test_the_nlp_layer_reaches_the_generator(session, generator):
    session.handle("um, write a deaf that reads a jay son file")
    _, instruction = generator.calls[0]
    assert "def" in instruction
    assert "JSON" in instruction


def test_undo_restores_the_previous_state(session, generator, workspace):
    generator.files = {
        "first": {"main.py": "one\n"},
        "second": {"main.py": "two\n"},
    }
    session.handle("build the first thing")
    session.handle("build the second thing")
    assert workspace.read("main.py") == "two\n"

    session.handle("undo that")
    assert workspace.read("main.py") == "one\n"


def test_undo_with_nothing_to_undo_says_so(session, speaker):
    session.handle("undo that")
    assert "nothing to undo" in speaker.said[-1].lower()


def test_delete_asks_first_and_yes_confirms(session, generator, workspace, speaker):
    generator.files = {"thing": {"junk.py": "x\n"}}
    session.handle("make a thing")
    session.handle("delete junk dot pie")
    assert workspace.exists("junk.py")  # not yet — it asked
    assert "yes or no" in speaker.said[-1]

    session.handle("yes")
    assert not workspace.exists("junk.py")


def test_delete_cancelled_by_no(session, generator, workspace):
    generator.files = {"thing": {"junk.py": "x\n"}}
    session.handle("make a thing")
    session.handle("delete junk dot pie")
    session.handle("no")
    assert workspace.exists("junk.py")


def test_a_new_instruction_supersedes_a_pending_question(session, generator, workspace):
    generator.files = {"thing": {"junk.py": "x\n"}, "other": {"other.py": "y\n"}}
    session.handle("make a thing")
    session.handle("delete junk dot pie")
    session.handle("make the other one")  # answers neither yes nor no

    assert workspace.exists("junk.py")  # the deletion was dropped
    assert workspace.exists("other.py")  # and the new instruction still ran


def test_show_falls_back_to_building_when_the_file_does_not_exist(session, generator):
    session.handle("show me the total at the end")
    assert generator.calls[-1][0] == "build"


def test_running_a_program_reports_its_output(session, generator, workspace, speaker):
    generator.files = {"greeter": {"main.py": "print('hello there')\n"}}
    session.handle("make a greeter")
    session.handle("run it")
    assert "hello there" in " ".join(speaker.said)


def test_a_failing_program_offers_a_repair(session, generator, workspace, speaker):
    generator.files = {"broken": {"main.py": "raise SystemExit(3)\n"}}
    session.handle("make a broken thing")
    session.handle("run it")
    assert "fix it" in speaker.said[-1].lower()

    session.handle("yes")
    assert any(call[0] == "repair" for call in generator.calls)


def test_ideas_are_offered_and_can_be_built(session, generator, workspace):
    generator.ideas = [idea("Tide Clock"), idea("Echo Mapper", 90)]
    generator.files = {"Tide Clock": {"tide.py": "print(1)\n"}}
    session.handle("give me ideas")
    session.handle("build the first one")

    assert workspace.exists("tide.py")


def test_ideas_below_the_bar_never_reach_the_session(session, generator, speaker):
    generator.ideas = []  # the generator filters; the session reports honestly
    session.handle("give me ideas")
    assert "nothing cleared the bar" in speaker.said[-1].lower()


def test_improvements_are_suggested_after_a_build(config, workspace, generator, speaker):
    from voicevibecoder.console import Console
    from voicevibecoder.session import Session

    generator.files = {"parser": {"main.py": "print(1)\n"}}
    generator.improvements = [idea("Streaming Histogram")]
    session = Session(
        config=config.merged(suggest_improvements=True),
        workspace=workspace,
        generator=generator,
        speaker=speaker,
        console=Console(color=False),
    )
    session.handle("make a parser")

    assert any(call[0] == "improve" for call in generator.calls)
    assert "Streaming Histogram" in speaker.said[-1]
    # ...and it is immediately buildable by ordinal.
    session.handle("build the first one")
    assert generator.calls[-1][1] == "build Streaming Histogram"


def test_how_could_this_be_better_asks_for_improvements(session, generator, workspace):
    workspace.write("main.py", "print(1)\n")
    generator.improvements = [idea("Two Pass Diff")]
    session.handle("how could this be better")
    assert any(call[0] == "improve" for call in generator.calls)


def test_dictation_writes_words_verbatim(session, workspace):
    session.handle("dictate into notes dot m d")
    session.handle("the parser reads a line at a time, um, on purpose")
    session.handle("end dictation")

    # Verbatim: fillers and all. Dictation is not an instruction.
    assert "um" in workspace.read("notes.md")
    assert session.dictation_target is None


def test_explain_does_not_touch_the_workspace(session, generator, workspace):
    workspace.write("main.py", "print(1)\n")
    session.handle("explain what the main file does")
    assert generator.calls[-1][0] == "explain"
    assert workspace.files() == ["main.py"]


def test_a_question_that_is_not_a_command_still_gets_answered(session, generator):
    session.handle("what is the entry point doing right now")
    assert generator.calls[-1][0] == "explain"


def test_quit_ends_the_loop(session):
    assert session.handle("make something") is True
    assert session.handle("quit") is False


def test_generator_failures_do_not_kill_the_session(session, generator, speaker):
    generator.raises = RuntimeError("the API is down")
    assert session.handle("make a thing") is True
    assert "the API is down" in " ".join(speaker.said)


def test_every_utterance_is_journalled(session, generator, workspace):
    session.handle("make a thing")
    session.handle("run it")
    session.handle("gibberish that means nothing")

    lines = (workspace.state_dir / "transcript.jsonl").read_text().splitlines()
    entries = [json.loads(line) for line in lines]
    assert [entry["intent"] for entry in entries] == ["build", "run", "build"]
    assert entries[0]["heard"] == "make a thing"


def test_wake_phrase_gates_utterances(config, workspace, generator, speaker):
    from voicevibecoder.console import Console
    from voicevibecoder.session import Session

    session = Session(
        config=config.merged(wake_phrase="hey vibe"),
        workspace=workspace,
        generator=generator,
        speaker=speaker,
        console=Console(color=False),
    )
    session.handle("make a thing")  # not addressed to it
    assert not generator.calls

    session.handle("hey vibe make a thing")
    assert generator.calls[-1][0] == "build"


def test_new_project_switches_workspace_and_forgets_context(session, generator):
    session.handle("new project called radar")
    assert session.workspace.root.name == "radar"
    assert generator.resets == 1


def test_repeat_says_the_last_thing_again(session, speaker):
    session.handle("undo that")
    last = speaker.said[-1]
    session.handle("say that again")
    assert speaker.said[-1] == last


def test_the_session_opens_by_asking_what_to_build(session, speaker):
    session.greet()
    assert speaker.said == ["What do you want your program to do?"]


def test_a_workspace_you_return_to_is_described_first(session, workspace, speaker):
    workspace.write("main.py", "print(1)\n")
    session.greet()
    greeting = speaker.said[-1]
    assert "1 file in the workspace" in greeting
    assert "Run it means main.py" in greeting
    assert greeting.endswith("What do you want your program to do next?")


def test_a_new_project_asks_again(session, speaker):
    session.handle("new project called radar")
    assert speaker.said[-1].endswith("What do you want your program to do?")
