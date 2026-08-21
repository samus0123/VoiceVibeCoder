"""The command grammar: control phrases vs. build instructions."""

from __future__ import annotations

import pytest

from voicevibecoder.intent.grammar import Kind, parse, strip_wake_phrase


@pytest.mark.parametrize(
    ("said", "kind"),
    [
        ("run it", Kind.RUN),
        ("execute the program", Kind.RUN),
        ("undo that", Kind.UNDO),
        ("scratch that", Kind.UNDO),
        ("list files", Kind.LIST),
        ("help", Kind.HELP),
        ("quit", Kind.QUIT),
        ("yes", Kind.YES),
        ("no", Kind.NO),
        ("repeat that", Kind.REPEAT),
        ("end dictation", Kind.DICTATE_END),
        ("give me ideas", Kind.IDEAS),
        ("what should i build", Kind.IDEAS),
        ("how could this be better", Kind.IMPROVE),
        ("build the second one", Kind.BUILD_IDEA),
    ],
)
def test_control_phrases_are_claimed_by_the_grammar(said, kind):
    assert parse(said).kind is kind


@pytest.mark.parametrize(
    "said",
    [
        "run the simulation ten thousand times",
        "show me how to write a for loop",
        "delete the duplicate entries from the list",
        "make a snake game",
        "list every prime under a thousand",
        "build a parser for the log format",
    ],
)
def test_instructions_are_never_mistaken_for_commands(said):
    assert parse(said).kind is Kind.BUILD


def test_file_slots_only_accept_things_that_look_like_files():
    assert parse("show me main dot pie").slot("target") == "main dot pie"
    assert parse("delete main.py").slot("target") == "main.py"
    assert parse("show me utils").slot("target") == "utils"


def test_entrypoint_declaration():
    intent = parse("the main file is app dot pie")
    assert intent.kind is Kind.ENTRYPOINT
    assert intent.slot("target") == "app dot pie"


def test_new_project_captures_a_name():
    assert parse("new project called radar").slot("name") == "radar"


def test_explain_needs_a_question_body():
    assert parse("explain the parser").kind is Kind.EXPLAIN
    assert parse("explain").kind is Kind.BUILD  # a bare verb is not a question


def test_ideas_capture_a_topic():
    assert parse("give me ideas for something with my camera").slot("topic") == (
        "something with my camera"
    )


def test_silence_is_its_own_intent():
    assert parse("").kind is Kind.SILENCE
    assert parse("   ").kind is Kind.SILENCE


def test_wake_phrase_is_stripped_before_parsing():
    assert strip_wake_phrase("hey vibe, run it", "hey vibe") == "run it"
    assert parse("hey vibe run it", wake_phrase="hey vibe").kind is Kind.RUN
