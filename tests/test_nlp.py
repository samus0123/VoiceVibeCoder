"""The NLP pipeline: what was heard vs. what was meant."""

from __future__ import annotations

import pytest

from voicevibecoder.intent import nlp


@pytest.mark.parametrize(
    ("said", "expected"),
    [
        ("write a deaf that uses numb pie", "write a def that uses numpy"),
        ("read the jay son file", "read the JSON file"),
        ("add a four loop", "add a for loop"),
        ("push it to get hub", "push it to github"),
        ("use a sink and a wait", "use async and await"),
    ],
)
def test_lexicon_repairs_technical_vocabulary(said, expected):
    assert nlp.analyze(said).text == expected


def test_lexicon_leaves_ordinary_words_alone():
    # "four" only becomes "for" next to loop words.
    assert nlp.analyze("split it into four files").text == "split it into four files"
    assert nlp.analyze("take a bass line").text == "take a bass line"


def test_fillers_are_removed_but_never_the_whole_utterance():
    assert nlp.analyze("um, uh, make it faster").text == "make it faster"
    assert nlp.analyze("um").text  # would otherwise vanish entirely


def test_self_repair_keeps_only_the_final_thought():
    utterance = nlp.analyze("make it red, actually no, make it blue")
    assert utterance.text == "make it blue"
    assert utterance.retracted == "make it red"


def test_self_repair_marker_with_nothing_after_it_is_just_a_tic():
    assert nlp.analyze("make it blue i mean").text.startswith("make it blue")


@pytest.mark.parametrize(
    ("said", "number"),
    [
        ("ten thousand", "10000"),
        ("twenty five", "25"),
        ("three hundred and forty two", "342"),
        ("one million", "1000000"),
    ],
)
def test_spoken_numbers_become_digits(said, number):
    assert nlp.analyze(f"generate {said} rows").text == f"generate {number} rows"


def test_small_single_numbers_stay_as_words():
    assert nlp.analyze("split it in three").text == "split it in three"


def test_questions_are_recognised():
    assert nlp.analyze("what does the parser do").is_question
    assert nlp.analyze("how does it read the file").is_question
    assert not nlp.analyze("make a parser").is_question


def test_corrections_are_reported_for_the_console():
    utterance = nlp.analyze("write a deaf")
    assert utterance.changed
    assert ("deaf", "def") in utterance.corrections
    assert "deaf -> def" in utterance.diff()


def test_empty_input_is_handled():
    assert nlp.analyze("").text == ""
    assert nlp.analyze(None).text == ""


def test_words_to_number_rejects_non_numbers():
    assert nlp.words_to_number(["banana"]) is None
    assert nlp.words_to_number([]) is None
