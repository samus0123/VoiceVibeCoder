"""The quality bar, applied client-side."""

from __future__ import annotations

import json

from voicevibecoder.codegen.ideas import parse_ideas, resolve_choice


def payload(*ideas):
    return json.dumps({"ideas": list(ideas)})


def raw(name, score):
    return {
        "name": name,
        "pitch": "It does a thing.",
        "mechanism": "a trick",
        "build_instruction": f"build {name}",
        "score": score,
    }


def test_ideas_below_the_bar_are_dropped():
    ideas = parse_ideas(payload(raw("Good", 92), raw("Mediocre", 61)), bar=80)
    assert [idea.name for idea in ideas] == ["Good"]


def test_ideas_are_ranked_and_capped_at_three():
    ideas = parse_ideas(
        payload(*(raw(f"Idea{n}", 80 + n) for n in range(6))), bar=80
    )
    assert [idea.score for idea in ideas] == [85, 84, 83]


def test_an_empty_list_is_a_valid_answer():
    assert parse_ideas(payload(), bar=80) == []
    assert parse_ideas('{"ideas": []}', bar=80) == []


def test_malformed_payloads_never_raise():
    assert parse_ideas("not json at all", bar=80) == []
    assert parse_ideas("", bar=80) == []
    assert parse_ideas(None, bar=80) == []


def test_a_raised_bar_can_reject_everything():
    assert parse_ideas(payload(raw("Decent", 88)), bar=95) == []


def test_ordinals_resolve_to_positions():
    assert resolve_choice("second", 3) == 2
    assert resolve_choice("the third one", 3) == 3
    assert resolve_choice("two", 3) == 2
    assert resolve_choice("last", 3) == 3
    assert resolve_choice("fourth", 3) is None


def test_a_single_idea_needs_no_ordinal():
    assert resolve_choice("", 1) == 1
    assert resolve_choice("", 3) is None
