"""Spoken paths into real ones."""

from __future__ import annotations

import pytest

from voicevibecoder.intent.normalize import normalize, to_filename


@pytest.mark.parametrize(
    ("said", "path"),
    [
        ("main dot pie", "main.py"),
        ("app dot j s", "app.js"),
        ("config dot jay son", "config.json"),
        ("source slash web app dot j s", "source/web_app.js"),
        ("my notes dot markdown", "my_notes.md"),
        ("index dot h t m l", "index.html"),
        ("test underscore parser dot pie", "test_parser.py"),
    ],
)
def test_spoken_paths(said, path):
    assert to_filename(said) == path


def test_already_written_paths_are_untouched():
    assert to_filename("main.py") == "main.py"
    assert to_filename("src/app.js") == "src/app.js"
    assert to_filename("README") == "README"


def test_empty_input():
    assert to_filename("") == ""


def test_normalize_strips_dictated_punctuation_and_fillers():
    assert normalize("make it faster period") == "make it faster"
    assert normalize("um make it faster") == "make it faster"
