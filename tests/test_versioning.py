"""Git journalling: unlimited commits, spoken or typed messages."""

from __future__ import annotations

import shutil

import pytest

from voicevibecoder.workspace.versioning import AI_TRAILER, GitJournal, truncate_subject

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def test_a_repo_is_created_and_changes_are_committed(tmp_path):
    journal = GitJournal(tmp_path)
    (tmp_path / "main.py").write_text("print(1)\n")
    commit = journal.commit("Add a counter", "Spoken instruction: make a counter")

    assert commit is not None
    assert commit.subject == "Add a counter"
    assert (tmp_path / ".git").is_dir()


def test_commits_are_unlimited_and_ordered(tmp_path):
    journal = GitJournal(tmp_path)
    for index in range(6):
        (tmp_path / "main.py").write_text(f"print({index})\n")
        journal.commit(f"Change {index}")

    history = journal.history(10)
    assert len(history) == 6
    assert history[0].subject == "Change 5"


def test_nothing_to_commit_is_not_an_error(tmp_path):
    journal = GitJournal(tmp_path)
    (tmp_path / "main.py").write_text("print(1)\n")
    assert journal.commit("First") is not None
    assert journal.commit("Again") is None


def test_the_ai_trailer_is_included_by_default(tmp_path):
    journal = GitJournal(tmp_path)
    assert AI_TRAILER in journal.compose("Add a thing", "Spoken instruction: x")
    assert AI_TRAILER not in GitJournal(tmp_path, trailer=False).compose("Add a thing")


def test_disabled_journal_is_a_no_op(tmp_path):
    journal = GitJournal(tmp_path, enabled=False)
    (tmp_path / "main.py").write_text("print(1)\n")
    assert journal.commit("nope") is None
    assert journal.history() == []
    assert not (tmp_path / ".git").exists()


def test_long_spoken_summaries_are_truncated_to_a_subject():
    subject = truncate_subject(
        "Wrote a program that reads the log file line by line and prints a "
        "histogram of the request durations it finds there"
    )
    assert len(subject) <= 72
    assert subject.startswith("Wrote a program")


def test_a_multiline_summary_becomes_one_line():
    assert truncate_subject("First line\nsecond line") == "First line"
    assert truncate_subject("") == "Update workspace"
