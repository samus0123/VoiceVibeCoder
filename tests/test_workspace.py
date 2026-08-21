"""Containment, snapshots and the runner."""

from __future__ import annotations

import pytest

from voicevibecoder.workspace.project import Workspace, WorkspaceError
from voicevibecoder.workspace.runner import RunnerError, run_program


@pytest.mark.parametrize(
    "path",
    [
        "../escape.py",
        "../../etc/passwd",
        "/etc/passwd",
        "~/secrets.txt",
        "sub/../../out.py",
        ".voicevibe/state.json",
    ],
)
def test_paths_outside_the_workspace_are_refused(workspace, path):
    with pytest.raises(WorkspaceError):
        workspace.resolve(path)


def test_symlinks_cannot_be_used_to_escape(workspace, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace.root / "link").symlink_to(outside)
    with pytest.raises(WorkspaceError):
        workspace.write("link/evil.py", "x")


def test_write_read_delete_roundtrip(workspace):
    workspace.write("src/app.py", "print(1)\n")
    assert workspace.read("src/app.py") == "print(1)\n"
    assert workspace.files() == ["src/app.py"]
    workspace.delete("src/app.py")
    assert workspace.files() == []


def test_reading_a_missing_file_is_an_error(workspace):
    with pytest.raises(WorkspaceError):
        workspace.read("nope.py")


def test_oversized_writes_are_refused(workspace):
    with pytest.raises(WorkspaceError):
        workspace.write("big.py", "x" * 2_000_000)


def test_internal_files_are_not_listed(workspace):
    workspace.write("main.py", "x")
    (workspace.root / "__pycache__").mkdir()
    (workspace.root / "__pycache__" / "main.pyc").write_bytes(b"\x00")
    assert workspace.files() == ["main.py"]


def test_snapshot_and_undo_restores_deleted_files(workspace):
    workspace.write("a.py", "one")
    workspace.snapshot("before")
    workspace.delete("a.py")
    workspace.write("b.py", "two")

    workspace.undo()
    assert workspace.files() == ["a.py"]
    assert workspace.read("a.py") == "one"


def test_undo_is_a_stack(workspace):
    workspace.write("a.py", "v1")
    workspace.snapshot("v1")
    workspace.write("a.py", "v2")
    workspace.snapshot("v2")
    workspace.write("a.py", "v3")

    assert workspace.undo() == "v2"
    assert workspace.read("a.py") == "v2"
    assert workspace.undo() == "v1"
    assert workspace.read("a.py") == "v1"
    assert workspace.undo() is None


def test_entrypoint_is_guessed_then_remembered(workspace):
    workspace.write("tests/helper.py", "x")
    workspace.write("main.py", "x")
    assert workspace.guess_entrypoint() == "main.py"

    workspace.set_entrypoint("tests/helper.py")
    assert workspace.guess_entrypoint() == "tests/helper.py"


def test_entrypoint_survives_reopening(workspace, tmp_path):
    workspace.write("app.py", "x")
    workspace.set_entrypoint("app.py")
    assert Workspace(workspace.root).state.entrypoint == "app.py"


def test_deleting_the_entrypoint_clears_it(workspace):
    workspace.write("app.py", "x")
    workspace.set_entrypoint("app.py")
    workspace.delete("app.py")
    assert workspace.state.entrypoint is None


def test_running_a_program_captures_stdout(workspace):
    workspace.write("main.py", "print('hello')\n")
    result = run_program(workspace.root, "main.py")
    assert result.ok
    assert "hello" in result.stdout


def test_a_failing_program_reports_the_error_line(workspace):
    workspace.write("main.py", "raise ValueError('boom')\n")
    result = run_program(workspace.root, "main.py")
    assert not result.ok
    assert "boom" in result.error_line()
    assert "stderr" in result.transcript()


def test_an_endless_program_is_stopped(workspace):
    workspace.write("main.py", "while True:\n    pass\n")
    result = run_program(workspace.root, "main.py", timeout_s=1.0)
    assert result.timed_out
    assert "still running" in result.summary()


def test_unknown_file_types_are_refused(workspace):
    workspace.write("data.xyz", "not a program")
    with pytest.raises(RunnerError):
        run_program(workspace.root, "data.xyz")
