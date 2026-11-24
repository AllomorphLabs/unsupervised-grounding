import subprocess

import pytest

from grounding.runs.record import git_state, make_record, read_record, write_record


def test_round_trip(tmp_path):
    record = make_record({"world": "pilot", "steps": 3}, seed=42)
    write_record(tmp_path, record)
    assert read_record(tmp_path) == record


def test_records_core_fields():
    record = make_record({}, seed=1)
    assert record.seed == 1
    assert record.python
    assert "numpy" in record.packages


def test_refuses_to_overwrite(tmp_path):
    write_record(tmp_path, make_record({}, seed=1))
    with pytest.raises(FileExistsError):
        write_record(tmp_path, make_record({}, seed=2))


def test_git_state_tracks_commit_and_changes(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, timeout=30)

    git("init")
    # A throwaway repository: signing is turned off here only, so a machine that signs
    # commits by default doesn't stop the test to ask for a passphrase.
    git(
        "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
        "commit", "--allow-empty", "-m", "x",
    )  # fmt: skip
    commit, dirty = git_state(tmp_path)
    assert commit is not None and len(commit) == 40
    assert dirty is False

    (tmp_path / "new.txt").write_text("change")
    assert git_state(tmp_path) == (commit, True)


def test_git_state_outside_a_repo(tmp_path):
    assert git_state(tmp_path / "missing") == (None, None)
