import pytest

from grounding.runs import conditions
from grounding.runs.conditions import check_can_analyse


def test_pilot_can_be_analysed():
    check_can_analyse("pilot")


def test_unknown_conditions_are_refused():
    with pytest.raises(ValueError):
        check_can_analyse("never-registered")


def test_main_conditions_wait_for_the_lock(monkeypatch, tmp_path):
    monkeypatch.setitem(conditions.CONDITIONS, "main-test", True)
    lock = tmp_path / "LOCK"
    with pytest.raises(PermissionError):
        check_can_analyse("main-test", lock)
    lock.write_text("abc123\n")
    check_can_analyse("main-test", lock)


def test_no_lock_has_been_written_yet():
    # Pre-registration comes first: nothing main may be analysed before it is published.
    assert not conditions.LOCK.exists()
