"""The condition registry, and the pre-registration firewall (technical decision 6).

Every condition is listed here. Main conditions can't be analysed until the
pre-registration is published, which ``prereg/LOCK`` records by naming the
commit it was published at. Pilot, calibration and control-tuning work is not
main and can be analysed at any time.
"""

from pathlib import Path

LOCK = Path(__file__).resolve().parents[3] / "prereg" / "LOCK"

# name -> whether it is a main condition. Add every new condition here.
CONDITIONS: dict[str, bool] = {
    "pilot": False,
}


def check_can_analyse(condition: str, lock: Path = LOCK) -> None:
    """Raise unless results of ``condition`` may be looked at now."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; register it in runs/conditions.py")
    if CONDITIONS[condition] and not lock.exists():
        raise PermissionError(
            f"{condition!r} is a main condition: its results can't be analysed until the "
            "pre-registration is published and prereg/LOCK records its commit"
        )
