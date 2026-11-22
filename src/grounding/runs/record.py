"""Run records: everything needed to rerun a run exactly.

Each run directory gets a ``run.json`` holding the run's config, root seed,
git commit, whether the working tree had uncommitted changes, and the versions
of the packages that matter. A run must be reproducible from these alone.
"""

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

RECORD_FILE = "run.json"

# Packages whose versions can change results.
TRACKED_PACKAGES = ("grounding", "numpy", "torch", "wordfreq")


@dataclass(frozen=True)
class RunRecord:
    config: dict[str, Any]
    seed: int
    commit: str | None
    dirty: bool | None
    python: str
    packages: dict[str, str] = field(default_factory=dict)
    created: str = ""


def git_state(repo: Path | None = None) -> tuple[str | None, bool | None]:
    """Return the current commit and whether there are uncommitted changes.

    Returns ``(None, None)`` if git or the repository is unavailable.
    """

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain"))
    except OSError, subprocess.CalledProcessError:
        return None, None
    return commit, dirty


def package_versions(names: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str]:
    """Return installed versions of ``names``, skipping any that aren't installed."""
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def make_record(config: dict[str, Any], seed: int, repo: Path | None = None) -> RunRecord:
    commit, dirty = git_state(repo)
    return RunRecord(
        config=config,
        seed=seed,
        commit=commit,
        dirty=dirty,
        python=platform.python_version(),
        packages=package_versions(),
        created=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def write_record(run_dir: Path, record: RunRecord) -> Path:
    """Write ``record`` to ``run_dir/run.json``. Refuses to overwrite an existing record."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / RECORD_FILE
    with path.open("x", encoding="utf-8") as f:
        json.dump(asdict(record), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def read_record(run_dir: Path) -> RunRecord:
    with (run_dir / RECORD_FILE).open(encoding="utf-8") as f:
        return RunRecord(**json.load(f))
