"""Corpus configs, read from YAML files in ``configs/``."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CorpusConfig:
    """How to build one corpus pair.

    ``episodes`` is per stream. Episode lengths (in steps) are uniform from
    ``min_len`` to ``max_len`` inclusive, the same for both streams
    (technical decision 12). ``held_out`` is the share of each stream's
    episodes kept back for scoring. ``recap`` is the chance of a past-tense
    recap after each step of the language stream. ``observation`` is how the
    observation stream is drawn: ``state`` (state vectors) or ``grid`` (grid video).
    """

    world: str
    episodes: int
    min_len: int = 20
    max_len: int = 60
    held_out: float = 0.1
    recap: float = 0.1
    observation: str = "state"
    grid_actor: bool = False  # grid video only: mark the cell of the agent that acted
    world_args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.episodes < 1:
            raise ValueError("need at least one episode per stream")
        if not 1 <= self.min_len <= self.max_len:
            raise ValueError("need 1 <= min_len <= max_len")
        if not 0 <= self.held_out < 1:
            raise ValueError("held_out must be in [0, 1)")
        if not 0 <= self.recap <= 1:
            raise ValueError("recap must be in [0, 1]")
        if self.observation not in ("state", "grid"):
            raise ValueError(f"unknown observation {self.observation!r}; use state or grid")

    def lengths(self) -> dict[int, float]:
        """The episode-length distribution, as used for visit weights."""
        n = self.max_len - self.min_len + 1
        return {k: 1 / n for k in range(self.min_len, self.max_len + 1)}

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: Path) -> CorpusConfig:
        return cls(**yaml.safe_load(Path(path).read_text(encoding="utf-8")))
