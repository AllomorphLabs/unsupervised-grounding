"""Loading the streams for training.

These loaders read only the public files of a pair. They never open
``hidden/``, which holds the ground truth (see ``grounding.analysis.labels``).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LangStream:
    tokens: np.ndarray  # token numbers, all episodes concatenated
    episode_start: np.ndarray  # offset of each episode in ``tokens``
    held_out: np.ndarray  # per episode
    words: tuple[str, ...]  # token strings by number; for people reading the data

    def episode(self, i: int) -> np.ndarray:
        end = self.episode_start[i + 1] if i + 1 < len(self.episode_start) else len(self.tokens)
        return self.tokens[self.episode_start[i] : end]


@dataclass(frozen=True)
class ObsStream:
    frames: np.ndarray  # one row per frame, all episodes concatenated
    episode_start: np.ndarray
    held_out: np.ndarray

    def episode(self, i: int) -> np.ndarray:
        end = self.episode_start[i + 1] if i + 1 < len(self.episode_start) else len(self.frames)
        return self.frames[self.episode_start[i] : end]


def load_lang(data_dir: Path) -> LangStream:
    data_dir = Path(data_dir)
    with np.load(data_dir / "lang.npz") as z:
        arrays = {k: z[k] for k in ("tokens", "episode_start", "held_out")}
    words = tuple(json.loads((data_dir / "lang_tokens.json").read_text(encoding="utf-8")))
    return LangStream(**arrays, words=words)


def load_obs(data_dir: Path) -> ObsStream:
    with np.load(Path(data_dir) / "obs.npz") as z:
        return ObsStream(**{k: z[k] for k in ("frames", "episode_start", "held_out")})
