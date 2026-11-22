"""Reading the ground truth of a corpus pair, for scoring and probes only.

Training code must never import this module. It is the only code that opens a
pair's ``hidden/`` directory.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from grounding.data.corpus import HIDDEN


@dataclass(frozen=True)
class LangLabels:
    sentences: np.ndarray  # rows: episode, step, kind, of_step, token_start
    states: np.ndarray  # every episode's states, concatenated; state t is after t steps
    state_start: np.ndarray  # offset of each episode in ``states``


@dataclass(frozen=True)
class ObsLabels:
    states: np.ndarray  # true state of each frame, in the unshuffled layout
    actors: np.ndarray  # agent that acted on the step into each frame; -1 for starts


def load_lang_labels(data_dir: Path) -> LangLabels:
    with np.load(Path(data_dir) / HIDDEN / "lang_labels.npz") as z:
        return LangLabels(**{k: z[k] for k in ("sentences", "states", "state_start")})


def load_obs_labels(data_dir: Path) -> ObsLabels:
    with np.load(Path(data_dir) / HIDDEN / "obs_labels.npz") as z:
        return ObsLabels(**{k: z[k] for k in ("states", "actors")})


def load_key(data_dir: Path) -> dict[str, Any]:
    """The vocabulary and the slot shuffle: everything needed to map results back."""
    return json.loads((Path(data_dir) / HIDDEN / "key.json").read_text(encoding="utf-8"))
