"""Scoring an alignment against the ground truth.

    uv run python -m grounding.analysis.score runs/pilot/pair-000 --kind model

For each held-out sentence end, the selected restart maps the language point
into the observation space, and its nearest held-out frame (by cosine) is
taken as the prediction. The prediction is right if that frame's true state
is the state the sentence describes: the start state for an opening sentence,
otherwise the state after the sentence's step.

Reported, for the selected restart and for every restart:

- ``exact``: share of predictions whose whole state is right.
- ``modulo_symmetry``: share right up to some symmetry of the world.
- ``by_symmetry``: for each symmetry, the share of predictions equal to the
  true state relabelled by it. For the agent swap, comparing this with the
  identity says whether the pair's mapping swapped the agents.
- ``components``: share right for each part of the state (each agent's place
  and hunger, each object's location), which is far from zero even when
  ``exact`` is.
- ``chance``: the same scores with predictions shuffled among sentences.

The condition must pass the pre-registration firewall first.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from grounding.align.run import AlignConfig, load_fitted, points, sample_episodes
from grounding.analysis.labels import load_key, load_lang_labels, load_obs_labels
from grounding.render.lang.grammar import KINDS
from grounding.runs.conditions import check_can_analyse
from grounding.runs.record import read_record
from grounding.sim.symmetry import Relabel, automorphisms
from grounding.sim.world import World, state_from_array, state_to_array
from grounding.sim.worlds import make_world


def lang_truth(data_dir: Path, positions: np.ndarray) -> np.ndarray:
    """The true state each sentence end describes, as state arrays."""
    labels = load_lang_labels(data_dir)
    starts = labels.sentences[:, 4]
    row = np.searchsorted(starts, positions, side="right") - 1
    episode, step, kind = (labels.sentences[row, i] for i in (0, 1, 2))
    index = np.where(kind == KINDS.index("intro"), 0, step + 1)
    return labels.states[labels.state_start[episode] + index]


def obs_truth(data_dir: Path, positions: np.ndarray) -> np.ndarray:
    return load_obs_labels(data_dir).states[positions]


@torch.no_grad()
def nearest(queries: torch.Tensor, targets: torch.Tensor, chunk: int = 4096) -> np.ndarray:
    """Index of each query's nearest target by cosine similarity."""
    t = torch.nn.functional.normalize(targets, dim=-1)
    out = []
    for i in range(0, len(queries), chunk):
        q = torch.nn.functional.normalize(queries[i : i + chunk], dim=-1)
        out.append((q @ t.T).argmax(1).cpu())
    return torch.cat(out).numpy()


def relabel_states(world: World, r: Relabel, states: np.ndarray) -> np.ndarray:
    return np.stack([state_to_array(r.state(state_from_array(world, s))) for s in states])


def components(world: World) -> dict[str, list[int]]:
    """Columns of a state array belonging to each part of the state."""
    n = len(world.agents)
    out = {f"{a} place": [i] for i, a in enumerate(world.agents)}
    out |= {f"{a} hungry": [n + i] for i, a in enumerate(world.agents)}
    out |= {
        f"{o} location": [2 * n + 2 * i, 2 * n + 2 * i + 1] for i, o in enumerate(world.objects)
    }
    return out


def score_predictions(
    world: World, group: list[Relabel], predicted: np.ndarray, truth: np.ndarray
) -> dict:
    exact = np.all(predicted == truth, axis=1)
    images = {r.describe(world): relabel_states(world, r, truth) for r in group}
    by_symmetry = {k: float(np.all(predicted == v, axis=1).mean()) for k, v in images.items()}
    modulo = np.any([np.all(predicted == v, axis=1) for v in images.values()], axis=0)
    parts = {
        name: float(np.all(predicted[:, cols] == truth[:, cols], axis=1).mean())
        for name, cols in components(world).items()
    }
    return {
        "exact": float(exact.mean()),
        "modulo_symmetry": float(modulo.mean()),
        "by_symmetry": by_symmetry,
        "components": parts,
    }


def score_pair(pair: Path, kind: str, cfg: AlignConfig, device: str | None = None) -> dict:
    pair = Path(pair)
    condition = pair.parent.name
    check_can_analyse(condition)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = pair / "data"
    key = load_key(data)
    world = make_world(**_world_spec(pair))
    if list(world.agents) != key["world"]["agents"]:
        raise ValueError("world in the run record doesn't match the pair's key")
    group = automorphisms(world)

    lang, obs = points(pair, kind, cfg.layer, dev, sample_episodes(pair, cfg)["score"])
    if not (lang.held_out.all() and obs.held_out.all()):
        raise ValueError("scoring must use held-out episodes only")
    truth = lang_truth(data, lang.position)
    frame_states = obs_truth(data, obs.position)

    summary = json.loads((pair / "align" / kind / "summary.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)
    results = []
    for r in summary["restarts"]:
        fitted = load_fitted(pair / "align" / kind / f"restart-{r['restart']:02d}.pt", dev)
        mapped = fitted.to_obs(lang.x)
        frames = fitted.norm_y(obs.x.to(dev))
        predicted = frame_states[nearest(mapped, frames)]
        row = {"restart": r["restart"], "score": r["score"]}
        row |= score_predictions(world, group, predicted, truth)
        row["chance"] = score_predictions(
            world, group, predicted[rng.permutation(len(predicted))], truth
        )
        results.append(row)

    out = {
        "condition": condition,
        "kind": kind,
        "selected": summary["selected"],
        "symmetries": [r.describe(world) for r in group],
        "sentences": len(truth),
        "frames": len(frame_states),
        "result": results[summary["selected"]],
        "restarts": results,
    }
    dest = pair / "analysis"
    dest.mkdir(exist_ok=True)
    (dest / f"score-{kind}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _world_spec(pair: Path) -> dict:
    config = read_record(pair / "data").config
    return {"name": config["world"], **config.get("world_args", {})}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("pair", type=Path)
    p.add_argument("--kind", choices=("model", "raw"), required=True)
    p.add_argument("--config", type=Path, default=Path("configs/pilot-align.yaml"))
    p.add_argument("--device", default=None)
    args = p.parse_args()
    out = score_pair(args.pair, args.kind, AlignConfig.from_yaml(args.config), args.device)
    print(json.dumps(out["result"], indent=2))


if __name__ == "__main__":
    main()
