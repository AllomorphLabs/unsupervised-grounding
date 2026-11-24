"""Aligning one pair: every restart, recorded.

    uv run python -m grounding.align.run runs/pilot/pair-000 --kind model
    uv run python -m grounding.align.run runs/pilot/pair-000 --kind raw

``--kind model`` aligns the two trained models' hidden states; ``--kind raw``
is the co-occurrence baseline, the same method on raw statistics. Only the
training episodes are used, and nothing from ``hidden/``. Writes to
``<pair>/align/<kind>/``:

    restart-<r>.pt   each restart's maps and selection score
    summary.json     every restart's score, and which one was selected
    run.json         configs, seeds, git commit, package versions
"""

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from grounding.align.features import Points, model_points, raw_points
from grounding.align.vec2vec import Fitted, Normaliser, Vec2VecConfig, fit, mlp
from grounding.data.load import load_lang, load_obs
from grounding.models.train import load_model
from grounding.runs.record import make_record, read_record, write_record
from grounding.runs.seeds import part_rngs

KINDS = ("model", "raw")
# Parts of the "align" seed. Only ever append (see part_rngs).
ALIGN_PARTS = (*KINDS, "episodes", "decipher")


@dataclass(frozen=True)
class AlignConfig:
    layer: int = -1  # which hidden layer of each model; -1 is the last
    lang_model: str = "lang"  # folder under <pair>/models/ holding the language model
    window: int = 1  # language points pool this many recent positions (1: no pooling)
    episodes: int = 5000  # training episodes sampled per side for alignment
    score_episodes: int = 1000  # held-out episodes sampled per side for scoring
    vec2vec: Vec2VecConfig = field(default_factory=Vec2VecConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> AlignConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        vec = Vec2VecConfig(**raw.pop("vec2vec", {}))
        return cls(**raw, vec2vec=vec)


def restart_seeds(root: int, kind: str, n: int) -> list[int]:
    rng = part_rngs(root, "align", ALIGN_PARTS)[kind]
    return [int(s) for s in rng.integers(2**31, size=n)]


def sample_episodes(pair: Path, cfg: AlignConfig) -> dict[str, dict[str, np.ndarray]]:
    """Episodes to align (from training episodes) and to score (from held-out ones), per side.

    Drawn from the pair's alignment seed, in a fixed order, and shared by every kind of
    alignment, so the model and the co-occurrence baseline see the same episodes.
    """
    pair = Path(pair)
    rng = part_rngs(read_record(pair / "data").seed, "align", ALIGN_PARTS)["episodes"]
    held = {"lang": load_lang(pair / "data").held_out, "obs": load_obs(pair / "data").held_out}
    out: dict[str, dict[str, np.ndarray]] = {"align": {}, "score": {}}
    for use, want_held, n in (("align", False, cfg.episodes), ("score", True, cfg.score_episodes)):
        for side in ("lang", "obs"):
            pool = np.flatnonzero(held[side] == want_held)
            out[use][side] = np.sort(rng.choice(pool, size=min(n, len(pool)), replace=False))
    return out


def points(
    pair: Path, kind: str, cfg: AlignConfig, device: str, episodes: dict[str, np.ndarray]
) -> tuple[Points, Points]:
    """Language and observation points for ``kind``, from the given episodes per side."""
    pair = Path(pair)
    if kind == "raw":
        return (
            raw_points(pair / "data", "lang", episodes["lang"]),
            raw_points(pair / "data", "obs", episodes["obs"]),
        )
    lang = load_model(pair / "models" / cfg.lang_model, device)
    obs = load_model(pair / "models" / "obs", device)
    return (
        model_points(pair / "data", "lang", lang, cfg.layer, episodes["lang"], window=cfg.window),
        model_points(pair / "data", "obs", obs, cfg.layer, episodes["obs"]),
    )


def load_fitted(path: Path, device: str = "cpu") -> Fitted:
    s = torch.load(path, map_location=device, weights_only=True)
    dx, dy = len(s["mean_x"]), len(s["mean_y"])
    hidden = s["f"]["0.weight"].shape[0]
    f, g = mlp(dx, dy, hidden), mlp(dy, dx, hidden)
    f.load_state_dict(s["f"]), g.load_state_dict(s["g"])
    nx, ny = Normaliser(torch.zeros(1, dx)), Normaliser(torch.zeros(1, dy))
    nx.mean, ny.mean = s["mean_x"], s["mean_y"]
    return Fitted(f.to(device).eval(), g.to(device).eval(), nx, ny, s["seed"], s["score"], [])


def align_pair(
    pair: Path, kind: str, cfg: AlignConfig, device: str | None = None, name: str | None = None
) -> dict:
    """Run every restart. Results go to ``<pair>/align/<name>/`` (default: the kind)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    pair = Path(pair)
    out = pair / "align" / (name or kind)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; refusing to overwrite an alignment")
    out.mkdir(parents=True, exist_ok=True)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    root = read_record(pair / "data").seed
    lang, obs = points(pair, kind, cfg, dev, sample_episodes(pair, cfg)["align"])
    x, y = lang.x, obs.x
    seeds = restart_seeds(root, kind, cfg.vec2vec.restarts)

    restarts = []
    for r, seed in enumerate(seeds):
        fitted = fit(x, y, cfg.vec2vec, seed, dev)
        torch.save(fitted.state(), out / f"restart-{r:02d}.pt")
        restarts.append(
            {"restart": r, "seed": seed, "score": fitted.score, "history": fitted.history}
        )
    selected = int(np.argmin([r["score"] for r in restarts]))
    summary = {
        "kind": kind,
        "selected": selected,
        "points": {"lang": len(x), "obs": len(y)},
        "restarts": restarts,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    config = {"kind": kind, **asdict(cfg), "pair_seed": root}
    write_record(out, make_record(config, root))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("pair", type=Path)
    p.add_argument("--kind", choices=KINDS, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/pilot-align.yaml"))
    p.add_argument("--device", default=None)
    p.add_argument("--name", default=None, help="results folder under align/ (default: kind)")
    args = p.parse_args()
    cfg = AlignConfig.from_yaml(args.config)
    summary = align_pair(args.pair, args.kind, cfg, args.device, args.name)
    print(json.dumps({k: v for k, v in summary.items() if k != "restarts"}, indent=2))


if __name__ == "__main__":
    main()
