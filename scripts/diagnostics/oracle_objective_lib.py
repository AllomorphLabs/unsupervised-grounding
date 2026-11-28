"""Shared by oracle_objective.py and score_parts.py: data setup and the true-frame model."""

from pathlib import Path

import numpy as np
import torch

from grounding.align.decipher import (
    DecipherConfig,
    GridDecipherer,
    make_batches,
    make_fit,
    sample,
    standardise,
)
from grounding.analysis.labels import load_key
from grounding.analysis.score import _world_spec, lang_truth
from grounding.models.train import load_model
from grounding.render.grid.frames import EpisodeRenderer, GridKey
from grounding.runs.record import read_record
from grounding.sim.world import state_from_array
from grounding.sim.worlds import make_world

R = Path("C:/Sites/allomorphlabs/unsupervised-grounding")


def setup(pair):
    import os

    cfg = DecipherConfig.from_yaml(
        Path(os.environ.get("GRID_CFG", R / "configs/pilot-grid-decipher-split.yaml"))
    )
    D = GridDecipherer(pair, cfg, "cuda")
    data = pair / "data"
    eps = sample(D.lang, read_record(data).seed, cfg)
    lm = load_model(pair / "models" / cfg.lang_model, "cuda")
    batches = {k: make_batches(D.lang, v, cfg, lm, "cuda") for k, v in eps.items()}
    mu, sd = standardise(batches["fit"])
    standardise(batches["select"], mu, sd)
    standardise(batches["score"], mu, sd)
    return {
        "cfg": cfg,
        "D": D,
        "data": data,
        "batches": batches,
        "key": GridKey.from_dict(load_key(data)["grid"]),
        "world": make_world(**_world_spec(pair)),
    }


def oracle_fit(S, m):
    """Word tables and summary model fitted to the true frames behind every fit sentence."""
    cfg, D, data, key, world = S["cfg"], S["D"], S["data"], S["key"], S["world"]
    rng = np.random.default_rng(0)
    N = cfg.particles
    paths = []
    for b in S["batches"]["fit"]:
        Tm, Be = b.real.shape
        traj = torch.zeros(Be * N, Tm, D.d, dtype=torch.uint8, device="cuda")
        mv = torch.zeros(Be * N, Tm, dtype=torch.bool, device="cuda")
        for e in range(Be):
            n = int(b.real[:, e].sum())
            truth = lang_truth(data, b.ends[e][:n])
            pen = EpisodeRenderer(world, key, rng)
            frames = [pen.start(state_from_array(world, truth[0]))]
            frames += [pen.step(state_from_array(world, truth[t])) for t in range(1, n)]
            fr = torch.tensor(np.stack(frames), dtype=torch.uint8, device="cuda")
            moved = torch.zeros(Tm, dtype=torch.bool, device="cuda")
            moved[1:n] = (fr[1:] != fr[:-1]).any(1)
            rows = torch.arange(N, device="cuda") + e * N
            traj[rows, :n] = fr
            mv[rows] = moved
        paths.append((b, traj, mv))
    f = make_fit(D, m)
    for it in range(max(cfg.summary_from, 6)):
        f.update(paths, it)
    return f
