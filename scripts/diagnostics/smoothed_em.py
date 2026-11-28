"""Diagnostic (entry 106; uses hidden/ only to score): exact object inference inside the fit.

Continues a saved fit for some rounds. In each, the filter runs on the fit episodes as usual, then
each episode's decoded path has every object's history re-inferred exactly (smoothlib), and that
path, in place of the particles' sampled paths, goes to the usual update. So the word tables and
the summary model learn from the exactly inferred histories. Afterwards the score episodes are
decoded, as decoded and with exact object inference.

    GRID_CFG=<config> uv run python scripts/diagnostics/smoothed_em.py <facts pair> \
        <checkpoint> [rounds]
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import smoothlib as sl  # noqa: E402
from oracle_objective_lib import setup  # noqa: E402

from grounding.align.decipher import make_fit  # noqa: E402
from grounding.analysis.score import lang_truth, score_predictions  # noqa: E402
from grounding.render.grid.frames import decode  # noqa: E402
from grounding.sim.symmetry import automorphisms  # noqa: E402
from grounding.sim.world import state_to_array  # noqa: E402

pair, ck = Path(sys.argv[1]), Path(sys.argv[2])
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
S = setup(pair)
D, data, key, world, cfg = S["D"], S["data"], S["key"], S["world"], S["cfg"]
fitb = S["batches"]["fit"]
saved = torch.load(ck, map_location="cuda", weights_only=False)
fit = make_fit(D, saved["state"]["m"])
fit.__dict__.update(saved["state"])
it0 = saved["iteration"]
objects = sl.init(fit, D, cfg)
group = automorphisms(world)
N = cfg.particles


def score(per_batch):
    frames, positions = [], []
    for b, F in per_batch:
        keep = b.real.T.cpu().numpy()
        fr = F.to(torch.uint8).cpu().numpy()
        for e in range(F.shape[0]):
            frames.append(fr[e][keep[e]])
            positions.append(b.ends[e])
    frames, positions = np.concatenate(frames), np.concatenate(positions)
    pred = np.stack([state_to_array(decode(world, x, key)) for x in frames])
    r = score_predictions(world, group, pred, lang_truth(data, positions))
    return r["modulo_symmetry"], {k: round(v, 3) for k, v in r["components"].items()}


def report(label):
    judge = fit.run(fitb[:1], seed=424242)[1]
    dec = sl.decoded(S["batches"]["score"])
    ex0, _ = score(dec)
    for k in objects:
        dec = [(b, sl.smooth(b, F, k)) for b, F in dec]
    ex, comp = score(dec)
    print(
        f"SEM {label:18s} judge {judge:.4f} exact as decoded {ex0:.3f}, "
        f"with exact objects {ex:.3f} "
        f"{json.dumps(comp)}",
        flush=True,
    )


report("start")
for r in range(rounds):
    paths, _ = fit.run(fitb, seed=7000 + r)
    dec = sl.decoded(fitb, paths)
    del paths
    for k in objects:
        dec = [(b, sl.smooth(b, F, k)) for b, F in dec]
    new = []
    for b, F in dec:
        traj = F.to(torch.uint8).repeat_interleave(N, 0)
        mv = torch.zeros(traj.shape[:2], dtype=torch.bool, device=traj.device)
        mv[:, 1:] = (traj[:, 1:] != traj[:, :-1]).any(-1)
        new.append((b, traj, mv))
    del dec
    fit.update(new, it0 + r)
    del new
    report(f"after round {r + 1}")
