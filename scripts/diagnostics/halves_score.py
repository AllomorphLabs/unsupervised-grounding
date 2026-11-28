"""Diagnostic (entry 102; uses hidden/): does the method's own score (the filter's fit on the
selection episodes, words and summaries together) prefer the true-frame summary model to the fit's
own? As hunger_halves.py, and also prints each variant's selection score.

Loads a saved fit (a checkpoint of refit_restarts.py) on a facts pair, and a model fitted to the
TRUE frames (redrawn as the facts pair is: oracle_objective_lib, units.canonical_frames). Then
decodes the score episodes four ways: the fit as saved; with the true-frame word tables; with the
true-frame summary model; and the true-frame model itself. Whichever swap restores hunger shows
which half the fit gets wrong.

    GRID_CFG=<config> uv run python scripts/diagnostics/hunger_halves.py <facts pair> <original pair> <checkpoint>
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from oracle_objective_lib import setup  # noqa: E402

from grounding.align.decipher import make_fit  # noqa: E402
from grounding.align.units import canonical_frames, find_units  # noqa: E402
from grounding.analysis.score import lang_truth, score_predictions  # noqa: E402
from grounding.data.load import load_obs  # noqa: E402
from grounding.render.grid.frames import EpisodeRenderer, decode  # noqa: E402
from grounding.sim.symmetry import automorphisms  # noqa: E402
from grounding.sim.world import state_from_array, state_to_array  # noqa: E402

pair, original, ck = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
S = setup(pair)
cfg, D, data, key, world = S["cfg"], S["D"], S["data"], S["key"], S["world"]
raw_units = find_units(load_obs(original / "data"))
saved = torch.load(ck, map_location="cuda", weights_only=False)
m = saved["state"]["m"]
print(f"HALVES {pair.name} from {ck.name} (iteration {saved['iteration']}), matching {m}", flush=True)

# The true-frame model, on frames redrawn from their facts as the facts pair's are.
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
        fr = canonical_frames(np.stack(frames), raw_units)
        fr = torch.tensor(fr, dtype=torch.uint8, device="cuda")
        moved = torch.zeros(Tm, dtype=torch.bool, device="cuda")
        moved[1:n] = (fr[1:] != fr[:-1]).any(1)
        rows = torch.arange(N, device="cuda") + e * N
        traj[rows, :n] = fr
        mv[rows] = moved
    paths.append((b, traj, mv))
oracle = make_fit(D, m)
for it in range(max(cfg.summary_from, 6)):
    oracle.update(paths, it)
del paths

fit = make_fit(D, m)
fit.__dict__.update(saved["state"])
words_true = copy.copy(fit)
words_true.__dict__ = dict(fit.__dict__)
for k in ("t", "sig"):
    if k in oracle.__dict__:
        words_true.__dict__[k] = oracle.__dict__[k]
summary_true = copy.copy(fit)
summary_true.__dict__ = dict(fit.__dict__)
summary_true.__dict__["gauss"] = oracle.gauss
group = automorphisms(world)
for label, f in (
    ("as saved", fit),
    ("true word tables", words_true),
    ("true summary model", summary_true),
    ("true-frame model", oracle),
):
    sel = [f.run(S["batches"]["select"], seed=q)[1] for q in (11, 12)]
    fitll = f.run(S["batches"]["fit"], seed=7)[1]
    print(f"SCORE {label:20s} select {np.mean(sel):.4f} fit {fitll:.4f}", flush=True)
    frames, positions = f.decode(S["batches"]["score"])
    pred = np.stack([state_to_array(decode(world, x, key)) for x in frames])
    r = score_predictions(world, group, pred, lang_truth(data, positions))
    comp = {k: round(v, 3) for k, v in r["components"].items()}
    print(f"HALVES {label:20s} exact {r['modulo_symmetry']:.3f} {json.dumps(comp)}", flush=True)
