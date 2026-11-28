"""Diagnostic (entry 105; uses hidden/ to score, and to name the thing): random search on one
thing's place rows of a fit's summary model, judged by the method's own fit.

The rows are changed only within the top 8 directions of the left-over noise (where the
true-frame model's rows for pair 2's stone lie, entry 105). A (1+1) evolution strategy: each step
adds a random change to the best rows so far, keeps it if the method's fit on the first fit batch
(fixed draws) improves, and widens or narrows its steps by the one-fifth rule. The best is decoded.

    GRID_CFG=<config> uv run python runs/queue/stone_search.py <facts pair> <checkpoint> <thing> [steps]
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from oracle_objective_lib import setup  # noqa: E402

from grounding.align.decipher import Gauss, make_fit  # noqa: E402
from grounding.align.latent import _noise_directions  # noqa: E402
from grounding.analysis.score import lang_truth, score_predictions  # noqa: E402
from grounding.render.grid.frames import decode  # noqa: E402
from grounding.sim.symmetry import automorphisms  # noqa: E402
from grounding.sim.world import state_to_array  # noqa: E402

pair, ck, thing = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 80
S = setup(pair)
D, data, key, world = S["D"], S["data"], S["key"], S["world"]
fitb = S["batches"]["fit"]
saved = torch.load(ck, map_location="cuda", weights_only=False)
fit = make_fit(D, saved["state"]["m"])
fit.__dict__.update(saved["state"])
rows = [j for j, u in enumerate(D.units.units) if u[0] == "at" and u[1] == thing]
base = fit.gauss
N = _noise_directions(base, 8)  # 8 x Dz, scaled by spread
gen = torch.Generator(device="cpu").manual_seed(0)
group = automorphisms(world)


def with_rows(C):
    W = base.Wb.clone()
    W[rows] = base.Wb[rows] + C.to(W.device) @ N
    return Gauss(W, base.A, base.pairs)


def judge(g):
    fit.gauss = g
    return fit.run(fitb[:1], seed=424242)[1]


best_C = torch.zeros(len(rows), N.shape[0])
best = judge(with_rows(best_C))
print(f"SEARCH start judge {best:.4f}", flush=True)
sigma, wins = 0.5, 0
for s in range(steps):
    C = best_C + sigma * torch.randn(best_C.shape, generator=gen)
    jv = judge(with_rows(C))
    if jv > best + 1e-4:
        best, best_C, wins = jv, C, wins + 1
    if (s + 1) % 10 == 0:
        rate = wins / 10
        sigma *= 1.5 if rate > 0.2 else 0.7
        wins = 0
        print(f"SEARCH step {s + 1}: best judge {best:.4f}, step size {sigma:.3f}", flush=True)
fit.gauss = with_rows(best_C)
frames, positions = fit.decode(S["batches"]["score"])
pred = np.stack([state_to_array(decode(world, x, key)) for x in frames])
r = score_predictions(world, group, pred, lang_truth(data, positions))
comp = {k: round(v, 3) for k, v in r["components"].items()}
print(f"SEARCH end judge {best:.4f} exact {r['modulo_symmetry']:.3f} {json.dumps(comp)}", flush=True)
