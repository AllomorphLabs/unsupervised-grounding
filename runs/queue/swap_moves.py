"""Diagnostic (entry 102; uses hidden/ only to score): swap moves on a fit's summary model.

Each candidate exchanges rows of the summary model between units of the same kind, read from the
frame structure alone: a thing's two appearances (which colour means what), two things'
appearances, or two things' places. Each is judged by the method's own fit (the filter on the
first fit batch, fixed draws, as split_judge), and decoded to show what it does to accuracy.

    GRID_CFG=<config> uv run python runs/queue/swap_moves.py <facts pair> <checkpoint>
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from oracle_objective_lib import setup  # noqa: E402

from grounding.align.decipher import Gauss, make_fit  # noqa: E402
from grounding.analysis.score import lang_truth, score_predictions  # noqa: E402
from grounding.render.grid.frames import decode  # noqa: E402
from grounding.sim.symmetry import automorphisms  # noqa: E402
from grounding.sim.world import state_to_array  # noqa: E402

pair, ck = Path(sys.argv[1]), Path(sys.argv[2])
S = setup(pair)
D, data, key, world = S["D"], S["data"], S["key"], S["world"]
saved = torch.load(ck, map_location="cuda", weights_only=False)
fit = make_fit(D, saved["state"]["m"])
fit.__dict__.update(saved["state"])
units = D.units.units
idx = {u: j for j, u in enumerate(units)}
looks = {}
for j, u in enumerate(units):
    if u[0] == "looks":
        looks.setdefault(u[1], []).append(j)
at = {}
for j, u in enumerate(units):
    if u[0] == "at":
        at.setdefault(u[1], {})[u[2]] = j

cands = [("none", [])]
for k, js in looks.items():
    if len(js) == 2:
        cands.append((f"flip looks of thing {k}", [(js[0], js[1])]))
two = [k for k, js in looks.items() if len(js) == 2]
for a, b in itertools.combinations(two, 2):
    A, B = looks[a], looks[b]
    cands.append((f"looks {a}<->{b}", [(A[0], B[0]), (A[1], B[1])]))
    cands.append((f"looks {a}<->{b} crossed", [(A[0], B[1]), (A[1], B[0])]))
for a, b in itertools.combinations(sorted(at), 2):
    if set(at[a]) == set(at[b]):
        cands.append((f"places {a}<->{b}", [(at[a][p], at[b][p]) for p in at[a]]))


def swapped(g, swaps):
    Wb = g.Wb.clone()
    for i, j in swaps:
        Wb[[i, j]] = Wb[[j, i]]
    return Gauss(Wb, g.A, g.pairs)


group = automorphisms(world)
base = fit.gauss
for label, swaps in cands:
    fit.gauss = swapped(base, swaps)
    judge = fit.run(S["batches"]["fit"][:1], seed=424242)[1]
    frames, positions = fit.decode(S["batches"]["score"])
    pred = np.stack([state_to_array(decode(world, x, key)) for x in frames])
    r = score_predictions(world, group, pred, lang_truth(data, positions))
    comp = {k: round(v, 3) for k, v in r["components"].items()}
    print(f"SWAP {label:28s} judge {judge:.4f} exact {r['modulo_symmetry']:.3f} {json.dumps(comp)}",
          flush=True)
