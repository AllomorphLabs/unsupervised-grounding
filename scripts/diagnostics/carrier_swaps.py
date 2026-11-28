"""Diagnostic (entry 106; uses hidden/ only to score): carrier swaps on a whole fit.

For each object (a thing carried by two others), its two "with" facts (carried by one agent or
the other) are exchanged in the word tables (state, arrival, departure) and the summary model,
judged by the method's own fit and decoded. Adapted from place_perms_all.py:

As place_perms.py, but each reordering of a thing's places is applied to the word tables (its
place facts, their arrivals and their departures) as well as the summary model.

For each thing, every reordering of its "at" rows (which place each of its place facts means) is
judged by the method's own fit (the filter on the first fit batch, fixed draws, as split_judge).
The best few per thing, and the best overall, are then decoded to show what they do to accuracy.

    GRID_CFG=<config> uv run python scripts/diagnostics/place_perms.py <facts pair> <checkpoint>
"""

import dataclasses
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
at = {}
for j, u in enumerate(D.units.units):
    if u[0] == "at":
        at.setdefault(u[1], {})[u[2]] = j
base = (fit.gauss, fit.t)
group = automorphisms(world)


base_t = fit.t
U = D.U


def use(model):
    fit.gauss, fit.t = model


def judged(model):
    use(model)
    return fit.run(S["batches"]["fit"][:1], seed=424242)[1]


def scored(model):
    use(model)
    frames, positions = fit.decode(S["batches"]["score"])
    pred = np.stack([state_to_array(decode(world, x, key)) for x in frames])
    r = score_predictions(world, group, pred, lang_truth(data, positions))
    return r["modulo_symmetry"], {k: round(v, 3) for k, v in r["components"].items()}


def permuted(model, k, perm):
    g, t = model
    places = sorted(at[k])
    rows = [at[k][p] for p in places]
    src = [rows[i] for i in perm]
    Wb = g.Wb.clone()
    Wb[rows] = g.Wb[src]
    ps, pc = t.ps.clone(), t.pc.clone()
    ps[:, rows] = t.ps[:, src]
    pc[:, rows] = t.pc[:, src]  # arrivals
    pc[:, [U + r for r in rows]] = t.pc[:, [U + r for r in src]]  # departures
    return Gauss(Wb, g.A, g.pairs), dataclasses.replace(t, ps=ps, pc=pc)


with_ = {}
for j, u in enumerate(D.units.units):
    if u[0] == "with":
        for k in u[1:]:
            with_.setdefault(k, []).append(j)
objects = [k for k, js in with_.items() if len(js) == 2]


def swapped(model, rows):
    g, t = model
    a, b_ = rows
    src = [b_, a]
    Wb = g.Wb.clone()
    Wb[[a, b_]] = g.Wb[src]
    ps, pc = t.ps.clone(), t.pc.clone()
    ps[:, [a, b_]] = t.ps[:, src]
    pc[:, [a, b_]] = t.pc[:, src]
    pc[:, [U + a, U + b_]] = t.pc[:, [U + r for r in src]]
    return Gauss(Wb, g.A, g.pairs), dataclasses.replace(t, ps=ps, pc=pc)


j0 = judged(base)
ex, comp = scored(base)
print(f"CARRY baseline judge {j0:.4f} exact {ex:.3f} {json.dumps(comp)}", flush=True)
for k in objects:
    m = swapped(base, with_[k])
    jv = judged(m)
    ex, comp = scored(m)
    print(f"CARRY swap thing {k} {[D.units.units[j] for j in with_[k]]} judge {jv:.4f} exact {ex:.3f} "
          f"{json.dumps(comp)}", flush=True)
