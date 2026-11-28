"""Diagnostic (entry 105; uses hidden/ only to score): permutation moves on a fit's summary model.

For each thing, every reordering of its "at" rows (which place each of its place facts means) is
judged by the method's own fit (the filter on the first fit batch, fixed draws, as split_judge).
The best few per thing, and the best overall, are then decoded to show what they do to accuracy.

    GRID_CFG=<config> uv run python scripts/diagnostics/place_perms.py <facts pair> <checkpoint>
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
at = {}
for j, u in enumerate(D.units.units):
    if u[0] == "at":
        at.setdefault(u[1], {})[u[2]] = j
base = fit.gauss
group = automorphisms(world)


def judged(g):
    fit.gauss = g
    return fit.run(S["batches"]["fit"][:1], seed=424242)[1]


def scored(g):
    fit.gauss = g
    frames, positions = fit.decode(S["batches"]["score"])
    pred = np.stack([state_to_array(decode(world, x, key)) for x in frames])
    r = score_predictions(world, group, pred, lang_truth(data, positions))
    return r["modulo_symmetry"], {k: round(v, 3) for k, v in r["components"].items()}


def permuted(g, k, perm):
    Wb = g.Wb.clone()
    places = sorted(at[k])
    rows = [at[k][p] for p in places]
    Wb[rows] = g.Wb[[rows[i] for i in perm]]
    return Gauss(Wb, g.A, g.pairs)


j0 = judged(base)
print(f"PERM baseline judge {j0:.4f}", flush=True)
best_all = []
for k in sorted(at):
    res = []
    for perm in itertools.permutations(range(len(at[k]))):
        if perm == tuple(range(len(at[k]))):
            continue
        res.append((judged(permuted(base, k, perm)), perm))
    res.sort(reverse=True)
    print(
        f"PERM thing {k}: best {res[0][0]:.4f} {res[0][1]}, next {res[1][0]:.4f} {res[1][1]} "
        f"(baseline {j0:.4f})",
        flush=True,
    )
    best_all.append((res[0][0], k, res[0][1]))
best_all.sort(reverse=True)
ex, comp = scored(base)
print(f"PERM decode baseline exact {ex:.3f} {json.dumps(comp)}", flush=True)
for jv, k, perm in best_all[:3]:
    ex, comp = scored(permuted(base, k, perm))
    print(
        f"PERM decode thing {k} {perm} judge {jv:.4f} exact {ex:.3f} {json.dumps(comp)}", flush=True
    )
