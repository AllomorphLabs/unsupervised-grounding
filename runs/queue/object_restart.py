"""Diagnostic (entry 106; uses hidden/ only to score): restart one thing inside a fitted model.

The thing whose place facts have the weakest effects in the summary model (found from the fit
alone) is wiped: its summary rows go to zero (the bias absorbs their average) and its word-table
rows go back to the background word frequencies. The real fit then continues for some rounds,
from a few starting seeds. The unchanged fit gets the same rounds, so the comparison is fair. Each
is judged by the method's own fit (first fit batch, fixed draws) and decoded.

    GRID_CFG=<config> uv run python runs/queue/object_restart.py <facts pair> <checkpoint> [rounds] [starts]
"""

import copy
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
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 6
starts = int(sys.argv[4]) if len(sys.argv) > 4 else 2
S = setup(pair)
D, data, key, world = S["D"], S["data"], S["key"], S["world"]
fitb = S["batches"]["fit"]
saved = torch.load(ck, map_location="cuda", weights_only=False)
fit = make_fit(D, saved["state"]["m"])
fit.__dict__.update(saved["state"])
it0 = saved["iteration"]
units = D.units.units
views, prior = D.unit_prior()
group = automorphisms(world)

# The thing whose place facts have the weakest summary effects (rows centred within the thing).
at = {}
for j, u in enumerate(units):
    if u[0] == "at":
        at.setdefault(u[1], []).append(j)
W = fit.gauss.Wb
size = {k: float((W[js] - W[js].mean(0)).norm(dim=1).mean()) for k, js in at.items()}
weakest = min(size, key=size.get)
mine = [j for j, u in enumerate(units) if u[0] in ("at", "looks", "gone") and u[1] == weakest]
mine += [j for j, u in enumerate(units) if u[0] == "with" and weakest in u[1:]]
print(f"RESTART place-effect sizes {json.dumps({k: round(v, 2) for k, v in size.items()})}; "
      f"restarting thing {weakest} ({len(mine)} facts)", flush=True)


def wiped(st, seed):
    st = copy.deepcopy(st)
    g, t = st["gauss"], st["t"]
    Wb = g.Wb.clone()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for j in mine:
        Wb[-1] = Wb[-1] + (prior @ views[:, j]) * Wb[j]
        Wb[j] = 0.01 * torch.randn(Wb.shape[1], generator=gen).to(Wb.device)
    st["gauss"] = Gauss(Wb, g.A, g.pairs)
    ps, pc = t.ps.clone(), t.pc.clone()
    for j in mine:
        ps[:, j] = t.p0
        pc[:, j] = t.p0
        pc[:, D.U + j] = t.p0
    st["t"] = dataclasses.replace(t, ps=ps, pc=pc)
    return st


def continued(st, seed):
    f = copy.copy(fit)
    f.__dict__ = dict(fit.__dict__)
    f.__dict__.update(copy.deepcopy(st))
    for r in range(rounds):
        paths, _ = f.run(fitb, seed=seed * 100 + r)
        f.update(paths, it0 + r)
        del paths
    return f


def report(label, f):
    judge = f.run(fitb[:1], seed=424242)[1]
    frames, positions = f.decode(S["batches"]["score"])
    pred = np.stack([state_to_array(decode(world, x, key)) for x in frames])
    r = score_predictions(world, group, pred, lang_truth(data, positions))
    comp = {k: round(v, 3) for k, v in r["components"].items()}
    print(f"RESTART {label:24s} judge {judge:.4f} exact {r['modulo_symmetry']:.3f} {json.dumps(comp)}",
          flush=True)
    return judge


st0 = {k: v for k, v in fit.__dict__.items() if k != "D"}
report("as saved", fit)
report(f"unchanged, {rounds} more rounds", continued(st0, 1))
for s in range(starts):
    report(f"restarted, start {s + 1}", continued(wiped(st0, 100 + s), 1))
