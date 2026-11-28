"""Diagnostic (entry 105; uses hidden/ to score, and to name the thing tried): split moves judged
after one round of the real fit.

For each place fact of one thing, the summary model's row for it is restarted along the main
directions of the left-over noise (both signs), as the split moves do. Each candidate then gets
one round of the method's own fit (the filter with words and summaries, then the usual update),
and is judged by the method's fit on the first fit batch with fixed draws. The current model gets
the same round with no move, so every candidate is compared fairly. The best candidate is kept if
it beats that, and the next fact starts from it. Finally the result is decoded.

    GRID_CFG=<config> uv run python scripts/diagnostics/em_moves.py <facts pair> <checkpoint> \
        <thing>
"""

import copy
import json
import sys
import time
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
S = setup(pair)
D, data, key, world = S["D"], S["data"], S["key"], S["world"]
fitb = S["batches"]["fit"]
saved = torch.load(ck, map_location="cuda", weights_only=False)
fit = make_fit(D, saved["state"]["m"])
fit.__dict__.update(saved["state"])
facts = [j for j, u in enumerate(D.units.units) if u[0] == "at" and u[1] == thing]
views, prior = D.unit_prior()
group = automorphisms(world)
t0 = time.time()
it = saved["iteration"]


def state_of(f):
    return {k: v for k, v in f.__dict__.items() if k != "D"}


def one_round(st, seed):
    """A copy of the fit from state ``st``, after one round of the real fit."""
    f = copy.copy(fit)
    f.__dict__ = dict(fit.__dict__)
    f.__dict__.update(copy.deepcopy(st))
    paths, _ = f.run(fitb, seed=seed)
    f.update(paths, it)
    del paths
    return f


def judge(f):
    return f.run(fitb[:1], seed=424242)[1]


def scored(f):
    frames, positions = f.decode(S["batches"]["score"])
    pred = np.stack([state_to_array(decode(world, x, key)) for x in frames])
    r = score_predictions(world, group, pred, lang_truth(data, positions))
    return r["modulo_symmetry"], {k: round(v, 3) for k, v in r["components"].items()}


cur = state_of(fit)
ex, comp = scored(fit)
print(f"EM start: judge {judge(fit):.4f} exact {ex:.3f} {json.dumps(comp)}", flush=True)
for n, j in enumerate(facts):
    seed = 9000 + n
    base = one_round(cur, seed)
    best_j, best_f, best_lab = judge(base), base, "no move"
    g = cur["gauss"]
    for d, v in enumerate(_noise_directions(g, 2)):
        for sign in (1.0, -1.0):
            Wb = g.Wb.clone()
            w = sign * 2 * v
            Wb[j] = w
            Wb[-1] = Wb[-1] - (prior @ views[:, j]) * w
            st = dict(cur)
            st["gauss"] = Gauss(Wb, g.A, g.pairs)
            f = one_round(st, seed)
            jv = judge(f)
            if jv > best_j + 1e-4:
                best_j, best_f, best_lab = jv, f, f"direction {d} sign {sign:+.0f}"
    cur = state_of(best_f)
    print(
        f"EM fact {j} {D.units.units[j]}: kept {best_lab}, judge {best_j:.4f} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
final = copy.copy(fit)
final.__dict__ = dict(fit.__dict__)
final.__dict__.update(cur)
ex, comp = scored(final)
print(f"EM end: judge {judge(final):.4f} exact {ex:.3f} {json.dumps(comp)}", flush=True)
