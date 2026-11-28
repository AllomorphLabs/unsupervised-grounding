"""Diagnostic (entry 105; uses hidden/ to score, and to name the thing): refit one thing's place
rows of a fit's summary model jointly, by EM on the summaries, holding every other row fixed.

EM over the frame stream's unit views (as align.latent.fit_latent), but each M-step solves only
for the chosen thing's "at" rows given the rest, and the noise is held fixed. Started from the
fit's own rows, and from zero. Each result is judged by the method's own fit (first fit batch,
fixed draws) and decoded.

    GRID_CFG=<config> uv run python runs/queue/restricted_em.py <facts pair> <checkpoint> <thing>
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from oracle_objective_lib import setup  # noqa: E402

from grounding.align.decipher import Gauss, make_fit  # noqa: E402
from grounding.align.latent import _loglik, prune  # noqa: E402
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
rows = [j for j, u in enumerate(D.units.units) if u[0] == "at" and u[1] == thing]
z = torch.cat([b.z[b.real] for b in fitb])
views, prior = prune(*D.unit_prior())
logp = torch.log(prior.clamp(min=1e-12))
base = fit.gauss
P = Gauss.features(views, base.pairs)
F = P.shape[1]
others = [k for k in range(F) if k not in rows]
group = automorphisms(world)


def restricted(g, iterations=40, reg=1e-2):
    for i in range(iterations):
        ll, R, RZ = _loglik(z, views, logp, g, 2048)
        XtX = P.T @ (P * R[:, None])
        XtY = P.T @ RZ
        W = g.Wb.clone()
        rhs = XtY[rows] - XtX[rows][:, others] @ W[others]
        W[rows] = torch.linalg.solve(XtX[rows][:, rows] + reg * torch.eye(len(rows), device=W.device), rhs)
        g = Gauss(W, g.A, g.pairs)
        if i % 10 == 0 or i == iterations - 1:
            print(f"REM   iteration {i}: summaries' loglik {ll:.4f}", flush=True)
    return g


def report(label, g):
    fit.gauss = g
    judge = fit.run(fitb[:1], seed=424242)[1]
    frames, positions = fit.decode(S["batches"]["score"])
    pred = np.stack([state_to_array(decode(world, x, key)) for x in frames])
    r = score_predictions(world, group, pred, lang_truth(data, positions))
    comp = {k: round(v, 3) for k, v in r["components"].items()}
    print(f"REM {label:22s} judge {judge:.4f} exact {r['modulo_symmetry']:.3f} {json.dumps(comp)}",
          flush=True)


report("as saved", base)
report("refit from own rows", restricted(base))
W0 = base.Wb.clone()
W0[rows] = 0.0
report("refit from zero", restricted(Gauss(W0, base.A, base.pairs)))
