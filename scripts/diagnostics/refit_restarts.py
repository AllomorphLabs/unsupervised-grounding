"""Diagnostic (entry 82): does restarting each finalist from several seeds change the selection?

Refits chosen finalists of a search from new seeds (the seed sets the fit's starting tables and
its particle-filter draws), and reports for each fit its own last fit value, its selection score
and, using hidden/ only to evaluate, its accuracy. Keeping each matching's best fit by its own fit
value uses no truth, so it could become part of the method.

    uv run python scripts/diagnostics/refit_restarts.py C:/.../runs/pilot-grid/pair-000 decipher-split \
        C:/.../configs/pilot-grid-decipher-split.yaml 0,1 1,2
"""

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from grounding.align.decipher import (
    DecipherConfig,
    GridDecipherer,
    GridMatching,
    make_batches,
    make_fit,
    sample,
    standardise,
)
from grounding.analysis.labels import load_key
from grounding.analysis.score import _world_spec, lang_truth, score_predictions
from grounding.models.train import load_model
from grounding.render.grid.frames import GridKey, decode
from grounding.runs.record import read_record
from grounding.sim.symmetry import automorphisms
from grounding.sim.world import state_to_array
from grounding.sim.worlds import make_world

pair, name, cfg_path = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
which = [int(x) for x in sys.argv[4].split(",")]
seeds = [int(x) for x in sys.argv[5].split(",")]
dev = "cuda"
cfg = DecipherConfig.from_yaml(cfg_path)
D = GridDecipherer(pair, cfg, dev)
data = pair / "data"
eps = sample(D.lang, read_record(data).seed, cfg)
lm = load_model(pair / "models" / cfg.lang_model, dev)
batches = {k: make_batches(D.lang, v, cfg, lm, dev) for k, v in eps.items()}
mu, sd = standardise(batches["fit"])
standardise(batches["select"], mu, sd)
standardise(batches["score"], mu, sd)
del lm
key = GridKey.from_dict(load_key(data)["grid"])
world = make_world(**_world_spec(pair))
group = automorphisms(world)
summary = json.loads((pair / "align" / name / "summary.json").read_text())
rows = []
t0 = time.time()
for i in which:
    mt = summary["finalists"][i]["matching"]
    m = GridMatching(tuple(mt["things"]), tuple(mt["places"]))
    for s in seeds:
        D.cfg = replace(cfg, seed=s)
        f = make_fit(D, m)
        fits = []
        # Checkpoints after every iteration, so a run can be resumed with changed settings:
        # FIT_RESUME=<a saved iteration-NN.pt> continues from it (only with one finalist and seed).
        import os

        ck = pair / "align" / name / "checkpoints" / f"{cfg_path.stem}-{i}-{s}"
        resume = os.environ.get("FIT_RESUME")
        f.fit(
            batches["fit"],
            log=lambda line, fits=fits: fits.append(line) or print(line, flush=True),
            checkpoints=ck,
            resume=Path(resume) if resume else None,
        )
        last = float(fits[-1].split("fit ")[1].split()[0])
        sel = [f.run(batches["select"], seed=q)[1] for q in (11, 12)]
        frames, positions = f.decode(batches["score"])
        pred = np.stack([state_to_array(decode(world, x, key)) for x in frames])
        truth = lang_truth(data, positions)
        r = score_predictions(world, group, pred, truth)
        np.savez_compressed(
            pair / "align" / name / f"restart-{cfg_path.stem}-{i}-{s}.npz", frames=frames, positions=positions
        )
        # Exact frames up to the agent swap and each object near-symmetry (monitoring only).
        from grounding.sim.symmetry import Relabel

        P4 = tuple(range(len(world.places)))
        near = {}
        swaps = (("stone-apple", (0, 2, 1)), ("glass-apple", (2, 1, 0)), ("glass-stone", (1, 0, 2)))
        for nm, o in swaps:
            g2 = [Relabel((0, 1), o, P4), Relabel((1, 0), o, P4)]
            near[nm] = round(score_predictions(world, g2, pred, truth)["modulo_symmetry"], 3)
        row = {
            "finalist": i,
            "seed": s,
            "fit": last,
            "select": float(np.mean(sel)),
            "select_seeds": sel,
            "modulo": r["modulo_symmetry"],
            "components": r["components"],
            "modulo_object_swaps": near,
        }
        rows.append(row)
        c = {k: round(v, 3) for k, v in r["components"].items()}
        print(
            f"RESULT finalist {i} seed {s}: fit {last:.4f}, select {np.mean(sel):.4f} "
            f"({sel[0]:.4f}, {sel[1]:.4f}), modulo {r['modulo_symmetry']:.3f} {json.dumps(c)} "
            f"with object swaps {json.dumps(near)} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        del f
D.cfg = cfg
(pair / "align" / name / "restarts.json").write_text(json.dumps(rows, indent=2))
print(json.dumps(rows))
