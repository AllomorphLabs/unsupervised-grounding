"""Score each finalist of a running search as soon as its decoded frames are saved (entry 100;
uses hidden/). The search is frozen and nothing here feeds back into it: this only gives an early
look at how well each finalist's fit decodes, before the search has chosen.

    uv run python runs/queue/score_finalists_live.py runs/pilot-grid-facts/pair-002 decipher-split 6
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from grounding.analysis.labels import load_key
from grounding.analysis.score import _world_spec, lang_truth, score_predictions
from grounding.render.grid.frames import GridKey
from grounding.render.grid.frames import decode as read_grid
from grounding.sim.symmetry import automorphisms
from grounding.sim.world import state_to_array
from grounding.sim.worlds import make_world

pair, name, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
data, folder = pair / "data", pair / "align" / name
key = GridKey.from_dict(load_key(data)["grid"])
world = make_world(**_world_spec(pair))
group = automorphisms(world)
done: set[int] = set()
while len(done) < n:
    for i in range(n):
        f, pt = folder / f"finalist-{i}.npz", folder / f"finalist-{i}.pt"
        if i in done or not (f.exists() and pt.exists()):
            continue
        time.sleep(10)  # let the writes finish
        with np.load(f) as z:
            frames, positions = z["frames"], z["positions"]
        matching = torch.load(pt, weights_only=False).get("matching")
        pred = np.stack([state_to_array(read_grid(world, x, key)) for x in frames])
        r = score_predictions(world, group, pred, lang_truth(data, positions))
        comp = {k: round(v, 3) for k, v in r["components"].items()}
        print(
            f"LIVE finalist {i + 1}/{n} {matching}: exact {r['modulo_symmetry']:.3f} "
            f"{json.dumps(comp)}",
            flush=True,
        )
        done.add(i)
    if len(done) < n:
        time.sleep(60)
print("LIVE all finalists scored", flush=True)
