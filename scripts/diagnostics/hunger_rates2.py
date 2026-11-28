# Diagnostic on tuning pair 1 (uses hidden/): how often is each agent decoded hungry, and how
# often truly hungry?
import sys
from pathlib import Path

import numpy as np

from grounding.analysis.labels import load_key
from grounding.analysis.score import _world_spec, lang_truth
from grounding.render.grid.frames import GridKey, decode
from grounding.sim.world import state_from_array
from grounding.sim.worlds import make_world

pair = Path(sys.argv[1])
z = np.load(pair / sys.argv[2])
key = GridKey.from_dict(load_key(pair / "data")["grid"])
world = make_world(**_world_spec(pair))
pred = [decode(world, x, key) for x in z["frames"]]
truth = lang_truth(pair / "data", z["positions"])

tr = [state_from_array(world, t) for t in truth]
for a in range(2):
    ph = np.array([s.hungry[a] for s in pred])
    th = np.array([s.hungry[a] for s in tr])
    print(
        f"agent {world.agents[a]}: decoded hungry {ph.mean():.3f}, truly hungry {th.mean():.3f}, "
        f"hit when hungry {ph[th].mean():.3f}, false alarm {ph[~th].mean():.3f}"
    )
