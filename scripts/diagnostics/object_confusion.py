# Diagnostic (uses hidden/): where is each object decoded, against where it truly is?
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from grounding.analysis.labels import load_key
from grounding.analysis.score import _world_spec, lang_truth
from grounding.render.grid.frames import GridKey, decode
from grounding.sim.world import state_from_array
from grounding.sim.worlds import make_world

pair, f, obj = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
data = pair / "data"
key = GridKey.from_dict(load_key(data)["grid"])
world = make_world(**_world_spec(pair))
z = np.load(pair / f)
pred = [decode(world, x, key) for x in z["frames"]]
truth = [state_from_array(world, t) for t in lang_truth(data, z["positions"])]
o = world.objects.index(obj)


def name(loc):
    k, i = loc
    return (
        ["at", "held", "broken", "gone"][k]
        + ":"
        + (world.places[i] if k in (0, 2) else world.agents[i] if k == 1 else "-")
    )


c = Counter((name(t.loc[o]), name(p.loc[o])) for t, p in zip(truth, pred, strict=True))
rows = sorted({a for a, _ in c})
cols = sorted({b for _, b in c})
print(f"{obj}: rows true, columns decoded (share of all sentences)")
print(" " * 12 + " ".join(f"{b:>10s}" for b in cols))
for a in rows:
    print(f"{a:>12s} " + " ".join(f"{c[(a, b)] / len(truth):10.3f}" for b in cols))
