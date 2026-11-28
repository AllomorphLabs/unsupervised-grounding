"""Diagnostic (entry 108; uses hidden/): at each true drop of an object, is the dropper decoded
at the right place, and does the decoded object lie where the decoded dropper is?

    uv run python scripts/diagnostics/dropper_check.py <pair> <decode .npz under the pair> <object>
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

from grounding.analysis.labels import load_key
from grounding.analysis.score import _world_spec, lang_truth
from grounding.data.load import load_lang
from grounding.render.grid.frames import GridKey, decode
from grounding.sim.world import AT, BROKEN, HELD, state_from_array
from grounding.sim.worlds import make_world

pair, f, obj = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
data = pair / "data"
key = GridKey.from_dict(load_key(data)["grid"])
world = make_world(**_world_spec(pair))
z = np.load(pair / f)
pos = z["positions"]
pred = [decode(world, x, key) for x in z["frames"]]
truth = [state_from_array(world, t) for t in lang_truth(data, pos)]
ep = np.searchsorted(load_lang(data).episode_start, pos, side="right") - 1
o = world.objects.index(obj)
c = Counter()
n = 0
for t in range(1, len(truth)):
    if ep[t] != ep[t - 1]:
        continue
    a, b = truth[t - 1].loc[o], truth[t].loc[o]
    if not (a[0] == HELD and b[0] in (AT, BROKEN)):
        continue
    for d in (0, 1):
        u = t + d
        if u >= len(truth) or ep[u] != ep[t] or truth[u].loc[o] != b:
            continue
        n += d == 0
        who = a[1]
        dropper_ok = pred[u].pos[who] == truth[u].pos[who]
        k, i = pred[u].loc[o]
        lies = k in (AT, BROKEN)
        at_decoded_dropper = lies and i == pred[u].pos[who]
        at_true_place = lies and i == b[1]
        c[
            (
                d,
                "dropper decoded right" if dropper_ok else "dropper decoded wrong",
                "object at true place"
                if at_true_place
                else (
                    "object at decoded dropper's place"
                    if at_decoded_dropper
                    else ("object elsewhere" if lies else "object still held")
                ),
            )
        ] += 1
print(f"{obj}: {n} true drops")
for d in (0, 1):
    tot = sum(v for (dd, *_), v in c.items() if dd == d)
    print(f"  {d} sentences after:")
    for (dd, a_, b_), v in sorted(c.items(), key=lambda x: -x[1]):
        if dd == d:
            print(f"    {a_:22s} {b_:36s} {v / tot:.2f}")
