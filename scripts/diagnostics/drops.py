# Diagnostic (uses hidden/): at each true drop of an object, what does the decode show then
# and after?
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from grounding.analysis.labels import load_key
from grounding.analysis.score import _world_spec, lang_truth
from grounding.data.load import load_lang
from grounding.render.grid.frames import GridKey, decode
from grounding.sim.world import AT, HELD, state_from_array
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


def kind(p, t, place):
    k, i = p.loc[o]
    if k == AT:
        return "at the right place" if i == place else "at a wrong place"
    if k == HELD:
        return (
            "still held by the dropper"
            if i == truth[t - 1].loc[o][1]
            else "held by the other agent"
        )
    return "other"


c = {d: Counter() for d in (0, 1, 3)}
n = 0
for t in range(1, len(truth)):
    if ep[t] != ep[t - 1]:
        continue
    a, b = truth[t - 1].loc[o], truth[t].loc[o]
    if a[0] == HELD and b[0] == AT:
        n += 1
        for d in c:
            u = t + d
            if u < len(truth) and ep[u] == ep[t] and truth[u].loc[o] == b:
                c[d][kind(pred[u], u, b[1])] += 1
print(f"{obj}: {n} true drops")
for d, cc in c.items():
    s = sum(cc.values())
    print(
        f"  {d} sentences after the drop: "
        + ", ".join(f"{k} {v / s:.2f}" for k, v in cc.most_common())
    )
before = Counter()
for t in range(1, len(truth)):
    if ep[t] != ep[t - 1]:
        continue
    a, b = truth[t - 1].loc[o], truth[t].loc[o]
    if a[0] == HELD and b[0] == AT:
        k, i = pred[t - 1].loc[o]
        before["held before the drop" if k == HELD else "decoded lying before the drop"] += 1
print("  just before the drop: " + ", ".join(f"{k} {v / n:.2f}" for k, v in before.most_common()))
