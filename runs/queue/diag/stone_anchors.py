# Diagnostic (uses hidden/): are an object's errors in resting spells that a past drop or a future take would settle?
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
data = pair / "data"; key = GridKey.from_dict(load_key(data)["grid"]); world = make_world(**_world_spec(pair))
z = np.load(pair / f); pos = z["positions"]
pred = [decode(world, x, key) for x in z["frames"]]
truth = [state_from_array(world, t) for t in lang_truth(data, pos)]
ep = np.searchsorted(load_lang(data).episode_start, pos, side="right") - 1
o = world.objects.index(obj)
tot, bad = Counter(), Counter()
n = len(truth)
for t in range(n):
    if truth[t].loc[o][0] != AT:
        continue
    past = any(truth[u].loc[o][0] == HELD for u in range(t - 1, -1, -1) if ep[u] == ep[t]) if t and ep[t - 1] == ep[t] else False
    fut = False
    u = t + 1
    while u < n and ep[u] == ep[t]:
        if truth[u].loc[o][0] == HELD:
            fut = True; break
        u += 1
    cat = ("dropped earlier" if past else "lying since the start") + (", taken later" if fut else ", never taken after")
    tot[cat] += 1
    if pred[t].loc[o] != truth[t].loc[o]:
        bad[cat] += 1
allbad = sum(bad.values())
print(f"{obj} lying at a place: {sum(tot.values())} sentences, {allbad} decoded wrong")
for c in sorted(tot, key=lambda c: -bad[c]):
    print(f"  {c:42s} share of sentences {tot[c] / sum(tot.values()):.2f}, error rate {bad[c] / tot[c]:.2f}, share of errors {bad[c] / allbad:.2f}")
