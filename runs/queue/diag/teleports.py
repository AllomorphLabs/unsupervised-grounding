# Diagnostic: in the decoded sequence, how often does an object jump between places without being held in between?
import sys
from pathlib import Path
import numpy as np
from grounding.analysis.labels import load_key
from grounding.analysis.score import _world_spec, lang_truth
from grounding.data.load import load_lang
from grounding.render.grid.frames import GridKey, decode
from grounding.sim.world import AT, state_from_array
from grounding.sim.worlds import make_world
pair, f, obj = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
data = pair / "data"; key = GridKey.from_dict(load_key(data)["grid"]); world = make_world(**_world_spec(pair))
z = np.load(pair / f); pos = z["positions"]
ep = np.searchsorted(load_lang(data).episode_start, pos, side="right") - 1
o = world.objects.index(obj)
for label, states in (("decoded", [decode(world, x, key) for x in z["frames"]]),
                      ("truth", [state_from_array(world, t) for t in lang_truth(data, pos)])):
    jumps = steps = 0
    for t in range(1, len(states)):
        if ep[t] != ep[t - 1]:
            continue
        a, b = states[t - 1].loc[o], states[t].loc[o]
        if a[0] == AT and b[0] == AT:
            steps += 1
            jumps += a[1] != b[1]
    print(f"{label}: {obj} at a place in consecutive sentences {steps}, jumps to another place {jumps} ({jumps / max(steps, 1):.3f})")
