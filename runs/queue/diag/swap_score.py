# Diagnostic (uses hidden/): a finalist's exact accuracy once one object swap is allowed for.
import sys
import numpy as np
from pathlib import Path
from grounding.analysis.labels import load_key
from grounding.analysis.score import _world_spec, lang_truth, score_predictions
from grounding.render.grid.frames import GridKey, decode
from grounding.sim.symmetry import Relabel, automorphisms
from grounding.sim.world import state_to_array
from grounding.sim.worlds import make_world
pair, i = Path(sys.argv[1]), int(sys.argv[2]); a, b = sys.argv[3], sys.argv[4]
data = pair / "data"; key = GridKey.from_dict(load_key(data)["grid"]); world = make_world(**_world_spec(pair))
z = np.load(pair / f"align/decipher-split/finalist-{i}.npz")
states = [decode(world, x, key) for x in z["frames"]]
o = list(range(len(world.objects))); ia, ib = world.objects.index(a), world.objects.index(b); o[ia], o[ib] = o[ib], o[ia]
r = Relabel(tuple(range(len(world.agents))), tuple(o), tuple(range(len(world.places))))
pred = np.stack([state_to_array(r.state(s)) for s in states])
out = score_predictions(world, automorphisms(world), pred, lang_truth(data, z["positions"]))
print(f"finalist {i+1} with {a}-{b} swapped back: exact {out['modulo_symmetry']:.3f}", {k: round(v, 3) for k, v in out["components"].items()})
