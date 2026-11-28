# Diagnostic (uses hidden/): the true matching of a facts pair, in its own numbering of things.
import json, sys
from pathlib import Path
from grounding.align.units import find_units
from grounding.analysis.labels import load_key
from grounding.analysis.score import _world_spec
from grounding.data.load import load_obs
from grounding.render.grid.frames import GridKey, place_cells
from grounding.sim.worlds import make_world
pair = Path(sys.argv[1]); data = pair / "data"
u = find_units(load_obs(data)); key = load_key(data); world = make_world(**_world_spec(pair))
gk = GridKey.from_dict(key["grid"]); words = json.loads((data / "lang_tokens.json").read_text(encoding="utf-8"))
stem = key["lang"]["stems"]; widx = {w: i for i, w in enumerate(words)}; n_ch = u.shape[2]
names = list(world.agents) + list(world.objects)
chans = [set(a) for a in gk.agent] + [{c for c in o if c >= 0} for o in gk.obj]
things = [widx[stem[names[chans.index({v % n_ch for v in t})]]] for t in u.things]
places = []
for p in range(u.places):
    cell = u.place_of_cell.index(p)
    q = next(q for q in range(len(world.places)) if cell in place_cells(world, q, gk))
    places.append(widx[stem[world.places[q]]])
print("TRUE things", tuple(things), "places", tuple(places))
print("names", {n: widx[stem[n]] for n in names + list(world.places)})
