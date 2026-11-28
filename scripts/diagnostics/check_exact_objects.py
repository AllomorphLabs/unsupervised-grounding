"""Check (uses hidden/ to score): decoding a saved fit with ``exact_objects`` (the method code in
grounding.align.objects) reproduces the diagnostic scripts' result (object_smooth2.py).

    GRID_CFG=<config> uv run python scripts/diagnostics/check_exact_objects.py <facts pair> <checkpoint>
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from oracle_objective_lib import setup  # noqa: E402

from grounding.align.decipher import make_fit  # noqa: E402
from grounding.align.objects import objects  # noqa: E402
from grounding.analysis.score import lang_truth, score_predictions  # noqa: E402
from grounding.render.grid.frames import decode  # noqa: E402
from grounding.sim.symmetry import automorphisms  # noqa: E402
from grounding.sim.world import state_to_array  # noqa: E402

pair, ck = Path(sys.argv[1]), Path(sys.argv[2])
S = setup(pair)
D, data, key, world = S["D"], S["data"], S["key"], S["world"]
saved = torch.load(ck, map_location="cuda", weights_only=False)
fit = make_fit(D, saved["state"]["m"])
fit.__dict__.update(saved["state"])
print(f"CHECK objects found from the frame stream: {objects(D)}", flush=True)
for on in (False, True):
    D.cfg = replace(D.cfg, exact_objects=on)
    frames, positions = fit.decode(S["batches"]["score"])
    pred = np.stack([state_to_array(decode(world, x, key)) for x in frames])
    r = score_predictions(world, automorphisms(world), pred, lang_truth(data, positions))
    comp = {k: round(v, 3) for k, v in r["components"].items()}
    print(f"CHECK exact_objects={on}: exact {r['modulo_symmetry']:.3f} {json.dumps(comp)}", flush=True)
