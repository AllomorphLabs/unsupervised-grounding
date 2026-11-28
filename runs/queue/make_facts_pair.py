"""Make a grid pair's "facts" copy (entry 99): the same pair with each frame redrawn from its
units alone (units.canonical_frames), so one set of facts has one drawing.

Copies the language stream, the language model and the pair's record; hidden/ is copied only so
the copy can be scored like the original. Checks that the redrawn stream parses into exactly the
same units, and that every frame's unit view is unchanged, before writing anything.

    uv run python runs/queue/make_facts_pair.py runs/pilot-grid-lured/pair-001 \
        runs/pilot-grid-facts/pair-001
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from grounding.align.units import UnitView, canonical_frames, find_units
from grounding.data.load import ObsStream, load_obs

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
if (dst / "data" / "obs.npz").exists():
    sys.exit(f"{dst} already has data; refusing to overwrite")
obs = load_obs(src / "data")
units = find_units(obs)
new = canonical_frames(obs.frames, units)
again = find_units(ObsStream(new, obs.episode_start, obs.held_out, obs.shape))
# Each thing now lights only its fixed cells, so its list of values is shorter, and the parser
# may number the things differently (a carried object sits in its carrier's first cell). Map
# each redrawn thing to the original thing whose values contain its values; under that mapping
# the facts, the places and which things are always on screen must be the same.
assert again.places == units.places and again.shape == units.shape
assert len(again.things) == len(units.things)
perm = []
for a in again.things:
    hit = [j for j, b in enumerate(units.things) if set(a) <= set(b)]
    assert len(hit) == 1, "a redrawn thing doesn't match one original thing"
    perm.append(hit[0])
assert sorted(perm) == list(range(len(perm)))
assert tuple(units.always[j] for j in perm) == again.always


def mapped(u):
    k = u[0]
    if k == "at":
        return ("at", perm[u[1]], *u[2:])
    if k == "looks":
        return ("looks", perm[u[1]], u[2])
    if k == "gone":
        return ("gone", perm[u[1]])
    return ("with", *sorted((perm[u[1]], perm[u[2]])))


assert {mapped(u) for u in again.units} == set(units.units), "the redrawn facts differ"
print(f"redrawn thing k is original thing {perm}")
view, view2 = UnitView(units, "cpu"), UnitView(again, "cpu")
cols = [units.units.index(mapped(u)) for u in again.units]
for s in range(0, len(new), 250_000):
    a = view(torch.tensor(obs.frames[s : s + 250_000], dtype=torch.float32))[:, cols]
    b = view2(torch.tensor(new[s : s + 250_000], dtype=torch.float32))
    assert torch.equal(a, b), f"unit views differ in frames {s}-{s + 250_000}"
per_state = len(np.unique(obs.frames[:200_000], axis=0)), len(np.unique(new[:200_000], axis=0))
print(f"distinct frames in the first 200,000: {per_state[0]} drawn, {per_state[1]} redrawn")

(dst / "data").mkdir(parents=True, exist_ok=True)
for name in ("lang.npz", "lang_tokens.json", "run.json"):
    shutil.copy2(src / "data" / name, dst / "data" / name)
shutil.copytree(src / "data" / "hidden", dst / "data" / "hidden")
with np.load(src / "data" / "obs.npz") as z:
    np.savez_compressed(
        dst / "data" / "obs.npz",
        frames=new,
        episode_start=z["episode_start"],
        held_out=z["held_out"],
        shape=z["shape"],
    )
shutil.copytree(src / "models" / "lang-ps", dst / "models" / "lang-ps")
# The search's finalists, so refits can name the true matching by its index (same units). A
# fresh pair has none.
if not (src / "align" / "decipher-split" / "summary.json").exists():
    sys.exit(print(f"FACTS pair written to {dst} (no search to translate)"))
(dst / "align" / "decipher-split").mkdir(parents=True, exist_ok=True)
# Translated into the redrawn stream's numbering of the things: redrawn thing k is named by the
# word that named original thing perm[k].
summary = json.loads((src / "align" / "decipher-split" / "summary.json").read_text())


def translate(x):
    if isinstance(x, dict):
        if "things" in x and "places" in x and isinstance(x["things"], list):
            x = {**x, "things": [x["things"][j] for j in perm]}
        return {k: translate(v) for k, v in x.items()}
    if isinstance(x, list):
        return [translate(v) for v in x]
    return x


summary = translate(summary)
summary["note"] = f"translated from {src} (things renumbered by {perm}), entry 99"
(dst / "align" / "decipher-split" / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"FACTS pair written to {dst}")
