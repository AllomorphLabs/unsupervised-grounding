"""A grid pair's "facts" copy: the same pair with each frame redrawn from its units alone
(``units.canonical_frames``), so one set of facts has one drawing (technical decision 29;
experiment log, entry 99).

Copies the language stream, the language model and the pair's record; ``hidden/`` is copied only
so the copy can be scored like the original. Checks that the redrawn stream parses into the same
facts, and that every frame's unit view is unchanged, before writing anything.

    uv run python -m grounding.data.facts runs/pilot-grid-lured/pair-001 \
        runs/pilot-grid-facts/pair-001
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from grounding.align.units import Units, UnitView, canonical_frames, find_units
from grounding.data.load import ObsStream, load_obs


def thing_map(units: Units, again: Units) -> list[int]:
    """For each thing of the redrawn stream's parse, the original thing it is.

    Each thing now lights only its fixed cells, so its list of values is shorter, and the parser
    may number the things differently (a carried object sits in its carrier's first cell). Each
    redrawn thing is the original thing whose values contain its values; under that mapping the
    facts, the places and which things are always on screen must be the same."""
    assert again.places == units.places and again.shape == units.shape
    assert len(again.things) == len(units.things)
    perm = []
    for a in again.things:
        hit = [j for j, b in enumerate(units.things) if set(a) <= set(b)]
        assert len(hit) == 1, "a redrawn thing doesn't match one original thing"
        perm.append(hit[0])
    assert sorted(perm) == list(range(len(perm)))
    assert tuple(units.always[j] for j in perm) == again.always
    assert {mapped(u, perm) for u in again.units} == set(units.units), "the redrawn facts differ"
    return perm


def mapped(u: tuple, perm: list[int]) -> tuple:
    """A redrawn stream's unit, renamed into the original parse's numbering of the things."""
    k = u[0]
    if k == "at":
        return ("at", perm[u[1]], *u[2:])
    if k == "looks":
        return ("looks", perm[u[1]], u[2])
    if k == "gone":
        return ("gone", perm[u[1]])
    return ("with", *sorted((perm[u[1]], perm[u[2]])))


def check_views(obs: ObsStream, new: np.ndarray, units: Units, again: Units, perm: list[int]):
    """Every frame's unit view is the same in the redrawn stream (in its own numbering)."""
    view, view2 = UnitView(units, "cpu"), UnitView(again, "cpu")
    cols = [units.units.index(mapped(u, perm)) for u in again.units]
    for s in range(0, len(new), 250_000):
        a = view(torch.tensor(obs.frames[s : s + 250_000], dtype=torch.float32))[:, cols]
        b = view2(torch.tensor(new[s : s + 250_000], dtype=torch.float32))
        assert torch.equal(a, b), f"unit views differ in frames {s}-{s + 250_000}"


def translate(x, perm: list[int]):
    """A search summary's matchings, renamed into the redrawn stream's numbering of the things:
    redrawn thing k is named by the word that named original thing perm[k]."""
    if isinstance(x, dict):
        if "things" in x and "places" in x and isinstance(x["things"], list):
            x = {**x, "things": [x["things"][j] for j in perm]}
        return {k: translate(v, perm) for k, v in x.items()}
    if isinstance(x, list):
        return [translate(v, perm) for v in x]
    return x


def make_facts_pair(src: Path, dst: Path) -> list[int]:
    src, dst = Path(src), Path(dst)
    if (dst / "data" / "obs.npz").exists():
        raise FileExistsError(f"{dst} already has data; refusing to overwrite")
    obs = load_obs(src / "data")
    units = find_units(obs)
    new = canonical_frames(obs.frames, units)
    again = find_units(ObsStream(new, obs.episode_start, obs.held_out, obs.shape))
    perm = thing_map(units, again)
    print(f"redrawn thing k is original thing {perm}")
    check_views(obs, new, units, again, perm)
    n = min(200_000, len(new))
    print(
        f"distinct frames in the first {n:,}: {len(np.unique(obs.frames[:n], axis=0))} drawn, "
        f"{len(np.unique(new[:n], axis=0))} redrawn"
    )
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
    if (src / "models" / "lang-ps").exists():
        shutil.copytree(src / "models" / "lang-ps", dst / "models" / "lang-ps")
    # The search's finalists, so refits can name the true matching by its index. A fresh pair
    # has none.
    summary_path = src / "align" / "decipher-split" / "summary.json"
    if summary_path.exists():
        out = dst / "align" / "decipher-split"
        out.mkdir(parents=True, exist_ok=True)
        summary = translate(json.loads(summary_path.read_text()), perm)
        summary["note"] = f"translated from {src} (things renumbered by {perm}), entry 99"
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"FACTS pair written to {dst}")
    return perm


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("src", type=Path, help="a grid pair")
    p.add_argument("dst", type=Path, help="where its facts copy goes")
    args = p.parse_args(argv)
    make_facts_pair(args.src, args.dst)


if __name__ == "__main__":
    main()
