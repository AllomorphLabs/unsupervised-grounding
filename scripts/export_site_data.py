"""Export a small sample of pilot results for the Allomorph website.

usage: uv run python scripts/export_site_data.py OUT_DIR

Reads the pilot pairs' scored decipherment runs and writes two JSON files:
- results.json: every pair's headline results, for the method and its two controls, and
  the restart variance (entry 54 of docs/alignment-attempts.md);
- sample.json: a sample of real held-out language episodes from pair 2 (the untouched
  confirmation pair), each sentence with its words, a word-by-word gloss, the state the
  method decoded it to and the true state.

The sample episodes are the first held-out episodes in scoring order that are long enough
to be worth showing. They are not chosen by how well they decode. Uses the hidden ground
truth, so it's for reporting only, never for fitting.
"""

import json
import sys
from pathlib import Path

import numpy as np

from grounding.analysis.labels import load_key, load_lang_labels
from grounding.analysis.score import _world_spec, lang_truth, relabel_states
from grounding.data.load import load_lang, load_obs
from grounding.render.lang.grammar import KINDS
from grounding.render.state.vector import decode as read_frame
from grounding.render.state.vector import encode
from grounding.sim.symmetry import Relabel, automorphisms
from grounding.sim.world import AT, BROKEN, GONE, HELD, state_from_array, state_to_array
from grounding.sim.worlds import make_world

PILOT = Path("runs/pilot")
PAIRS = {0: "development", 1: "tuning", 2: "confirmation"}
RUNS = {"method": "decipher-v2", "counts": "decipher-counts", "baseline": "decipher-baseline"}
SEED_FILE = "diag-iters12-episodes512-emissionsentence-featureslinear-seed{}-scoreeps300.npz"
SAMPLE_PAIR, SAMPLE_EPISODES, MIN_SENTENCES = 2, 6, 14

VERBS = {
    "be": "is",
    "go": "goes",
    "take": "takes",
    "drop": "drops",
    "eat": "eats",
    "wait": "waits",
    "break": "breaks",
    "hungry": "is hungry",
}
PAST = {
    "be": "was",
    "go": "went",
    "take": "took",
    "drop": "dropped",
    "eat": "ate",
    "wait": "waited",
    "break": "broke",
    "hungry": "was hungry",
}
LOC = {AT: "at", HELD: "held", BROKEN: "broken", GONE: "gone"}


def results() -> list[dict]:
    rows = []
    for p, role in PAIRS.items():
        row = {"pair": p, "role": role}
        for k, name in RUNS.items():
            r = json.loads((PILOT / f"pair-{p:03d}/analysis/score-{name}.json").read_text("utf-8"))
            row[k] = r["result"]["modulo_symmetry"]
            if k == "method":
                row["chance"] = r["result"]["chance"]["modulo_symmetry"]
                row["components"] = r["result"]["components"]
                row["by_symmetry"] = r["result"]["by_symmetry"]
                row["sentences"] = r["sentences"]
        # Restarts 1 and 2 were scored on 300 held-out episodes. Score them as the score
        # files do: right if the decoded state equals the truth under any symmetry.
        world = make_world(**_world_spec(PILOT / f"pair-{p:03d}"))
        seeds = [row["method"]]
        for s in (1, 2):
            with np.load(PILOT / f"pair-{p:03d}/align" / SEED_FILE.format(s)) as z:
                pred, truth = z["pred"], z["truth"]
            images = [relabel_states(world, r, truth) for r in automorphisms(world)]
            seeds.append(float(np.any([np.all(pred == t, 1) for t in images], 0).mean()))
        row["restarts"] = seeds
        rows.append(row)
    return rows


def gloss(tokens: list[str], key: dict) -> tuple[list[str], str]:
    """Word-by-word glosses, and a plain English reading built from them."""
    concept = {stem: c for c, stem in key["lang"]["stems"].items()}
    # Agents are named by their own words, as a reader of the language would know them.
    concept |= {
        key["lang"]["stems"][a]: key["lang"]["stems"][a].capitalize()
        for a in key["world"]["agents"]
    }
    agents = {key["lang"]["stems"][a].capitalize() for a in key["world"]["agents"]}
    suffix = {f"-{s}": m for m, s in key["lang"]["suffixes"].items()}
    world = key["world"]
    words, subj, place, prep, obj, verb, past = [], None, None, None, None, None, False
    for t in tokens:
        if t in suffix:
            m = suffix[t]
            words.append(
                {"obj": "(object)", "to": "(to)", "at": "(at)", "now": "(now)", "past": "(before)"}[
                    m
                ]
            )
            if m in ("to", "at"):
                prep = m
            past = past or m == "past"
            continue
        c = concept[t]
        words.append(c)
        if c in agents or (c in world["objects"] and subj is None):
            if subj is None:
                subj = c
            else:
                obj = c
        elif c in world["objects"]:
            obj = c
        elif c in world["places"]:
            place = c
        else:
            verb = c
    name = lambda c: c if c in agents else f"the {c}"  # noqa: E731
    v = (PAST if past else VERBS)[verb]
    english = f"{name(subj)} {v}"
    if obj:
        english += f" {name(obj)}"
    if place:
        english += f" {'to' if prep == 'to' else 'at'} the {place}"
    return words, english[0].upper() + english[1:]


def readable(state: np.ndarray, world, agent_names: list[str]) -> dict:
    s = state_from_array(world, state)
    objects = {}
    for o, (kind, i) in zip(world.objects, s.loc, strict=True):
        where = (
            world.places[i] if kind in (AT, BROKEN) else agent_names[i] if kind == HELD else None
        )
        objects[o] = {"state": LOC[kind], "where": where}
    return {
        "agents": {
            agent_names[a]: {"place": world.places[s.pos[a]], "hungry": bool(s.hungry[a])}
            for a in range(len(world.agents))
        },
        "objects": objects,
    }


def sample() -> dict:
    pair = PILOT / f"pair-{SAMPLE_PAIR:03d}"
    data = pair / "data"
    key, lang, labels = load_key(data), load_lang(data), load_lang_labels(data)
    world = make_world(**_world_spec(pair))
    slots = key["obs_slots"]
    shuffle = Relabel(tuple(slots["agents"]), tuple(slots["objects"]), tuple(slots["places"]))
    summary = json.loads((pair / "align/decipher-v2/summary.json").read_text("utf-8"))
    with np.load(pair / f"align/decipher-v2/finalist-{summary['selected']}.npz") as z:
        frames, positions = z["frames"], z["positions"]
    truth = lang_truth(data, positions)
    pred = np.stack([state_to_array(read_frame(world, f, shuffle)) for f in frames])
    swap = Relabel((1, 0), tuple(range(len(world.objects))), tuple(range(len(world.places))))
    swapped = relabel_states(world, swap, truth)
    episode = np.searchsorted(lang.episode_start, positions, side="right") - 1
    stems = key["lang"]["stems"]
    words = [stems[a] for a in world.agents]  # the language's own names for agents A and B
    figures = ["figure 1", "figure 2"]  # the frame stream's agents, in its own (shuffled) order
    # In the frame stream, true agent a is frame agent slots["agents"][a].
    fig_of = {a: figures[slots["agents"][a]] for a in range(len(world.agents))}
    row = np.searchsorted(labels.sentences[:, 4], positions, side="right") - 1
    out = []
    for e in dict.fromkeys(episode.tolist()):  # scoring order, first appearance
        idx = np.flatnonzero(episode == e)
        if len(idx) < MIN_SENTENCES:
            continue
        start = lang.episode_start[e]
        sents, prev = [], start + 1  # skip the episode marker
        for i in idx:
            span = [lang.words[t] for t in lang.tokens[prev : positions[i]]]
            prev = positions[i] + 1
            g, english = gloss(span, key)
            # The decoded state is in frame terms, so its agents are named by figure.
            decoded = readable(pred[i], world, [fig_of[a] for a in range(len(world.agents))])
            sents.append(
                {
                    "words": span,
                    "gloss": g,
                    "english": english,
                    "kind": KINDS[int(labels.sentences[row[i], 2])],
                    "truth": readable(truth[i], world, [words[a].capitalize() for a in range(2)]),
                    "decoded": decoded,
                    "frame": "".join(map(str, frames[i].astype(int))),
                    "true_frame": "".join(
                        map(str, encode(world, state_from_array(world, truth[i]), None, shuffle))
                    ),
                    "exact": bool(np.all(pred[i] == truth[i])),
                    "swapped": bool(np.all(pred[i] == swapped[i])),
                }
            )
        out.append({"episode": int(e), "sentences": sents})
        if len(out) == SAMPLE_EPISODES:
            break
    return {
        "pair": SAMPLE_PAIR,
        "agents": {"A": words[0], "B": words[1]},
        "figures": {words[a]: fig_of[a] for a in range(2)},
        "lexicon": {stem: c for c, stem in stems.items()},
        "suffixes": {f"-{s}": m for m, s in key["lang"]["suffixes"].items()},
        "world": {
            "places": list(world.places),
            "edges": [list(e) for e in world.edges],
            "objects": list(world.objects),
        },
        "frame_size": int(load_obs(data).frames.shape[1]),
        "rule": f"First {SAMPLE_EPISODES} held-out episodes in scoring order with at least "
        f"{MIN_SENTENCES} sentences; not chosen by accuracy.",
        "episodes": out,
    }


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    data = {"results": results(), "sample": sample()}
    for name, part in data.items():
        (out / f"{name}.json").write_text(json.dumps(part, separators=(",", ":")), "utf-8")
    for r in data["results"]:
        print(
            r["pair"],
            {k: round(r[k], 3) for k in ("method", "counts", "baseline", "chance")},
            [round(x, 3) for x in r["restarts"]],
        )
    s = data["sample"]
    print(
        "sample:",
        [len(e["sentences"]) for e in s["episodes"]],
        s["figures"],
        f"{(out / 'sample.json').stat().st_size} bytes",
    )
    for x in s["episodes"][0]["sentences"][:8]:
        print(
            " ",
            " ".join(x["words"]),
            "|",
            x["english"],
            "| exact",
            x["exact"],
            "swapped",
            x["swapped"],
        )


if __name__ == "__main__":
    main()
