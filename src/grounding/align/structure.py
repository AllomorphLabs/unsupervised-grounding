"""Each stream's structure, found from that stream alone (experiment log, entry 25).

Alignment needs to know which frame values belong to which entity, and which words name
entities. Neither is given: frame values are in a shuffled order and token numbers are
random (technical decision 11). Both are read off each stream's own statistics here, before
the streams are compared. Nothing in this module looks at the other stream or the truth.

Frames (``frame_layout``):

- Values that hand over to each other between frames and are never on together form a
  one-of-n *block* (an entity's position, for example).
- The *acted* flags are the single values that are off at every episode start and of which
  exactly one is on in every later frame: one per agent.
- An *agent block* is a block whose values can all appear at an episode start and whose every
  change comes with one acted flag. Each agent's *hunger* flag is the remaining single value
  that switches on with that agent's acted flag.
- The other blocks are *objects*. An object's "at" values are those on at episode starts; its
  "held" values switch on from an "at" value along with one agent's acted flag; any other value
  is "broken" (at a place) if the acting agent's place is consistent when it switches on, and
  otherwise "gone".
- Places are lined up across blocks through take events: when an object leaves "at p" for
  "held", the acting agent is at p.

Language (``word_types``), from where stems sit in sentences: verbs are stems whose suffix ends
the sentence; places take two or more different suffixes; objects take exactly one; agents
take none.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from grounding.data.load import LangStream, ObsStream
from grounding.data.tokens import EPISODE, STOP


@dataclass(frozen=True)
class FrameLayout:
    """What each frame value means, up to the names of the entities.

    ``values[v]`` is ``(kind, entity, extra)``. Kinds: ``place`` (agent ``entity`` at place id
    ``extra``), ``hungry`` and ``acted`` (agent ``entity``), ``obj_at`` and ``obj_broken`` (object
    ``entity`` at place id ``extra``), ``obj_held`` (object ``entity`` held by agent ``extra``),
    ``obj_gone``. Place ids are positions in agent 0's block. Values never seen on are absent.
    """

    size: int
    blocks: tuple[tuple[int, ...], ...]  # every one-of-n block
    singles: tuple[int, ...]  # live values outside blocks (acted and hunger flags)
    agents: tuple[tuple[tuple[int, ...], int, int], ...]  # per agent: (place values, acted, hunger)
    objects: tuple[tuple[int, ...], ...]  # per object: its block
    places: int
    values: dict[int, tuple[str, int, int | None]]


@dataclass(frozen=True)
class WordTypes:
    agents: tuple[int, ...]
    objects: tuple[int, ...]
    places: tuple[int, ...]
    verbs: tuple[int, ...]


def _transitions(obs: ObsStream, use: np.ndarray):
    """Distinct consecutive-frame transitions within episodes, with counts."""
    ep = np.searchsorted(obs.episode_start, np.arange(len(obs.frames)), side="right") - 1
    nxt = np.flatnonzero((ep[:-1] == ep[1:]) & use[ep[:-1]])
    pairs = np.concatenate([obs.frames[nxt], obs.frames[nxt + 1]], 1)
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)
    d = obs.frames.shape[1]
    return uniq[:, :d].astype(np.int64), uniq[:, d:].astype(np.int64), counts.astype(np.float64)


def frame_layout(obs: ObsStream) -> FrameLayout:
    """Read the frame stream's structure from its training episodes."""
    use = ~obs.held_out
    ep = np.searchsorted(obs.episode_start, np.arange(len(obs.frames)), side="right") - 1
    fr = obs.frames[use[ep]].astype(np.float64)
    d = fr.shape[1]
    f0, f1, w = _transitions(obs, use)
    on, off = np.clip(f1 - f0, 0, 1), np.clip(f0 - f1, 0, 1)
    co = fr.T @ fr
    hand = (off * w[:, None]).T @ on
    hand = hand + hand.T
    parent = list(range(d))

    def root(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    for i in range(d):
        for j in range(i + 1, d):
            if co[i, j] == 0 and hand[i, j] > 0.01 * w.sum() / d:
                parent[root(i)] = root(j)
    groups: dict[int, list[int]] = {}
    for i in range(d):
        groups.setdefault(root(i), []).append(i)
    blocks = [tuple(g) for g in groups.values() if len(g) > 1 and np.all(fr[:, g].sum(1) == 1)]
    used = {v for b in blocks for v in b}
    singles = [v for v in range(d) if v not in used and fr[:, v].any()]

    starts = obs.frames[obs.episode_start[use]].astype(bool)
    start_on = starts.any(0)
    later = np.ones(len(obs.frames), bool)
    later[obs.episode_start] = False
    later_fr = obs.frames[later & use[ep]].astype(np.int64)
    # Acted flags: off at every start, exactly one of them on in (almost) every later frame.
    candidates = [v for v in singles if not start_on[v]]
    acted: list[int] = []
    for n in range(len(candidates), 0, -1):
        for subset in combinations(candidates, n):
            if np.mean(later_fr[:, list(subset)].sum(1) == 1) > 0.99:
                acted = list(subset)
                break
        if acted:
            break
    if not acted:
        raise ValueError("no set of single values marks exactly one actor per step")

    def share_with(mask: np.ndarray, flag: int) -> float:
        return float((w[mask] * f1[mask, flag]).sum() / max(w[mask].sum(), 1e-12))

    agents = []
    for b in blocks:
        changed = on[:, list(b)].sum(1) > 0
        shares = [share_with(changed, a) for a in acted]
        if start_on[list(b)].all() and max(shares) > 0.99:
            agents.append((b, acted[int(np.argmax(shares))]))
    if sorted(a for _, a in agents) != sorted(acted):
        raise ValueError("couldn't match each acted flag to one agent block")
    agents.sort(key=lambda x: acted.index(x[1]))
    others = [v for v in singles if v not in acted]
    hunger = [max(others, key=lambda h: share_with(on[:, h] > 0, a)) for _, a in agents]

    def actor_place(mask: np.ndarray, k: int) -> np.ndarray:
        """Weight on each of agent k's place values when ``mask`` happens while agent k acts."""
        block, flag = agents[k]
        m = mask & (f1[:, flag] > 0)
        return (w[m][:, None] * f0[m][:, list(block)]).sum(0)

    object_blocks = [b for b in blocks if all(b != ab for ab, _ in agents)]
    values: dict[int, tuple[str, int, int | None]] = {}
    for k, (_block, flag) in enumerate(agents):
        values[flag] = ("acted", k, None)
        values[hunger[k]] = ("hungry", k, None)
    # Line up places: an object's "at" value, as it's taken, against each agent's place.
    n_places = len(agents[0][0])
    vote = [np.zeros((n_places, n_places)) for _ in agents]
    kinds: list[dict[int, tuple]] = []
    for b in object_blocks:
        info: dict[int, tuple] = {}
        at = [v for v in b if start_on[v]]
        for v in b:
            if v in at:
                p0 = int(np.argmax(actor_place(off[:, v] > 0, 0)))
                info[v] = ("obj_at", p0)
                for k in range(1, len(agents)):
                    vote[k][p0, int(np.argmax(actor_place(off[:, v] > 0, k)))] += 1
            else:
                from_at = (on[:, v] > 0) & (f0[:, at].sum(1) > 0)
                held = [share_with(from_at, a) for _, a in agents]
                info[v] = ("obj_held", int(np.argmax(held))) if max(held) > 0.99 else ("other",)
        kinds.append(info)
    place_id = [{v: i for i, v in enumerate(agents[0][0])}]
    for k in range(1, len(agents)):
        block = agents[k][0]
        place_id.append({block[j]: int(np.argmax(vote[k][:, j])) for j in range(n_places)})
    for k, (block, _) in enumerate(agents):
        for v in block:
            values[v] = ("place", k, place_id[k][v])
    for n, info in enumerate(kinds):
        for v, t in info.items():
            if t[0] == "other":
                dist = np.zeros(n_places)
                for k in range(len(agents)):
                    for j, u in enumerate(agents[k][0]):
                        dist[place_id[k][u]] += actor_place(on[:, v] > 0, k)[j]
                t = (
                    ("obj_broken", int(np.argmax(dist)))
                    if dist.max() > 0.9 * dist.sum()
                    else ("obj_gone",)
                )
            values[v] = (t[0], n, t[1] if len(t) > 1 else None)
    return FrameLayout(
        size=d,
        blocks=tuple(blocks),
        singles=tuple(singles),
        agents=tuple((tuple(b), a, h) for (b, a), h in zip(agents, hunger, strict=True)),
        objects=tuple(object_blocks),
        places=n_places,
        values=values,
    )


def word_types(lang: LangStream, max_tokens: int = 3_000_000) -> WordTypes:
    """Agents, objects, places and verbs, from where stems sit in training sentences."""
    per_token = np.repeat(lang.held_out, np.diff(np.append(lang.episode_start, len(lang.tokens))))
    toks = lang.tokens[~per_token][:max_tokens]
    suffix = np.array([w.startswith("-") for w in lang.words])
    stop, epi = lang.words.index(STOP), lang.words.index(EPISODE)
    cur, nxt1, nxt2 = toks[:-2], toks[1:-1], toks[2:]
    out: dict[str, list[int]] = {"agents": [], "objects": [], "places": [], "verbs": []}
    for t in range(len(lang.words)):
        if suffix[t] or t in (stop, epi):
            continue
        m = cur == t
        if not m.any():
            continue
        after = nxt1[m]
        counts = np.bincount(after[suffix[after]], minlength=len(lang.words))
        n_suffixes = int((counts / max(counts.sum(), 1) > 0.05).sum())
        if counts.sum() > 0 and np.mean(nxt2[m][suffix[after]] == stop) > 0.5:
            out["verbs"].append(t)
        elif n_suffixes >= 2:
            out["places"].append(t)
        elif n_suffixes == 1:
            out["objects"].append(t)
        else:
            out["agents"].append(t)
    return WordTypes(**{k: tuple(v) for k, v in out.items()})
