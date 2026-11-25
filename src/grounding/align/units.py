"""The structure of a grid-video frame stream, found from that stream alone.

Decipherment reads each frame as a short list of *units*, the facts a sentence could be about.
This module finds them using only the frame stream's own statistics and what any image format
tells you: that a frame is a grid of cells with a few channels each. Nothing here knows what a
channel or a patch means, looks at the language stream, or reads ``hidden/``.

- **Places** are the separate patches of the image where anything ever appears: connected
  groups of cells (touching side by side) that are ever drawn on.
- **Things** are groups of values that behave as one thing that is always somewhere: at most
  one of them is ever on at once, and the thing's presence hands over from one value to another
  between frames. They are found by merging groups of values, most hand-overs first, while no two
  values of a merged group are ever on together.
- **Appearances** are the channels a thing is drawn in (an agent's normal and hungry colour).

The units are then, with where a thing is kept apart from how it looks, so that "became hungry"
and "walked somewhere while hungry" are different facts (experiment log, entry 61):

- ``("at", thing, place)``: the thing is in that place, however it is drawn;
- ``("looks", thing, channel)``: the thing is drawn in that channel, wherever it is (only for
  things drawn in more than one channel);
- ``("with", thing, thing)``: the two things are drawn in the same cell (one carries the other);
- ``("gone", thing)``: the thing is nowhere on screen.

Units that never occur in the training frames are left out.
"""

from dataclasses import dataclass

import numpy as np
import torch

from grounding.data.load import ObsStream


@dataclass(frozen=True)
class Units:
    size: int  # values per frame
    shape: tuple[int, int, int]
    places: int
    things: tuple[tuple[int, ...], ...]  # frame values of each thing
    units: tuple[tuple, ...]  # one descriptor per unit (see module docstring)
    place_of_cell: tuple[int, ...]  # per cell, its place, or -1
    always: tuple[bool, ...] = ()  # per thing, whether it is on screen in every training frame
    # State vectors (find_units_vector): the frame values whose sum is each unit. Grid video
    # leaves this empty and computes units from cells.
    members: tuple[tuple[int, ...], ...] = ()

    @property
    def n(self) -> int:
        return len(self.units)


def _episode_of(obs: ObsStream) -> np.ndarray:
    return np.searchsorted(obs.episode_start, np.arange(len(obs.frames)), side="right") - 1


def _components(live: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Label 4-connected groups of live cells 0, 1, ...; dead cells -1."""
    lab = np.full(rows * cols, -1)
    n = 0
    for start in range(rows * cols):
        if not live[start] or lab[start] >= 0:
            continue
        stack = [start]
        lab[start] = n
        while stack:
            c = stack.pop()
            r, k = divmod(c, cols)
            for rr, kk in ((r - 1, k), (r + 1, k), (r, k - 1), (r, k + 1)):
                if 0 <= rr < rows and 0 <= kk < cols:
                    q = rr * cols + kk
                    if live[q] and lab[q] < 0:
                        lab[q] = n
                        stack.append(q)
        n += 1
    return lab


def _things(fr: np.ndarray, f0: np.ndarray, f1: np.ndarray, live: np.ndarray, min_hand: float):
    """Group live values into things (see module docstring)."""
    idx = np.flatnonzero(live)
    x = fr[:, idx].astype(np.float64)
    co = x.T @ x
    on = np.clip(f1[:, idx] - f0[:, idx], 0, 1).astype(np.float64)
    off = np.clip(f0[:, idx] - f1[:, idx], 0, 1).astype(np.float64)
    hand = off.T @ on
    hand = hand + hand.T
    groups = [[i] for i in range(len(idx))]
    G = np.eye(len(idx))  # value -> group indicator
    while True:
        gco = G.T @ co @ G
        gh = G.T @ hand @ G
        np.fill_diagonal(gh, 0)
        gh[gco > 0] = 0
        a, b = np.unravel_index(np.argmax(gh), gh.shape)
        if gh[a, b] < min_hand:
            break
        a, b = min(a, b), max(a, b)
        groups[a] += groups[b]
        del groups[b]
        G[:, a] += G[:, b]
        G = np.delete(G, b, axis=1)
    return [tuple(int(idx[i]) for i in sorted(g)) for g in groups if len(g) > 1]


def find_units(obs: ObsStream, min_hand: float = 20.0, split: bool = True) -> Units:
    """Places, things and units of a grid-video stream, from its training episodes.

    With ``split`` (the default), where a thing is and how it looks are separate units; without
    it, each "at" unit also names the channel, ``("at", thing, place, channel)``, and there are no
    "looks" units."""
    if obs.shape is None:
        raise ValueError("units need a grid-video stream (its frames have a shape)")
    rows, cols, n_ch = obs.shape
    d = rows * cols * n_ch
    use = ~obs.held_out
    ep = _episode_of(obs)
    fr = obs.frames[use[ep]]
    live = fr.any(0)
    place_of_cell = _components(live.reshape(rows * cols, n_ch).any(1), rows, cols)
    n_places = int(place_of_cell.max()) + 1

    nxt = np.flatnonzero((ep[:-1] == ep[1:]) & use[ep[:-1]])
    things = _things(fr, obs.frames[nxt], obs.frames[nxt + 1], live, min_hand)

    # Which units ever occur.
    T = len(things)
    seen_at: set[tuple] = set()
    seen_look: set[tuple] = set()
    for k, vals in enumerate(things):
        v = np.array(vals)
        _, which = np.nonzero(fr[:, v])
        for val in set(v[which].tolist()):
            cell, ch = divmod(val, n_ch)
            seen_at.add((k, int(place_of_cell[cell])) + (() if split else (ch,)))
            seen_look.add((k, ch))
    units: list[tuple] = [("at", *x) for x in sorted(seen_at)]
    for k in range(T if split else 0):
        looks = sorted(ch for kk, ch in seen_look if kk == k)
        if len(looks) > 1:
            units += [("looks", k, ch) for ch in looks]
    present = np.stack([fr[:, list(vals)].sum(1) > 0 for vals in things], 1)
    for k in range(T):
        if not present[:, k].all():
            units.append(("gone", k))
    cell_of_thing = np.full((len(fr), T), -1)
    for k, vals in enumerate(things):
        v = np.array(vals)
        on = fr[:, v]
        r, w = np.nonzero(on)
        cell_of_thing[r, k] = v[w] // n_ch
    for a in range(T):
        for b in range(a + 1, T):
            same = (cell_of_thing[:, a] >= 0) & (cell_of_thing[:, a] == cell_of_thing[:, b])
            if same.any():
                units.append(("with", a, b))
    return Units(
        size=d,
        shape=(rows, cols, n_ch),
        places=n_places,
        things=tuple(things),
        units=tuple(units),
        place_of_cell=tuple(int(x) for x in place_of_cell),
        always=tuple(bool(present[:, k].all()) for k in range(T)),
    )


def find_units_vector(
    obs: ObsStream, min_hand: float = 20.0, share: float = 0.98, min_count: int = 10
) -> Units:
    """Things, places and units of a state-vector stream, from its statistics alone.

    A state vector is a list of yes/no facts in a shuffled order, so there are no cells or
    patches to find places from. Everything comes from which values change together
    (experiment log, entry 67):

    - *Things* are found as for grid video (``_things``): values of which at most one is on, and
      between which a thing's presence hands over.
    - A thing's *actor value* is a value outside it that is on whenever the thing changes: the
      thing changes only when that value says it acted. Things with one are the actors.
    - *Places:* when a value of another thing changes while an actor acted, and the actor is
      then always in the same one of its values, the two values are at the same place. Places
      are the connected groups of values linked this way that include an actor's values.
    - Each *named thing* (a thing with values at places) gets ``("at", thing, place, aspect)``
      units, where the aspect separates a thing's values at the same place: 0 if the value is ever
      seen at an episode start, 1 if not (an intact glass and a broken one).
    - A value of a thing that isn't at a place but changes only when an actor acts is
      ``("with", thing, actor)`` (an object held by an agent).
    - A value outside every thing that changes only when an actor acts, and an actor value
      itself, is ``("looks", actor, value)``: a flag of that actor (hungry; acted).
    - Any other value of a thing is ``("gone", thing)``.
    """
    use = ~obs.held_out
    ep = _episode_of(obs)
    fr = obs.frames[use[ep]]
    live = fr.any(0)
    d = fr.shape[1]
    nxt = np.flatnonzero((ep[:-1] == ep[1:]) & use[ep[:-1]])
    f0, f1 = obs.frames[nxt], obs.frames[nxt + 1]
    blocks = _things(fr, f0, f1, live, min_hand)
    tid = np.full(d, -1)
    for k, t in enumerate(blocks):
        tid[list(t)] = k
    a0, a1 = f0.astype(np.float64), f1.astype(np.float64)
    changed = (a0 != a1).astype(np.float64)
    stay = a0 * a1

    def coincides(mask: np.ndarray) -> int | None:
        """A value (outside ``mask``'s own thing) on in every frame where ``mask`` changes."""
        if mask.sum() < min_count:
            return None
        on = a1[mask].mean(0)
        return int(np.argmax(on)) if on.max() > 0.99 else None

    actor = {}
    for k, t in enumerate(blocks):
        m = changed[:, list(t)].sum(1) > 0
        on = a1[m].mean(0)
        on[list(t)] = 0
        w = int(np.argmax(on))
        if m.sum() >= min_count and on[w] > 0.99:
            actor[k] = w
    # Place links: value u of another thing changes while actor k acted; k is then at one value.
    parent = {}

    def root(x):
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x

    for u in np.flatnonzero(live):
        for k, w in actor.items():
            if tid[u] == k:
                continue
            m = (changed[:, u] > 0) & (a1[:, w] > 0)
            if m.sum() < min_count:
                continue
            row = stay[m][:, list(blocks[k])].sum(0)
            if row.sum() >= min_count and row.max() / row.sum() > share:
                parent[root(int(u))] = root(int(blocks[k][int(np.argmax(row))]))
    actor_values = {v for k in actor for v in blocks[k]}
    groups: dict[int, list[int]] = {}
    for v in list(parent):
        groups.setdefault(root(v), []).append(v)
    places = [sorted(g) for g in groups.values() if any(v in actor_values for v in g)]
    places.sort(key=min)
    place_of = {v: p for p, g in enumerate(places) for v in g}
    starts = obs.frames[obs.episode_start[use]].any(0)

    named = [k for k, t in enumerate(blocks) if any(v in place_of for v in t)]
    new_id = {k: i for i, k in enumerate(named)}
    units: list[tuple] = []
    members: list[tuple[int, ...]] = []
    at: dict[tuple, list[int]] = {}
    for k in named:
        for v in blocks[k]:
            if v in place_of:
                at.setdefault((new_id[k], place_of[v], 0 if starts[v] else 1), []).append(v)
    for key_ in sorted(at):
        units.append(("at", *key_))
        members.append(tuple(at[key_]))
    actor_of_value = {w: new_id[k] for k, w in actor.items() if k in new_id}
    for k in named:
        for v in blocks[k]:
            if v in place_of:
                continue
            w = coincides(changed[:, v] > 0)
            if w is not None and w in actor_of_value:
                units.append(("with", new_id[k], actor_of_value[w]))
            else:
                units.append(("gone", new_id[k]))
            members.append((int(v),))
    in_thing = {v for k in named for v in blocks[k]}
    for v in np.flatnonzero(live):
        v = int(v)
        if v in in_thing:
            continue
        if v in actor_of_value:
            units.append(("looks", actor_of_value[v], v))
            members.append((v,))
            continue
        w = coincides(changed[:, v] > 0)
        if w is not None and w in actor_of_value:
            units.append(("looks", actor_of_value[w], v))
            members.append((v,))
    return Units(
        size=d,
        shape=None,
        places=len(places),
        things=tuple(tuple(int(v) for v in blocks[k]) for k in named),
        units=tuple(units),
        place_of_cell=(),
        always=tuple(bool((fr[:, list(blocks[k])].sum(1) == 1).all()) for k in named),
        members=tuple(members),
    )


class UnitView:
    """Turns frames (n x values) into units (n x units), on any device, differentiably simple."""

    def __init__(self, u: Units, device: str):
        if u.members:
            self._init_vector(u, device)
            return
        self.members = None
        rows, cols, n_ch = u.shape
        d, n_cells = u.size, rows * cols
        # "at" and "looks" units are both sums of a thing's values, so one matrix gives both.
        simple = [x for x in u.units if x[0] in ("at", "looks")]
        P = torch.zeros(d, len(simple), device=device)
        for j, x in enumerate(simple):
            for val in u.things[x[1]]:
                cell, c = divmod(val, n_ch)
                at = x[0] == "at" and u.place_of_cell[cell] == x[2]
                if (at and (len(x) == 3 or c == x[3])) or (x[0] == "looks" and c == x[2]):
                    P[val, j] = 1
        self.P = P
        # What a frame consistent with the structure looks like: nothing outside the things,
        # each thing drawn at most once, and a thing that is always on screen drawn exactly once.
        self.outside = torch.ones(d, device=device)
        for vals in u.things:
            self.outside[list(vals)] = 0
        self.always = torch.tensor(u.always, device=device)
        self.M = torch.zeros(len(u.things), d, n_cells, device=device)  # thing -> its cells
        self.B = torch.zeros(d, len(u.things), device=device)  # value -> thing
        for k, vals in enumerate(u.things):
            for val in vals:
                self.M[k, val, val // n_ch] = 1
                self.B[val, k] = 1
        self.gone = [x[1] for x in u.units if x[0] == "gone"]
        self.with_ = [(x[1], x[2]) for x in u.units if x[0] == "with"]

    def _init_vector(self, u: Units, device: str) -> None:
        """State vectors: every unit is the sum of its member values."""
        self.members = u.members
        self.P = torch.zeros(u.size, len(u.units), device=device)
        for j, vals in enumerate(u.members):
            self.P[list(vals), j] = 1
        self.B = torch.zeros(u.size, len(u.things), device=device)
        for k, vals in enumerate(u.things):
            self.B[list(vals), k] = 1
        self.always = torch.tensor(u.always, device=device)
        self.outside = torch.ones(u.size, device=device)
        for vals in u.members:
            self.outside[list(vals)] = 0

    def valid(self, f: torch.Tensor) -> torch.Tensor:
        """Whether each frame keeps the structure found in the frame stream."""
        n = f @ self.B
        ok = (n <= 1.5).all(1) & ((n > 0.5) | ~self.always).all(1)
        return ok & ((f @ self.outside) < 0.5)

    def __call__(self, f: torch.Tensor) -> torch.Tensor:
        if self.members is not None:
            return f @ self.P
        parts = [f @ self.P]
        if self.gone:
            present = f @ self.B
            parts.append((1 - present[:, self.gone]).clamp(0, 1))
        if self.with_:
            cells = torch.einsum("nd,kdc->nkc", f, self.M)
            parts.append(
                torch.stack([(cells[:, a] * cells[:, b]).sum(1) for a, b in self.with_], 1).clamp(
                    0, 1
                )
            )
        return torch.cat(parts, 1)
