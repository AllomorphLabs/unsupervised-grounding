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


class UnitView:
    """Turns frames (n x values) into units (n x units), on any device, differentiably simple."""

    def __init__(self, u: Units, device: str):
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

    def valid(self, f: torch.Tensor) -> torch.Tensor:
        """Whether each frame keeps the structure found in the frame stream."""
        n = f @ self.B
        ok = (n <= 1.5).all(1) & ((n > 0.5) | ~self.always).all(1)
        return ok & ((f @ self.outside) < 0.5)

    def __call__(self, f: torch.Tensor) -> torch.Tensor:
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
