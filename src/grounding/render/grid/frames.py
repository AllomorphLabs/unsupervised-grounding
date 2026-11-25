"""Grid "video" observations: the main observation stream (proposal, design step 3).

Each frame is a small image: a grid of cells, each with a few yes/no channels.

- **Places** are separate patches of cells (``REGION`` rows by ``REGION`` columns each), side by
  side in one row with a one-cell gap between patches. Which patch shows which place is shuffled
  per pair. Nothing marks a patch as a place: it is just where things appear.
- **Agents and objects** are drawn as single cells, each in a channel of its own appearance. An
  agent has two appearances, normal and hungry; a fragile object has two, intact and broken.
  Which channel draws which entity is shuffled per pair, so the two interchangeable agents look
  alike except for a colour that is assigned at random: the agent symmetry survives (proposal,
  renderer checks).
- **Where in its patch** an entity is drawn is random: a free cell is chosen when it arrives (or at
  the episode start) and kept while it stays. An object being carried is drawn in its carrier's
  cell and moves with it; dropped, it lands in a random free cell. Nothing else ever shares a
  cell, so two things in one cell means one is carrying the other. An eaten object disappears.
- **Nothing marks who acted**, by default: a wait changes nothing on screen. With
  ``mark_actor``, one extra channel, shared by all agents, marks the cell of the agent that acted
  on the step that led to the frame (as the state vectors' "acted" flags do; experiment log,
  entry 66).

The frame is stored flat, cell by cell (row-major), with the channels of each cell together:
value ``cell * channels + channel``. The grid's shape is public (``obs.npz`` holds it), as an
image's size is; what each channel and patch means is hidden in the pair's key.
"""

from dataclasses import asdict, dataclass

import numpy as np

from grounding.sim.world import AT, BROKEN, GONE, HELD, State, World

REGION = (2, 3)  # rows, columns of cells per place
GAP = 1  # empty columns between places


@dataclass(frozen=True)
class GridKey:
    """How one pair draws the world: hidden, for scoring only."""

    slot: tuple[int, ...]  # place -> patch position, left to right
    agent: tuple[tuple[int, int], ...]  # agent -> (normal channel, hungry channel)
    obj: tuple[tuple[int, int], ...]  # object -> (intact channel, broken channel or -1)
    actor: int = -1  # the channel marking who acted, or -1 for none

    @property
    def channels(self) -> int:
        n = 2 * len(self.agent) + sum(1 + (b >= 0) for _, b in self.obj)
        return n + (self.actor >= 0)

    @classmethod
    def random(cls, world: World, rng: np.random.Generator, mark_actor: bool = False) -> GridKey:
        n_ch = 2 * len(world.agents) + len(world.objects) + len(world.fragile)
        ch = [int(c) for c in rng.permutation(n_ch)]
        agent = tuple((ch.pop(), ch.pop()) for _ in world.agents)
        obj = tuple(
            (ch.pop(), ch.pop() if o in world.fragile else -1) for o in range(len(world.objects))
        )
        slot = tuple(int(i) for i in rng.permutation(len(world.places)))
        return cls(slot, agent, obj, n_ch if mark_actor else -1)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> GridKey:
        return cls(
            tuple(d["slot"]),
            tuple(map(tuple, d["agent"])),
            tuple(map(tuple, d["obj"])),
            d.get("actor", -1),
        )


def grid_shape(world: World, key: GridKey) -> tuple[int, int, int]:
    rows, cols = REGION
    n = len(world.places)
    return rows, n * cols + (n - 1) * GAP, key.channels


def place_cells(world: World, place: int, key: GridKey) -> list[int]:
    """Cell numbers of a place's patch."""
    rows, cols = REGION
    _, width, _ = grid_shape(world, key)
    left = key.slot[place] * (cols + GAP)
    return [r * width + left + c for r in range(rows) for c in range(cols)]


def frame_size(world: World, key: GridKey) -> int:
    h, w, c = grid_shape(world, key)
    return h * w * c


class EpisodeRenderer:
    """Draws one episode frame by frame, remembering where in its patch each thing sits."""

    def __init__(self, world: World, key: GridKey, rng: np.random.Generator):
        self.world, self.key, self.rng = world, key, rng
        self.cells = {p: place_cells(world, p, key) for p in range(len(world.places))}
        self.agent_cell: list[int] = []
        self.obj_cell: list[int | None] = []
        self.prev: State | None = None

    def _free(self, place: int, taken: set[int]) -> int:
        free = [c for c in self.cells[place] if c not in taken]
        return free[self.rng.integers(len(free))]

    def _taken(self, state: State, skip_agent: int = -1, skip_obj: int = -1) -> set[int]:
        out = {c for a, c in enumerate(self.agent_cell) if a != skip_agent}
        for o, (kind, _) in enumerate(state.loc):
            if o != skip_obj and kind in (AT, BROKEN) and self.obj_cell[o] is not None:
                out.add(self.obj_cell[o])
        return out

    def start(self, s: State) -> np.ndarray:
        self.agent_cell, self.obj_cell = [], [None] * len(s.loc)
        taken: set[int] = set()
        for p in s.pos:
            c = self._free(p, taken)
            taken.add(c)
            self.agent_cell.append(c)
        for o, (kind, idx) in enumerate(s.loc):
            if kind in (AT, BROKEN):
                c = self._free(idx, taken)
                taken.add(c)
                self.obj_cell[o] = c
        self.prev = s
        return self.draw(s)

    def step(self, s: State, actor: int | None = None) -> np.ndarray:
        prev = self.prev
        for a, (p0, p1) in enumerate(zip(prev.pos, s.pos, strict=True)):
            if p0 != p1:
                self.agent_cell[a] = self._free(p1, self._taken(prev, skip_agent=a))
        for o, (old, new) in enumerate(zip(prev.loc, s.loc, strict=True)):
            if new[0] in (AT, BROKEN) and old != new:
                self.obj_cell[o] = self._free(new[1], self._taken(s, skip_obj=o))
            elif new[0] in (HELD, GONE):
                self.obj_cell[o] = None
        self.prev = s
        return self.draw(s, actor)

    def draw(self, s: State, actor: int | None = None) -> np.ndarray:
        k = self.key
        n_ch = k.channels
        f = np.zeros(frame_size(self.world, k), dtype=np.uint8)
        for a, cell in enumerate(self.agent_cell):
            f[cell * n_ch + k.agent[a][int(s.hungry[a])]] = 1
        for o, (kind, idx) in enumerate(s.loc):
            intact, broken = k.obj[o]
            if kind == AT:
                f[self.obj_cell[o] * n_ch + intact] = 1
            elif kind == HELD:
                f[self.agent_cell[idx] * n_ch + intact] = 1
            elif kind == BROKEN:
                f[self.obj_cell[o] * n_ch + broken] = 1
        if k.actor >= 0 and actor is not None:
            f[self.agent_cell[actor] * n_ch + k.actor] = 1
        return f


def decode(world: World, frame: np.ndarray, key: GridKey) -> State:
    """The state a frame shows. For scoring only: the key is hidden.

    Each entity is read from its strongest cell, so a frame that isn't exactly one thing per entity
    (a model's guess) still decodes to a state. Cells outside every patch are ignored.
    """
    h, w, n_ch = grid_shape(world, key)
    v = np.asarray(frame, dtype=np.float64).reshape(h * w, n_ch)
    where = np.full(h * w, -1)
    for p in range(len(world.places)):
        where[place_cells(world, p, key)] = p
    live = where >= 0
    v = v * live[:, None]
    pos, hungry, cells = [], [], []
    for normal, hung in key.agent:
        both = np.maximum(v[:, normal], v[:, hung])
        c = int(np.argmax(both))
        cells.append(c)
        pos.append(int(where[c]))
        hungry.append(bool(v[c, hung] > v[c, normal]))
    loc = []
    for intact, broken in key.obj:
        ci = int(np.argmax(v[:, intact]))
        best_i = v[ci, intact]
        cb = int(np.argmax(v[:, broken])) if broken >= 0 else -1
        best_b = v[cb, broken] if broken >= 0 else 0.0
        if max(best_i, best_b) < 0.5:
            loc.append((GONE, -1))
        elif best_b > best_i:
            loc.append((BROKEN, int(where[cb])))
        elif ci in cells:
            loc.append((HELD, cells.index(ci)))
        else:
            loc.append((AT, int(where[ci])))
    return State(tuple(pos), tuple(hungry), tuple(loc))
