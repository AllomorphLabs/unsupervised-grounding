"""State-vector observations: the pilot's observation stream.

Each frame is a flat vector of 0s and 1s, in blocks:

- per agent: which place it is at (one value per place), whether it is
  hungry, and whether it acted on this step;
- per object: where it is (at a place, held by an agent, broken at a place,
  or gone), one value per possibility.

The first frame of an episode shows the start, with no agent marked as acting.
Each later frame shows the state after a step and marks the agent that acted,
so every event, including a wait, is visible (technical decision 12).

The order of agents, objects and places within the vector is shuffled per
pair (technical decision 11): a frame encodes ``shuffle.state(s)`` in the
fixed layout, so slot ``k`` holds whichever entity the shuffle sends to ``k``.
"""

import numpy as np

from grounding.sim.symmetry import Relabel
from grounding.sim.world import AT, BROKEN, GONE, HELD, State, World


def frame_size(world: World) -> int:
    a, o, p = len(world.agents), len(world.objects), len(world.places)
    return a * (p + 2) + o * (2 * p + a + 1)


def encode(world: World, state: State, actor: int | None, shuffle: Relabel) -> np.ndarray:
    """One frame. ``actor`` is the agent that acted on the step that led here, if any."""
    a, p = len(world.agents), len(world.places)
    s = shuffle.state(state)
    acted = None if actor is None else shuffle.agents[actor]
    v = np.zeros(frame_size(world), dtype=np.uint8)
    off = 0
    for i in range(a):
        v[off + s.pos[i]] = 1
        v[off + p] = s.hungry[i]
        v[off + p + 1] = acted == i
        off += p + 2
    for kind, idx in s.loc:
        if kind == AT:
            v[off + idx] = 1
        elif kind == HELD:
            v[off + p + idx] = 1
        elif kind == BROKEN:
            v[off + p + a + idx] = 1
        elif kind == GONE:
            v[off + 2 * p + a] = 1
        off += 2 * p + a + 1
    return v


def decode(world: World, frame: np.ndarray, shuffle: Relabel) -> State:
    """The state a frame shows (the inverse of ``encode``, ignoring who acted).

    Each block is read by its largest value, so a frame that isn't exactly one-hot (for
    example a model's guess) still decodes to a state. For scoring only: the shuffle is part
    of the pair's hidden key.
    """
    a, p = len(world.agents), len(world.places)
    inv = shuffle.inverse()
    v = np.asarray(frame)
    pos, hungry, off = [0] * a, [False] * a, 0
    for slot in range(a):
        agent = inv.agents[slot]
        pos[agent] = inv.places[int(np.argmax(v[off : off + p]))]
        hungry[agent] = bool(v[off + p] > 0.5)
        off += p + 2
    loc: list[tuple[int, int]] = [(AT, 0)] * len(world.objects)
    for slot in range(len(world.objects)):
        k = int(np.argmax(v[off : off + 2 * p + a + 1]))
        if k < p:
            where = (AT, inv.places[k])
        elif k < p + a:
            where = (HELD, inv.agents[k - p])
        elif k < 2 * p + a:
            where = (BROKEN, inv.places[k - p - a])
        else:
            where = (GONE, -1)
        loc[inv.objects[slot]] = where
        off += 2 * p + a + 1
    return State(tuple(pos), tuple(hungry), tuple(loc))
