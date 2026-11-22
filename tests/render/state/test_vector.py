import numpy as np
import pytest

from grounding.render.state.vector import encode, frame_size
from grounding.sim.symmetry import Relabel
from grounding.sim.world import AT, BROKEN, GONE, HELD, State, explore
from grounding.sim.worlds import pilot


@pytest.fixture(scope="module")
def world():
    return pilot()


@pytest.fixture(scope="module")
def states(world):
    rng = np.random.default_rng(0)
    every = list(explore(world))
    return [every[i] for i in rng.choice(len(every), 300, replace=False)]


def identity(world) -> Relabel:
    return Relabel(*(tuple(range(len(x))) for x in (world.agents, world.objects, world.places)))


def decode(world, v: np.ndarray) -> tuple[State, int | None]:
    """Read a frame in the unshuffled layout back into a state and an actor."""
    a, p = len(world.agents), len(world.places)
    pos, hungry, actor, loc, off = [], [], None, [], 0
    for i in range(a):
        block = v[off : off + p + 2]
        pos.append(int(np.flatnonzero(block[:p])[0]))
        hungry.append(bool(block[p]))
        if block[p + 1]:
            actor = i
        off += p + 2
    for _ in world.objects:
        k = int(np.flatnonzero(v[off : off + 2 * p + a + 1])[0])
        if k < p:
            loc.append((AT, k))
        elif k < p + a:
            loc.append((HELD, k - p))
        elif k < 2 * p + a:
            loc.append((BROKEN, k - p - a))
        else:
            loc.append((GONE, -1))
        off += 2 * p + a + 1
    return State(tuple(pos), tuple(hungry), tuple(loc)), actor


def test_frame_size(world):
    assert frame_size(world) == 2 * (4 + 2) + 3 * (2 * 4 + 2 + 1) == 45


def test_frames_are_exactly_one_hot_per_block(world, states):
    for s in states:
        v = encode(world, s, 1, identity(world))
        assert set(np.unique(v)) <= {0, 1}
        # One place per agent and one location per object.
        assert v.sum() == len(world.agents) + len(world.objects) + sum(s.hungry) + 1


def test_a_frame_determines_the_state_and_actor(world, states):
    for s in states:
        for actor in (None, 0, 1):
            assert decode(world, encode(world, s, actor, identity(world))) == (s, actor)


def test_shuffled_frames_hold_the_relabelled_state(world, states):
    r = Relabel((1, 0), (2, 0, 1), (3, 1, 0, 2))
    for s in states:
        assert decode(world, encode(world, s, 0, r)) == (r.state(s), r.agents[0])


def test_relabelling_the_world_only_permutes_coordinates(world, states):
    # The renderer check for observations: every agent, object and place is laid out the
    # same way, so relabelling the world moves values between coordinates and changes
    # nothing else. Find the coordinate permutation from one state, then check it on all.
    r = Relabel((1, 0), (1, 2, 0), (2, 3, 1, 0))
    ident = identity(world)
    columns = np.stack([encode(world, s, a, ident) for s in states for a in (None, 0, 1)])
    moved = np.stack(
        [
            encode(world, r.state(s), None if a is None else r.agents[a], ident)
            for s in states
            for a in (None, 0, 1)
        ]
    )
    # Each moved coordinate must equal an original coordinate on every frame, using each
    # original once. Coordinates identical on every sample (e.g. "gone" for an object that
    # can't be eaten) are interchangeable, so take the first unused match.
    used: set[int] = set()
    for j in range(columns.shape[1]):
        matches = [
            i
            for i in range(columns.shape[1])
            if i not in used and np.array_equal(moved[:, j], columns[:, i])
        ]
        assert matches, f"coordinate {j} is not a relabelled copy of any coordinate"
        used.add(matches[0])
