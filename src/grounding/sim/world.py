"""A small world of agents, objects and places.

The world is a transition system. ``transitions(world, state)`` lists every
possible (event, next state) outcome with its exact probability, as a
``Fraction``. Random stepping only samples from that list, so the symmetry
search in ``symmetry.py`` and the simulator can never disagree.

Each step, one agent chosen uniformly at random acts. It picks uniformly
among its legal actions, weighted by any habits it has. A fixed turn order
would make agents distinguishable, so there is none.

Rules:

- An agent can move to a neighbouring place, take an object at its place if
  its hands are empty, drop what it holds, eat if hungry, or wait.
- A fragile object breaks when dropped. Broken objects can't be taken.
- A hungry agent can eat at a feeding place, or eat an edible object it
  holds, which uses the object up.
- After acting without eating, an agent that isn't hungry becomes hungry
  with probability ``world.hunger``.
"""

from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property, lru_cache
from itertools import product
from typing import NamedTuple

import numpy as np

# Where an object is: (kind, index). The index is a place or an agent, or -1.
AT, HELD, BROKEN, GONE = range(4)


class State(NamedTuple):
    pos: tuple[int, ...]  # place of each agent
    hungry: tuple[bool, ...]  # per agent
    loc: tuple[tuple[int, int], ...]  # per object: (kind, index)


class Event(NamedTuple):
    verb: str  # wait, go, take, drop, eat
    agent: int
    obj: int | None = None
    place: int | None = None


@dataclass(frozen=True)
class Habit:
    """Agent ``agent`` is ``weight`` times as likely to choose moving to ``place``.

    Weights are exact fractions, so graded habits (1.1, 1.25, ...) stay exact.
    """

    agent: int
    place: int
    weight: Fraction

    def __post_init__(self) -> None:
        if isinstance(self.weight, float):
            raise TypeError("pass habit weights as Fraction, int or str ('5/4'), not float")
        weight = Fraction(self.weight)
        if weight <= 0:
            raise ValueError(f"habit weight must be positive, got {weight}")
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True)
class World:
    agents: tuple[str, ...]
    places: tuple[str, ...]
    edges: tuple[tuple[int, int], ...]  # undirected
    objects: tuple[str, ...]
    fragile: frozenset[int] = frozenset()  # objects
    edible: frozenset[int] = frozenset()  # objects
    feeding: frozenset[int] = frozenset()  # places
    hunger: Fraction = Fraction(1, 5)
    habits: tuple[Habit, ...] = ()

    @cached_property
    def neighbours(self) -> tuple[tuple[int, ...], ...]:
        adj: list[set[int]] = [set() for _ in self.places]
        for p, q in self.edges:
            adj[p].add(q)
            adj[q].add(p)
        return tuple(tuple(sorted(a)) for a in adj)


Kernel = dict[tuple[Event, State], Fraction]


def initial_states(world: World) -> list[State]:
    """Episode starts, sampled uniformly: any positions, objects intact, nobody hungry."""
    n_agents, n_objects, n_places = len(world.agents), len(world.objects), len(world.places)
    return [
        State(pos, (False,) * n_agents, tuple((AT, p) for p in places))
        for pos in product(range(n_places), repeat=n_agents)
        for places in product(range(n_places), repeat=n_objects)
    ]


def held_by(state: State, agent: int) -> int | None:
    for obj, loc in enumerate(state.loc):
        if loc == (HELD, agent):
            return obj
    return None


def actions(world: World, state: State, agent: int) -> list[tuple[Event, Fraction]]:
    """Legal actions for ``agent`` with their weights."""
    here = state.pos[agent]
    held = held_by(state, agent)
    one = Fraction(1)
    acts = [(Event("wait", agent), one)]
    for q in world.neighbours[here]:
        weight = one
        for h in world.habits:
            if h.agent == agent and h.place == q:
                weight *= h.weight
        acts.append((Event("go", agent, place=q), weight))
    if held is None:
        for obj, loc in enumerate(state.loc):
            if loc == (AT, here):
                acts.append((Event("take", agent, obj=obj), one))
    else:
        acts.append((Event("drop", agent, obj=held, place=here), one))
    if state.hungry[agent]:
        if here in world.feeding:
            acts.append((Event("eat", agent, place=here), one))
        if held in world.edible:
            acts.append((Event("eat", agent, obj=held), one))
    return acts


def apply(world: World, state: State, e: Event) -> State:
    """The deterministic effect of ``e``, before any change in hunger."""
    pos, hungry, loc = list(state.pos), list(state.hungry), list(state.loc)
    match e.verb:
        case "go":
            pos[e.agent] = e.place
        case "take":
            loc[e.obj] = (HELD, e.agent)
        case "drop":
            loc[e.obj] = (BROKEN if e.obj in world.fragile else AT, e.place)
        case "eat":
            hungry[e.agent] = False
            if e.obj is not None:
                loc[e.obj] = (GONE, -1)
    return State(tuple(pos), tuple(hungry), tuple(loc))


def _get_hungry(state: State, agent: int) -> State:
    hungry = list(state.hungry)
    hungry[agent] = True
    return state._replace(hungry=tuple(hungry))


def transitions(world: World, state: State) -> Kernel:
    """Every (event, next state) outcome from ``state``, with exact probabilities."""
    out: Kernel = {}

    def add(key: tuple[Event, State], p: Fraction) -> None:
        out[key] = out.get(key, Fraction(0)) + p

    n = len(world.agents)
    for agent in range(n):
        acts = actions(world, state, agent)
        total = sum(w for _, w in acts)
        for e, w in acts:
            p = w / (total * n)
            nxt = apply(world, state, e)
            if e.verb == "eat" or nxt.hungry[agent]:
                add((e, nxt), p)
            else:
                add((e, nxt), p * (1 - world.hunger))
                add((e, _get_hungry(nxt, agent)), p * world.hunger)
    return out


def explore(world: World) -> dict[State, Kernel]:
    """Every state reachable from an episode start, mapped to its transitions."""
    kernels: dict[State, Kernel] = {}
    frontier = initial_states(world)
    while frontier:
        nxt = []
        for s in frontier:
            if s in kernels:
                continue
            k = kernels[s] = transitions(world, s)
            nxt.extend(s2 for _, s2 in k if s2 not in kernels)
        frontier = nxt
    return kernels


@lru_cache(maxsize=64)
def _starts(world: World) -> tuple[State, ...]:
    return tuple(initial_states(world))


@lru_cache(maxsize=500_000)
def _outcomes(world: World, state: State) -> tuple[tuple[tuple[Event, State], ...], np.ndarray]:
    # Cached, because building corpora steps through the same states many times.
    items = list(transitions(world, state).items())
    probs = np.array([float(p) for _, p in items])
    return tuple(o for o, _ in items), probs / probs.sum()


def sample_initial(world: World, rng: np.random.Generator) -> State:
    starts = _starts(world)
    return starts[rng.integers(len(starts))]


def step(world: World, state: State, rng: np.random.Generator) -> tuple[Event, State]:
    """Sample one (event, next state) outcome from ``state``."""
    outcomes, probs = _outcomes(world, state)
    return outcomes[rng.choice(len(outcomes), p=probs)]


def state_to_array(state: State) -> np.ndarray:
    """A state as integers: agent places, agent hunger, then (kind, index) per object."""
    return np.array(
        [*state.pos, *map(int, state.hungry), *(x for loc in state.loc for x in loc)],
        dtype=np.int16,
    )


def state_from_array(world: World, a: np.ndarray) -> State:
    n, m = len(world.agents), len(world.objects)
    vals = [int(x) for x in a]
    loc = tuple((vals[2 * n + 2 * i], vals[2 * n + 2 * i + 1]) for i in range(m))
    return State(tuple(vals[:n]), tuple(bool(x) for x in vals[n : 2 * n]), loc)
