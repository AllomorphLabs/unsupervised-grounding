"""Symmetries of a world, found by exhaustive search.

A relabelling permutes agents among agents, objects among objects and places
among places. It is an automorphism if it commutes with the dynamics: for every
reachable state, relabelling and then stepping gives exactly the same
(event, next state) probabilities as stepping and then relabelling. It must
also map the set of episode starts onto itself, or the data it produces would
not be symmetric.

A near-symmetry almost commutes. Compare the real world's transitions at a
state ``s`` with those of the relabelled world, ``r(K(r⁻¹(s)))``: what the data
would look like if the entities were swapped. Two measures summarise the gap,
each averaged over states with explicit weights (see ``visits.py``):

- ``violation_rate`` (the main measure): mean total variation distance. Total
  variation is the probability that the two steps differ when coupled as
  closely as possible, so this is the expected share of steps on which the
  real and relabelled worlds disagree.
- ``evidence_rate`` (secondary): mean KL divergence, in nats, from the real to
  the relabelled transitions. This is how much evidence each step gives for
  telling the two labellings apart, and is infinite if the real world can do
  something the relabelled one never could.
"""

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations, product

from grounding.sim.world import (
    AT,
    BROKEN,
    HELD,
    Event,
    Kernel,
    State,
    World,
    initial_states,
    transitions,
)
from grounding.sim.world import explore as explore_world


@dataclass(frozen=True)
class Relabel:
    """Maps agent ``i`` to ``agents[i]``, and likewise for objects and places."""

    agents: tuple[int, ...]
    objects: tuple[int, ...]
    places: tuple[int, ...]

    @property
    def is_identity(self) -> bool:
        return all(
            i == j for perm in (self.agents, self.objects, self.places) for i, j in enumerate(perm)
        )

    def inverse(self) -> Relabel:
        def inv(perm: tuple[int, ...]) -> tuple[int, ...]:
            out = [0] * len(perm)
            for i, j in enumerate(perm):
                out[j] = i
            return tuple(out)

        return Relabel(inv(self.agents), inv(self.objects), inv(self.places))

    def _loc(self, kind: int, idx: int) -> tuple[int, int]:
        if kind in (AT, BROKEN):
            return kind, self.places[idx]
        if kind == HELD:
            return kind, self.agents[idx]
        return kind, idx

    def state(self, s: State) -> State:
        pos = [0] * len(s.pos)
        hungry = [False] * len(s.hungry)
        for a, (p, h) in enumerate(zip(s.pos, s.hungry, strict=True)):
            pos[self.agents[a]] = self.places[p]
            hungry[self.agents[a]] = h
        loc: list[tuple[int, int]] = [(0, 0)] * len(s.loc)
        for o, (kind, idx) in enumerate(s.loc):
            loc[self.objects[o]] = self._loc(kind, idx)
        return State(tuple(pos), tuple(hungry), tuple(loc))

    def event(self, e: Event) -> Event:
        return Event(
            e.verb,
            self.agents[e.agent],
            None if e.obj is None else self.objects[e.obj],
            None if e.place is None else self.places[e.place],
        )

    def kernel(self, k: Kernel) -> Kernel:
        return {(self.event(e), self.state(s)): p for (e, s), p in k.items()}

    def describe(self, world: World) -> str:
        """The entities this relabelling moves, e.g. ``agents A->B, B->A``."""
        parts = []
        for kind, names, perm in (
            ("agents", world.agents, self.agents),
            ("objects", world.objects, self.objects),
            ("places", world.places, self.places),
        ):
            moved = [f"{names[i]}->{names[j]}" for i, j in enumerate(perm) if i != j]
            if moved:
                parts.append(f"{kind} " + ", ".join(moved))
        return "; ".join(parts) or "identity"


def candidates(world: World) -> Iterator[Relabel]:
    """Every relabelling that keeps agents, objects and places within their own type."""
    for a, o, p in product(
        permutations(range(len(world.agents))),
        permutations(range(len(world.objects))),
        permutations(range(len(world.places))),
    ):
        yield Relabel(a, o, p)


def _kernel_at(world: World, kernels: dict[State, Kernel], s: State) -> Kernel:
    k = kernels.get(s)
    return k if k is not None else transitions(world, s)


def is_automorphism(world: World, r: Relabel, kernels: dict[State, Kernel]) -> bool:
    starts = initial_states(world)
    if {r.state(s) for s in starts} != set(starts):
        return False
    return all(_kernel_at(world, kernels, r.state(s)) == r.kernel(k) for s, k in kernels.items())


def automorphisms(world: World, kernels: dict[State, Kernel] | None = None) -> list[Relabel]:
    """The world's symmetry group, identity first."""
    kernels = explore_world(world) if kernels is None else kernels
    return [r for r in candidates(world) if is_automorphism(world, r, kernels)]


def total_variation(p: Kernel, q: Kernel) -> Fraction:
    return (
        sum(
            (abs(p.get(key, Fraction(0)) - q.get(key, Fraction(0))) for key in p.keys() | q.keys()),
            Fraction(0),
        )
        / 2
    )


def kl_divergence(p: Kernel, q: Kernel) -> float:
    """KL(p || q) in nats; infinite if p puts weight where q puts none."""
    total = 0.0
    for key, pk in p.items():
        qk = q.get(key)
        if qk is None:
            return math.inf
        total += float(pk) * math.log(pk / qk)
    return total


def relabelled_kernels(
    world: World, r: Relabel, kernels: dict[State, Kernel]
) -> Iterator[tuple[State, Kernel, Kernel]]:
    """For each reachable state: (state, real transitions, relabelled-world transitions)."""
    inv = r.inverse()
    for s, k in kernels.items():
        yield s, k, r.kernel(_kernel_at(world, kernels, inv.state(s)))


def _check_weights(weights: Mapping[State, float], kernels: dict[State, Kernel]) -> None:
    if weights.keys() != kernels.keys():
        raise ValueError("weights must cover exactly the reachable states")
    if not math.isclose(sum(weights.values()), 1.0):
        raise ValueError("weights must sum to 1")


def violation_rate(
    world: World, r: Relabel, kernels: dict[State, Kernel], weights: Mapping[State, float]
) -> float:
    """Main measure: weighted mean total variation between the real and relabelled worlds.

    ``weights`` is required, so the averaging is always a deliberate choice:
    ``visits.visit_weights`` for the main analysis, ``visits.uniform_weights``
    as a sensitivity check.
    """
    _check_weights(weights, kernels)
    return sum(
        weights[s] * float(total_variation(k, rk))
        for s, k, rk in relabelled_kernels(world, r, kernels)
    )


def evidence_rate(
    world: World, r: Relabel, kernels: dict[State, Kernel], weights: Mapping[State, float]
) -> float:
    """Secondary measure: weighted mean KL divergence from the real to the relabelled world."""
    _check_weights(weights, kernels)
    total = 0.0
    for s, k, rk in relabelled_kernels(world, r, kernels):
        if weights[s] > 0:
            total += weights[s] * kl_divergence(k, rk)
    return total
