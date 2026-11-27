"""Symmetry search on the full pilot world; the violation measures on the tiny world, for speed."""

import math
from functools import cache

import pytest

from grounding.sim.symmetry import (
    Relabel,
    automorphisms,
    candidates,
    evidence_rate,
    violation_rate,
)
from grounding.sim.visits import uniform_weights, visit_weights
from grounding.sim.world import explore
from grounding.sim.worlds import pilot, tiny

# Pilot world: 2 agents, 3 objects, 4 places.
IDENTITY = Relabel((0, 1), (0, 1, 2), (0, 1, 2, 3))
AGENT_SWAP = Relabel((1, 0), (0, 1, 2), (0, 1, 2, 3))
MIRROR = Relabel((0, 1), (0, 1, 2), (3, 2, 1, 0))

# Tiny world: 2 agents, 2 objects, 3 places.
TINY_SWAP = Relabel((1, 0), (0, 1), (0, 1, 2))
TINY_GLASS_STONE = Relabel((0, 1), (1, 0), (0, 1, 2))
LENGTHS = {n: 1 / 10 for n in range(1, 11)}


@cache
def tiny_setup(habit=None):
    world = tiny(habit=habit)
    kernels = explore(world)
    return world, kernels, visit_weights(world, kernels, LENGTHS)


@pytest.fixture(scope="module")
def world():
    return pilot()


@pytest.fixture(scope="module")
def kernels(world):
    return explore(world)


def test_candidates_stay_within_type(world):
    # 2! agents x 3! objects x 4! places
    assert len(list(candidates(world))) == 2 * 6 * 24


def test_relabelling_twice_with_a_swap_is_the_identity(kernels):
    for s in list(kernels)[:200]:
        assert AGENT_SWAP.state(AGENT_SWAP.state(s)) == s


def test_inverse_undoes_a_relabelling(kernels):
    r = Relabel((1, 0), (2, 0, 1), (1, 3, 0, 2))
    for s in list(kernels)[:200]:
        assert r.inverse().state(r.state(s)) == s


def test_pilot_group_is_exactly_the_agent_swap(world, kernels):
    assert automorphisms(world, kernels) == [IDENTITY, AGENT_SWAP]


def test_a_habit_breaks_the_agent_swap():
    assert automorphisms(pilot(habit=4)) == [IDENTITY]


def test_without_feeding_the_places_mirror_as_a_pair():
    # River swaps with hill and bush with cave, together, alongside the agent swap.
    group = automorphisms(pilot(feeding=False))
    assert set(group) == {IDENTITY, AGENT_SWAP, MIRROR, Relabel((1, 0), (0, 1, 2), (3, 2, 1, 0))}


def test_exact_symmetries_score_zero_on_both_measures():
    for habit in (None, 1):
        world, kernels, w = tiny_setup(habit)
        assert violation_rate(world, TINY_SWAP, kernels, w) == 0
        assert evidence_rate(world, TINY_SWAP, kernels, w) == 0


def test_both_measures_rise_with_habit_strength():
    scores = []
    for habit in ("5/4", 2, 4):
        world, kernels, w = tiny_setup(habit)
        scores.append(
            (
                violation_rate(world, TINY_SWAP, kernels, w),
                evidence_rate(world, TINY_SWAP, kernels, w),
            )
        )
    violations, evidence = zip(*scores, strict=True)
    assert 0 < violations[0] < violations[1] < violations[2]
    assert 0 < evidence[0] < evidence[1] < evidence[2] < math.inf


def test_uniform_weighting_gives_a_different_answer():
    # The two weightings disagree, which is why the choice is pre-registered.
    world, kernels, w = tiny_setup(4)
    by_visits = violation_rate(world, TINY_SWAP, kernels, w)
    by_states = violation_rate(world, TINY_SWAP, kernels, uniform_weights(kernels))
    assert by_visits != pytest.approx(by_states)


def test_a_rule_difference_is_infinitely_informative():
    # Dropping glass breaks it and dropping stone doesn't, so one step can tell them apart.
    world, kernels, w = tiny_setup()
    assert evidence_rate(world, TINY_GLASS_STONE, kernels, w) == math.inf
    assert 0 < violation_rate(world, TINY_GLASS_STONE, kernels, w) < 1


def test_weights_must_cover_the_reachable_states():
    world, kernels, _ = tiny_setup()
    with pytest.raises(ValueError):
        violation_rate(world, TINY_SWAP, kernels, {})


def test_lured_world_keeps_the_agent_swap_and_breaks_the_mirror_harder():
    from grounding.sim.worlds import lured

    mirror = Relabel((0, 1), (0, 1, 2), (3, 2, 1, 0))
    lengths = {n: 1 / 41 for n in range(20, 61)}
    rates = []
    for w in (pilot(), lured()):
        k = explore(w)
        group = automorphisms(w, k)
        assert len(group) == 2 and group[1].agents == (1, 0)
        rates.append(violation_rate(w, mirror, k, visit_weights(w, k, lengths)))
    assert rates[1] > 3 * rates[0]
