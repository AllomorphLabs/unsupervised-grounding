from fractions import Fraction

import numpy as np
import pytest

from grounding.sim.world import (
    AT,
    BROKEN,
    GONE,
    HELD,
    Event,
    Habit,
    State,
    actions,
    explore,
    initial_states,
    step,
    transitions,
)
from grounding.sim.worlds import APPLE, BUSH, CAVE, GLASS, RIVER, STONE, pilot


@pytest.fixture(scope="module")
def world():
    return pilot()


@pytest.fixture(scope="module")
def kernels(world):
    return explore(world)


def start(**kw) -> State:
    s = State(pos=(RIVER, CAVE), hungry=(False, False), loc=((AT, RIVER), (AT, RIVER), (AT, BUSH)))
    return s._replace(**kw)


def test_every_kernel_is_a_distribution(kernels):
    for k in kernels.values():
        assert sum(k.values()) == 1
        assert all(p > 0 for p in k.values())


def test_initial_states_are_all_reachable(world, kernels):
    assert set(initial_states(world)) <= kernels.keys()


def test_glass_breaks_when_dropped(world):
    s = start(loc=((HELD, 0), (AT, RIVER), (AT, BUSH)))
    outcomes = {e: s2 for (e, s2) in transitions(world, s)}
    assert outcomes[Event("drop", 0, obj=GLASS, place=RIVER)].loc[GLASS] == (BROKEN, RIVER)


def test_stone_survives_being_dropped(world):
    s = start(loc=((AT, RIVER), (HELD, 0), (AT, BUSH)))
    outcomes = {e: s2 for (e, s2) in transitions(world, s)}
    assert outcomes[Event("drop", 0, obj=STONE, place=RIVER)].loc[STONE] == (AT, RIVER)


def test_broken_objects_cannot_be_taken(world):
    s = start(loc=((BROKEN, RIVER), (AT, CAVE), (AT, BUSH)))
    assert not any(e.verb == "take" for e, _ in actions(world, s, 0))


def test_eating_the_apple_uses_it_up(world):
    s = start(hungry=(True, False), loc=((AT, CAVE), (AT, CAVE), (HELD, 0)))
    outcomes = {e: s2 for (e, s2) in transitions(world, s)}
    after = outcomes[Event("eat", 0, obj=APPLE)]
    assert after.loc[APPLE] == (GONE, -1)
    assert after.hungry[0] is False


def test_eating_needs_hunger_and_food(world):
    at_bush = start(pos=(BUSH, CAVE))
    assert not any(e.verb == "eat" for e, _ in actions(world, at_bush, 0))
    hungry_at_bush = at_bush._replace(hungry=(True, False))
    assert any(e.verb == "eat" for e, _ in actions(world, hungry_at_bush, 0))


def test_hunger_arrives_at_the_set_rate(world):
    k = transitions(world, start())
    wait = {s2.hungry: p for (e, s2), p in k.items() if e == Event("wait", 0)}
    assert wait[(True, False)] / (wait[(True, False)] + wait[(False, False)]) == Fraction(1, 5)


def test_habit_raises_the_preference(world):
    s = start(pos=(BUSH, CAVE))
    weights = {e.place: w for e, w in actions(pilot(habit=3), s, 0) if e.verb == "go"}
    assert weights == {RIVER: 3, CAVE: 1}


def test_graded_habits_stay_exact():
    s = start(pos=(BUSH, CAVE))
    weights = {e.place: w for e, w in actions(pilot(habit="5/4"), s, 0) if e.verb == "go"}
    assert weights == {RIVER: Fraction(5, 4), CAVE: 1}


def test_float_habits_are_rejected():
    with pytest.raises(TypeError):
        Habit(agent=0, place=RIVER, weight=1.25)


def test_step_samples_a_listed_outcome_reproducibly(world):
    s = start()
    a = step(world, s, np.random.default_rng(0))
    b = step(world, s, np.random.default_rng(0))
    assert a == b
    assert a in transitions(world, s)
