import math
from fractions import Fraction

import numpy as np
import pytest

from grounding.sim.visits import check_lengths, uniform_weights, visit_weights
from grounding.sim.world import HELD, explore, initial_states, sample_initial, step
from grounding.sim.worlds import GLASS, RIVER, tiny


@pytest.fixture(scope="module")
def world():
    return tiny()


@pytest.fixture(scope="module")
def kernels(world):
    return explore(world)


def test_single_step_episodes_weight_only_the_starts(world, kernels):
    w = visit_weights(world, kernels, {1: 1.0})
    starts = set(initial_states(world))
    for s, ws in w.items():
        assert ws == pytest.approx(1 / len(starts) if s in starts else 0.0)


def test_two_step_episodes_match_a_direct_calculation(world, kernels):
    starts = initial_states(world)
    expected: dict = {}
    for s in starts:
        expected[s] = expected.get(s, Fraction(0)) + Fraction(1, 2 * len(starts))
        for (_, s2), p in kernels[s].items():
            expected[s2] = expected.get(s2, Fraction(0)) + p / (2 * len(starts))
    w = visit_weights(world, kernels, {2: 1.0})
    for s, ws in w.items():
        assert ws == pytest.approx(float(expected.get(s, 0)), abs=1e-12)


def test_weights_sum_to_one(world, kernels):
    assert math.isclose(sum(visit_weights(world, kernels, {3: 0.5, 7: 0.5}).values()), 1.0)
    assert math.isclose(sum(uniform_weights(kernels).values()), 1.0)


def test_weights_match_simulated_episodes(world, kernels):
    # Two summaries of where the data spends its steps, exact vs simulated.
    lengths = {1: 1 / 3, 2: 1 / 3, 3: 1 / 3}
    w = visit_weights(world, kernels, lengths)

    def summaries(s):
        return (s.pos[0] == RIVER, s.loc[GLASS][0] == HELD)

    exact = np.array([sum(ws for s, ws in w.items() if summaries(s)[i]) for i in range(2)])

    rng = np.random.default_rng(0)
    hits, steps = np.zeros(2), 0
    for _ in range(4000):
        s = sample_initial(world, rng)
        for _ in range(rng.choice([1, 2, 3])):
            hits += summaries(s)
            steps += 1
            _, s = step(world, s, rng)
    simulated = hits / steps
    stderr = np.sqrt(exact * (1 - exact) / steps)
    assert np.all(np.abs(simulated - exact) < 4 * stderr)


@pytest.mark.parametrize(
    "lengths",
    [{}, {0: 1.0}, {2.5: 1.0}, {3: 0.5}, {3: 1.5, 4: -0.5}],
)
def test_bad_length_distributions_are_rejected(lengths):
    with pytest.raises(ValueError):
        check_lengths(lengths)
