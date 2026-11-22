"""How often the data visits each state.

A corpus is built from episodes. Each episode starts at a state drawn
uniformly from ``initial_states`` and takes ``L`` steps, with ``L`` drawn from
an episode-length distribution. Every step is taken from some state, and that
step is what a stream renders. The visit weight of a state is the expected
share of all steps taken from it:

    w(s) ∝ sum over t >= 0 of P(L > t) · P(state at step t = s)

This is computed exactly by pushing the start distribution through the
transition tables, step by step, in floating point. Exact fractions are kept
for the symmetry tests in ``symmetry.py``, where equality matters; here it
doesn't.
"""

from collections.abc import Mapping

import numpy as np

from grounding.sim.world import Kernel, State, World, initial_states

Weights = dict[State, float]


def check_lengths(lengths: Mapping[int, float]) -> None:
    """Validate an episode-length distribution: positive lengths, probabilities summing to 1."""
    if not lengths:
        raise ValueError("episode-length distribution is empty")
    if any(not isinstance(n, int) or n < 1 for n in lengths):
        raise ValueError("episode lengths must be positive integers")
    if any(p < 0 for p in lengths.values()):
        raise ValueError("episode-length probabilities must be non-negative")
    if not np.isclose(sum(lengths.values()), 1.0):
        raise ValueError("episode-length probabilities must sum to 1")


def visit_weights(
    world: World, kernels: dict[State, Kernel], lengths: Mapping[int, float]
) -> Weights:
    """The share of all steps taken from each reachable state.

    ``lengths`` maps each episode length (number of steps) to its probability.
    ``kernels`` must be the output of ``explore(world)``.
    """
    check_lengths(lengths)
    states = list(kernels)
    index = {s: i for i, s in enumerate(states)}

    # Transition tables as flat arrays, merging events that reach the same state.
    src, dst, prob = [], [], []
    for i, s in enumerate(states):
        merged: dict[int, float] = {}
        for (_, s2), p in kernels[s].items():
            j = index[s2]
            merged[j] = merged.get(j, 0.0) + float(p)
        for j, p in merged.items():
            src.append(i)
            dst.append(j)
            prob.append(p)
    src_a, dst_a, prob_a = np.array(src), np.array(dst), np.array(prob)

    n = len(states)
    dist = np.zeros(n)
    starts = initial_states(world)
    for s in starts:
        dist[index[s]] += 1 / len(starts)

    # survival[t] = P(L > t)
    longest = max(lengths)
    pmf = np.zeros(longest + 1)
    for length, p in lengths.items():
        pmf[length] = p
    survival = 1 - np.cumsum(pmf)

    total = np.zeros(n)
    for t in range(longest):
        total += survival[t] * dist
        dist = np.bincount(dst_a, weights=dist[src_a] * prob_a, minlength=n)
    total /= total.sum()
    return dict(zip(states, total.tolist(), strict=True))


def uniform_weights(kernels: dict[State, Kernel]) -> Weights:
    """Every reachable state counted equally. A sensitivity check, not the main measure."""
    return dict.fromkeys(kernels, 1 / len(kernels))
