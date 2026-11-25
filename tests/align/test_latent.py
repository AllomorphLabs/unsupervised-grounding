"""The summary model learned from the two streams' distributions alone (align.latent)."""

import torch

from grounding.align.decipher import Gauss
from grounding.align.latent import fit_latent, unit_prior


def synthetic(seed=0, n=4000):
    g = torch.Generator().manual_seed(seed)
    views = torch.tensor(
        [[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], dtype=torch.float32
    )
    prior = torch.rand(len(views), generator=g) + 0.2
    prior /= prior.sum()
    W = torch.randn(3, 4, generator=g) * 2
    bias = torch.randn(4, generator=g)
    idx = torch.multinomial(prior, n, replacement=True, generator=g)
    z = views[idx] @ W + bias + 0.3 * torch.randn(n, 4, generator=g)
    return z, views, prior, W, bias


def test_unit_prior_counts_distinct_views():
    u = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    views, prior = unit_prior(u)
    assert len(views) == 2 and torch.allclose(prior.sort().values, torch.tensor([0.25, 0.75]))


def test_refines_a_nudged_start_to_the_truth():
    z, views, prior, W, bias = synthetic()
    start = Gauss(torch.cat([W + 0.5, bias[None]]), torch.eye(4))
    lf = fit_latent(z, views, prior, iterations=300, restarts=2, init=start, jitter=0.05)
    assert (lf.gauss.Wb[:3] - W).abs().mean() < 0.1


def test_no_effect_is_a_trap_without_a_nudge():
    """A start in which one fact has no effect stays there, even nudged (entry 69)."""
    z, views, prior, W, bias = synthetic()
    Wb = torch.cat([W, bias[None]]).clone()
    Wb[2] = 0  # the third fact starts with no effect on the summaries
    start = Gauss(Wb, torch.eye(4))
    stuck = fit_latent(z, views, prior, iterations=300, restarts=1, init=start, jitter=0.0)
    nudged = fit_latent(z, views, prior, iterations=300, restarts=1, init=start, jitter=0.05)
    # Even nudged, the fact stays near "no effect": it is a local optimum, not a balance point.
    assert stuck.gauss.Wb[2].abs().mean() < 0.2 * W[2].abs().mean()
    assert nudged.gauss.Wb[2].abs().mean() < 0.2 * W[2].abs().mean()


def test_split_moves_escape_no_effect():
    from grounding.align.latent import split_moves

    z, views, prior, W, bias = synthetic()
    Wb = torch.cat([W, bias[None]]).clone()
    Wb[2] = 0
    out = split_moves(z, views, prior, Gauss(Wb, torch.eye(4)), dims=[2], log=lambda s: None)
    assert (out.gauss.Wb[2] - W[2]).abs().mean() < 0.2


def test_joint_split_moves_place_two_exclusive_appearances():
    """One thing drawn in one of two colours: both appearance facts start with no effect, and a
    joint move places them (entry 72). Only their difference is identified."""
    from grounding.align.latent import split_moves

    g = torch.Generator().manual_seed(3)
    views = torch.tensor([[a, b, 1 - b] for a in (0, 1) for b in (0, 1)], dtype=torch.float32)
    prior = torch.tensor([0.3, 0.2, 0.15, 0.35])
    W = torch.randn(3, 4, generator=g) * 2
    idx = torch.multinomial(prior, 4000, replacement=True, generator=g)
    z = views[idx] @ W + 0.3 * torch.randn(4000, 4, generator=g)
    Wb = torch.cat([W, z.mean(0, keepdim=True)]).clone()
    Wb[1:3] = 0
    out = split_moves(
        z, views, prior, Gauss(Wb, torch.eye(4)), dims=[1, 2], together=[(1, 2)], log=lambda s: None
    )
    got, want = out.gauss.Wb[1] - out.gauss.Wb[2], W[1] - W[2]
    assert (got - want).abs().mean() < 0.2


def test_more_orders_never_fit_worse():
    from grounding.align.latent import split_moves

    z, views, prior, W, bias = synthetic(seed=1)
    Wb = torch.cat([W, bias[None]]).clone()
    Wb[1:] = 0
    Wb[-1] = z.mean(0)
    start = Gauss(Wb, torch.eye(4))
    one = split_moves(z, views, prior, start, dims=[1, 2], log=lambda s: None)
    three = split_moves(z, views, prior, start, dims=[1, 2], orders=3, log=lambda s: None)
    assert three.loglik >= one.loglik - 1e-6
