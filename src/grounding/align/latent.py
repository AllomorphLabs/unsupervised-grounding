"""Learning the summary model from the two streams' own distributions, with no pairs.

Decipherment's summary model says how the language model's sentence summaries z depend on the
units u of the frame a sentence describes: z ~ N(W u + b, Sigma). It is normally fitted to the fit's
own guessed paths, and if those start wrong for some part of the state (hunger, entry 63), the
model never learns that part.

This fits it another way, using no paths at all:

- The frame stream says how often each combination of units occurs: ``prior``, the share of
  training frames showing each distinct unit view.
- Each summary is assumed to come from one such combination, weighted by that share, so the
  summaries as a whole are a mixture of Gaussians whose components are the frame stream's
  combinations, with means W u + b.
- Expectation-maximisation fits W, b and Sigma to the summaries alone.

Nothing pairs a sentence with a frame. The result is identified only up to relabellings that
leave the frame stream's combination frequencies unchanged (such as swapping the agents), which
the words and the dynamics then settle.
"""

import itertools
from dataclasses import dataclass

import numpy as np
import torch

from grounding.align.decipher import Gauss


@dataclass
class LatentFit:
    gauss: Gauss
    loglik: float  # mean log-likelihood per summary, up to a constant
    iterations: int


def unit_prior(units: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Distinct unit views among ``units`` (frames x units) and the share of each."""
    views, counts = torch.unique(units.round(), dim=0, return_counts=True)
    return views, counts.float() / counts.sum()


def _loglik(
    z: torch.Tensor,
    views: torch.Tensor,
    logp: torch.Tensor,
    g: Gauss,
    chunk: int,
    temp: float = 1.0,
):
    """Per-summary log-likelihood under the mixture, and the responsibilities' sufficient
    statistics (summed per view)."""
    M = g.mean(views)  # K x D
    MA = M @ g.A  # K x D
    m2 = (MA * MA).sum(1)
    total = 0.0
    R_sum = torch.zeros(len(views), device=z.device)  # sum_n r_nk
    RZ = torch.zeros(len(views), z.shape[1], device=z.device)  # sum_n r_nk z_n
    for i in range(0, len(z), chunk):
        zc = z[i : i + chunk]
        ZA = zc @ g.A
        d2 = (ZA * ZA).sum(1, keepdim=True) - 2 * ZA @ MA.T + m2[None]
        lp = logp[None] - 0.5 * d2 + g.norm
        ll = torch.logsumexp(lp, 1)
        total += float(ll.sum())
        # Annealing: responsibilities from a flattened likelihood while temp > 1.
        lpt = logp[None] - 0.5 * d2 / temp
        r = torch.exp(lpt - torch.logsumexp(lpt, 1, keepdim=True))
        R_sum += r.sum(0)
        RZ += r.T @ zc
    return total / len(z), R_sum, RZ


def fit_latent(
    z: torch.Tensor,
    views: torch.Tensor,
    prior: torch.Tensor,
    iterations: int = 60,
    restarts: int = 4,
    reg: float = 1e-2,
    seed: int = 0,
    chunk: int = 2048,
    pairs: torch.Tensor | None = None,
    temp0: float = 1.0,
    anneal: float = 0.9,
    init: Gauss | None = None,
    jitter: float = 0.05,
) -> LatentFit:
    """Fit z ~ sum_k prior_k N(W phi(view_k) + b, Sigma) by EM; best of ``restarts`` starts.

    Deterministic annealing: the responsibilities start from a likelihood flattened by a
    temperature ``temp0``, which shrinks by ``anneal`` each iteration to 1, so the fit settles on
    the broad structure before the detail (plain EM stalls in poor local optima here)."""
    dev = z.device
    logp = torch.log(prior.clamp(min=1e-12))
    best = None
    gen = torch.Generator(device="cpu").manual_seed(seed)
    D = z.shape[1]
    for _ in range(restarts):
        P = Gauss.features(views, pairs)
        if init is None:
            # Start: a random W, the data's mean and covariance.
            Wb = torch.randn(P.shape[1], D, generator=gen).to(dev) * 0.3
            Wb[-1] = z.mean(0)
            Sig = torch.cov(z.T) + 1e-3 * torch.eye(D, device=dev)
            g = Gauss(Wb, torch.linalg.inv(torch.linalg.cholesky(Sig)).T, pairs)
        else:
            # Start from a given model, nudged a little so no part of it sits exactly at "no
            # effect", from which EM can't move (experiment log, entry 63).
            Wb = init.Wb.clone() + jitter * torch.randn(init.Wb.shape, generator=gen).to(dev)
            g = Gauss(Wb, init.A.clone(), init.pairs)
        ll = -np.inf
        temp = temp0
        it = 0
        for it in range(iterations):  # noqa: B007
            ll_new, R, RZ = _loglik(z, views, logp, g, chunk, temp)
            # M-step: weighted ridge regression of z on phi(view), with weights R per view.
            XtX = P.T @ (P * R[:, None])
            XtY = P.T @ RZ
            eye = torch.eye(XtX.shape[0], device=dev)
            Wb = torch.linalg.solve(XtX + reg * eye, XtY)
            # Residual covariance: sum_n sum_k r_nk (z_n - m_k)(z_n - m_k)^T / N
            M = P @ Wb
            zz = z.T @ z
            cross = RZ.T @ M
            mm = M.T @ (M * R[:, None])
            Sig = (zz - cross - cross.T + mm) / len(z)
            Sig = (Sig + Sig.T) / 2 + 1e-3 * torch.eye(D, device=dev)
            g = Gauss(Wb, torch.linalg.inv(torch.linalg.cholesky(Sig)).T, pairs)
            if temp <= 1.0 and abs(ll_new - ll) < 1e-5:
                ll = ll_new
                break
            ll = ll_new
            temp = max(1.0, temp * anneal)
        if best is None or ll > best.loglik:
            best = LatentFit(g, ll, it + 1)
    return best


def prune(views: torch.Tensor, prior: torch.Tensor, mass: float = 0.999):
    """The most common views covering ``mass`` of the frames, renormalised (for speed)."""
    order = torch.argsort(prior, descending=True)
    keep = order[: int((torch.cumsum(prior[order], 0) < mass).sum()) + 1]
    return views[keep], prior[keep] / prior[keep].sum()


def split_moves(
    z: torch.Tensor,
    views: torch.Tensor,
    prior: torch.Tensor,
    start: Gauss,
    dims: list[int] | None = None,
    weakest: int = 0,
    directions: int = 4,
    iterations: int = 60,
    weak: float = 0.25,
    rounds: int = 1,
    together: list[tuple[int, int]] | None = None,
    orders: int = 1,
    log=print,
) -> LatentFit:
    """Escape "no effect" for facts whose effect on the summaries is near zero.

    EM can't move a fact's coefficients away from zero: the noise term absorbs that fact's
    variation, and every small step makes the fit worse first (tests/align/test_latent.py). So for
    each *weak* fact (coefficient norm below ``weak`` times the median), try restarting its
    coefficients along each of the main directions of the left-over noise (both signs), refit, and
    keep the restart if the likelihood improves. Facts are tried in turn, each from the best fit so
    far.

    ``together``: pairs of facts that may only be placed jointly, such as one thing's two
    appearances (entry 72): after the single moves, each pair whose facts are both still weak is
    restarted along two different noise directions at once.

    ``orders``: which facts get placed can depend on the order they are tried in (entry 74), so
    the whole procedure is repeated with the weak facts in that many orders (as found, reversed,
    then shuffled), keeping the fit of highest likelihood."""
    best = fit_latent(z, views, prior, iterations=iterations, restarts=1, init=start, jitter=0.0)
    log(f"  split moves: start loglik {best.loglik:.4f}")
    norms = best.gauss.Wb[:-1].norm(dim=1)
    log("  effect sizes: " + " ".join(f"{float(x):.2f}" for x in norms))
    if dims is None and weakest:
        dims = [int(j) for j in torch.argsort(norms)[:weakest]]
    if dims is None:
        live = norms[norms > 0]
        cut = weak * float(live.median()) if len(live) else 0.0
        dims = [j for j in range(len(norms)) if float(norms[j]) < cut]
    log(f"  weak facts: {dims}")
    first = best
    gen = torch.Generator().manual_seed(0)
    for o in range(orders):
        if o == 0:
            order = list(dims)
        elif o == 1:
            order = list(reversed(dims))
        else:
            order = [dims[int(i)] for i in torch.randperm(len(dims), generator=gen)]
        if orders > 1:
            log(f"  order {o + 1}/{orders}: {order}")
        fit = _rounds(
            z, views, prior, first, norms, order, directions, iterations, rounds, together, log
        )
        if o == 0 or fit.loglik > best.loglik:
            best = fit
    if orders > 1:
        log(f"  best over orders: {best.loglik:.4f}")
    return best


def _rounds(z, views, prior, best, norms, dims, directions, iterations, rounds, together, log):
    """Up to ``rounds`` passes of split moves over ``dims`` (then joint moves on ``together``),
    stopping when a pass gains nothing."""
    for r in range(rounds):
        before = best.loglik
        cut = max((float(norms[j]) for j in dims), default=0.0) + 1e-6
        best = _split_round(z, views, prior, best, dims, directions, iterations, log)
        if together:
            # Pairs whose facts both stayed as weak as the weak facts were before this round.
            now = best.gauss.Wb[:-1].norm(dim=1)
            pairs = [
                (a, b)
                for a, b in together
                if a in dims and b in dims and max(float(now[a]), float(now[b])) <= cut
            ]
            best = _joint_round(z, views, prior, best, pairs, min(directions, 3), iterations, log)
        norms = best.gauss.Wb[:-1].norm(dim=1)
        if best.loglik <= before + 1e-4:
            break
        if r + 1 < rounds:
            log(f"  round {r + 2}")
    return best


def _noise_directions(g: Gauss, n: int) -> torch.Tensor:
    """The ``n`` main directions of the left-over noise, largest first, scaled by their spread."""
    vals, vecs = torch.linalg.eigh(torch.linalg.inv(g.A @ g.A.T))
    return (vecs[:, -n:].T * vals[-n:, None].sqrt()).flip(0)


def _try(z, views, prior, best: LatentFit, moves, iterations) -> LatentFit | None:
    """Restart the facts in ``moves`` ({fact: effect}) and refit; the result if better."""
    Wb = best.gauss.Wb.clone()
    for j, w in moves.items():
        Wb[j] = w
        # While the fact had no effect, the bias absorbed its average contribution.
        Wb[-1] = Wb[-1] - (prior @ views[:, j]) * w
    trial = fit_latent(
        z,
        views,
        prior,
        iterations=iterations,
        restarts=1,
        init=Gauss(Wb, best.gauss.A, best.gauss.pairs),
        jitter=0.0,
    )
    ok = np.isfinite(trial.loglik) and bool(
        torch.isfinite(trial.gauss.Wb).all() and torch.isfinite(trial.gauss.A).all()
    )
    return trial if ok and trial.loglik > best.loglik + 1e-4 else None


def _split_round(z, views, prior, best: LatentFit, dims, directions, iterations, log) -> LatentFit:
    """One pass of split moves over ``dims``, each from the best fit so far."""
    for j in dims:
        improved = False
        for v in _noise_directions(best.gauss, directions):
            for sign in (1.0, -1.0):
                trial = _try(z, views, prior, best, {j: sign * 2 * v}, iterations)
                if trial is not None:
                    best, improved = trial, True
        log(f"  fact {j}: {'improved to ' + format(best.loglik, '.4f') if improved else 'no gain'}")
    return best


def _joint_round(z, views, prior, best: LatentFit, pairs, directions, iterations, log):
    """Joint split moves: each pair of facts restarted along two different noise directions,
    every order and sign, from the best fit so far."""
    for a, b in pairs:
        improved = False
        top = _noise_directions(best.gauss, directions)
        for i, k in itertools.permutations(range(len(top)), 2):
            for sa, sb in itertools.product((1.0, -1.0), repeat=2):
                moves = {a: sa * 2 * top[i], b: sb * 2 * top[k]}
                trial = _try(z, views, prior, best, moves, iterations)
                if trial is not None:
                    best, improved = trial, True
        log(
            f"  facts {a}+{b} together: "
            + ("improved to " + format(best.loglik, ".4f") if improved else "no gain")
        )
    return best
