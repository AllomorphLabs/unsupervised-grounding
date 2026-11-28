"""Exact inference for each object, given the rest of each decoded frame (technical decision 30;
experiment log, entry 106).

The particle filter reads an episode forwards and weighs only the frames the world model samples,
so it loses track of an object lying unseen between being dropped and taken (at a drop it often
hands the object to the other agent). Here, given each episode's decoded frames, each object's
whole history is re-inferred by forward-backward over its states, holding the rest of each frame
fixed:

- an object is a thing that can be carried but never changes place on its own, counted from
  the frame stream alone;
- its states are the values it can be drawn in (a place's cell, or a carrier's cell, in its
  channel), and "absent" if the structure says it can be gone;
- at each sentence and for each pair of states (before, after), the evidence is the word tables'
  probability of the sentence, the learned world model's probability of the change given the
  frame before it (or the stay probability if nothing changed, as in the filter), the summary
  model's density, and whether the frame is valid.

The world model step uses the learned frame model, never transitions counted from the frame
stream, which are the lookup-table control. Grid video only.
"""

from __future__ import annotations

import math

import torch


def objects(D) -> list[int]:
    """The objects: things that can share a cell with another (be carried) but never change place
    on their own, counted from the frame stream (``GridDecipherer.moves_alone``). Agents move on
    their own; objects move only while carried."""
    carried = {k for u in D.units.units if u[0] == "with" for k in u[1:]}
    return [
        k
        for k in sorted(carried)
        if D.place_changes[k] > 0 and D.moves_alone[k] <= 0.01 * D.place_changes[k]
    ]


@torch.no_grad()
def mode_frames(fit, paths) -> list:
    """Each batch's decoded frames: per sentence, the commonest unit view among the particles,
    as one particle's frame with it (as ``GridFitMixin.decode``). [(batch, Be x Tm x d)]"""
    D = fit.D
    N = D.cfg.particles
    bits = 2 ** torch.arange(D.U, device=D.dev, dtype=torch.int64)
    out = []
    for b, traj, _ in paths:
        Tm, Be = b.real.shape
        flat = traj.reshape(-1, D.d).float()
        code = (D.view(flat).round().clamp(0, 1).long() * bits).sum(-1).reshape(Be, N, Tm)
        best = torch.mode(code, 1).values
        pick = (code == best[:, None]).float().argmax(1)
        tr = traj.reshape(Be, N, Tm, D.d)
        e_idx = torch.arange(Be, device=D.dev)[:, None]
        t_idx = torch.arange(Tm, device=D.dev)[None]
        out.append((b, tr[e_idx, pick, t_idx].float()))
    return out


def _candidates(D, F: torch.Tensor, k: int) -> torch.Tensor:
    """Be x Tm x S x d: each frame with object k in each of its states."""
    vals = list(D.units.things[k])
    gone = any(u[0] == "gone" and u[1] == k for u in D.units.units)
    S = len(vals) + (1 if gone else 0)
    base = F.clone()
    base[:, :, vals] = 0
    G = base[:, :, None].repeat(1, 1, S, 1)
    for s, v in enumerate(vals):
        G[:, :, s, v] = 1
    return G


@torch.no_grad()
def smooth(fit, b, F: torch.Tensor, k: int) -> torch.Tensor:
    """Frames ``F`` (Be x Tm x d) with object k's history re-inferred exactly."""
    D, cfg = fit.D, fit.D.cfg
    dev, d = D.dev, D.d
    Tm, Be = b.real.shape
    G = _candidates(D, F, k)
    S = G.shape[2]
    ok = D.valid(G.reshape(-1, d))
    if ok is None:
        ok = torch.ones(Be * Tm * S, dtype=torch.bool, device=dev)
    logok = torch.where(ok, 0.0, -1e9).reshape(Be, Tm, S)
    kappa = getattr(fit, "kappa_now", cfg.summary_power)
    focus = (fit.sig / fit.sig.sum()).tolist() if hasattr(fit, "sig") else None
    q = float(D.q)
    log_q, log_move = math.log(q), math.log(max(1 - q, 1e-12))
    lz = torch.zeros(Be, Tm, S, device=dev)
    if b.z is not None and fit.gauss is not None:
        for t in range(Tm):
            u = D.view(G[:, t].reshape(-1, d))
            lz[:, t] = kappa * fit.gauss.logp(u, b.z[t].repeat_interleave(S, 0)).reshape(Be, S)
    rows = torch.arange(Be, device=dev).repeat_interleave(S * S)
    eye = torch.where(torch.eye(S, dtype=torch.bool, device=dev), 0.0, -1e9)[None]
    alphas = [lz[:, 0] + logok[:, 0]]
    psis = []
    for t in range(1, Tm):
        P = G[:, t - 1][:, :, None].expand(Be, S, S, d).reshape(-1, d)
        C = G[:, t][:, None, :].expand(Be, S, S, d).reshape(-1, d)
        moved = (P != C).any(1)
        ew = fit.emit_words(
            b.tok[t], P, C, moved, D.actor_word(C, fit.places_t), rows=rows, focus=focus
        )
        h = D.fm.context(P[:, None])[:, 0]
        lt = torch.where(moved, log_move + D.fm.log_prob_next(h, C), torch.full_like(ew, log_q))
        lp = (torch.log(ew.clamp(min=1e-30)) + lt).reshape(Be, S, S)
        lp = lp + lz[:, t][:, None, :] + logok[:, t][:, None, :]
        lp = torch.where(b.real[t][:, None, None], lp, eye)
        psis.append(lp)
        alphas.append(torch.logsumexp(alphas[-1][:, :, None] + lp, 1))
    beta = torch.zeros(Be, S, device=dev)
    post = [None] * Tm
    post[-1] = alphas[-1]
    for t in range(Tm - 1, 0, -1):
        beta = torch.logsumexp(psis[t - 1] + beta[:, None, :], 2)
        post[t - 1] = alphas[t - 1] + beta
    pick = torch.stack(post, 1).argmax(-1)
    e_idx = torch.arange(Be, device=dev)[:, None]
    t_idx = torch.arange(Tm, device=dev)[None]
    return G[e_idx, t_idx, pick]


@torch.no_grad()
def exact_frames(fit, paths) -> list:
    """Each batch's decoded frames with every object's history re-inferred exactly."""
    out = mode_frames(fit, paths)
    for k in objects(fit.D):
        out = [(b, smooth(fit, b, F, k)) for b, F in out]
    return out


@torch.no_grad()
def exact_paths(fit, paths) -> list:
    """Filter paths replaced by each episode's exactly inferred path, given to every particle,
    so the update learns from it (``exact_objects_from``)."""
    N = fit.D.cfg.particles
    out = []
    for b, F in exact_frames(fit, paths):
        traj = F.to(torch.uint8).repeat_interleave(N, 0)
        moved = torch.zeros(traj.shape[:2], dtype=torch.bool, device=traj.device)
        moved[:, 1:] = (traj[:, 1:] != traj[:, :-1]).any(-1)
        out.append((b, traj, moved))
    return out
