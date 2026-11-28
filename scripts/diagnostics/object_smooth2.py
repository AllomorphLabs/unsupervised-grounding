"""Decoding with exact inference for each object (entry 106; uses hidden/ only to score).

The fit's decode gives every frame (agents read at 94-98%). Then, for each object (a thing two
others can carry, found from the parse), its whole history is re-inferred exactly, by forward-
backward over the object's states, holding the rest of each decoded frame fixed:

- an object's states are the values it is drawn in (a place's cell, or a carrier's cell, in its
  channel), and "absent" if the parse found it can be gone;
- for each sentence and each pair of states (before, after): the word tables' probability of the
  sentence, the learned world model's probability of the change (or the stay probability if
  nothing changed, as in the filter), the summary model's density, and whether the frame is valid.

The particle filter reads each episode forwards and samples what the world model proposes; this
weighs every state of the object at every sentence, and uses later sentences too.

    GRID_CFG=<config> uv run python scripts/diagnostics/object_smooth.py <facts pair> <checkpoint>
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from oracle_objective_lib import setup  # noqa: E402

from grounding.align.decipher import make_fit  # noqa: E402
from grounding.analysis.score import lang_truth, score_predictions  # noqa: E402
from grounding.render.grid.frames import decode  # noqa: E402
from grounding.sim.symmetry import automorphisms  # noqa: E402
from grounding.sim.world import state_to_array  # noqa: E402

pair, ck = Path(sys.argv[1]), Path(sys.argv[2])
S = setup(pair)
D, data, key, world, cfg = S["D"], S["data"], S["key"], S["world"], S["cfg"]
saved = torch.load(ck, map_location="cuda", weights_only=False)
fit = make_fit(D, saved["state"]["m"])
fit.__dict__.update(saved["state"])
dev, d, N = D.dev, D.d, cfg.particles
units = D.units.units
group = automorphisms(world)
kappa = getattr(fit, "kappa_now", cfg.summary_power)
focus = (fit.sig / fit.sig.sum()).tolist() if hasattr(fit, "sig") else None
q = float(D.q)

carriers = {}
for u in units:
    if u[0] == "with":
        for k in u[1:]:
            carriers.setdefault(k, set()).update(set(u[1:]) - {k})
objects = sorted(k for k, c in carriers.items() if len(c) == 2)
can_go = {u[1] for u in units if u[0] == "gone"}
print(f"SMOOTH objects {objects} (things with two carriers)", flush=True)


@torch.no_grad()
def decoded(batches):
    """The fit's decode, kept per batch: (batch, Be x Tm x d frames)."""
    bits = 2 ** torch.arange(D.U, device=dev, dtype=torch.int64)
    out = []
    for b, traj, _ in fit.run(batches, 1000 * cfg.seed)[0]:
        Tm, Be = b.real.shape
        flat = traj.reshape(-1, d).float()
        code = (D.view(flat).round().clamp(0, 1).long() * bits).sum(-1).reshape(Be, N, Tm)
        best = torch.mode(code, 1).values
        pick = (code == best[:, None]).float().argmax(1)
        tr = traj.reshape(Be, N, Tm, d)
        fr = tr[torch.arange(Be, device=dev)[:, None], pick, torch.arange(Tm, device=dev)[None]]
        out.append((b, fr.float()))
    return out


@torch.no_grad()
def candidates(F, k, looks):
    """Be x Tm x S x d: each frame with thing k in each of its states. Objects (``looks`` False):
    every value k is drawn in, and absent if it can be gone. Appearances (``looks`` True): k in
    the cell it is decoded in, in each of its channels."""
    vals = list(D.units.things[k])
    if not looks:
        Sn = len(vals) + (1 if k in can_go else 0)
        base = F.clone()
        base[:, :, vals] = 0
        G = base[:, :, None].repeat(1, 1, Sn, 1)
        for s, v in enumerate(vals):
            G[:, :, s, v] = 1
        return G
    n_ch = D.units.shape[2]
    chans = sorted({v % n_ch for v in vals})
    on = F[:, :, vals]  # Be x Tm x |vals|
    cell = torch.tensor([v // n_ch for v in vals], device=dev)[on.argmax(-1)]  # Be x Tm
    present = on.sum(-1) > 0
    base = F.clone()
    base[:, :, vals] = 0
    G = base[:, :, None].repeat(1, 1, len(chans), 1)
    valset = set(vals)
    for s, c in enumerate(chans):
        idx = cell * n_ch + c
        okv = torch.tensor([int(x) in valset for x in range(d)], device=dev)[idx] & present
        G[:, :, s].scatter_(-1, idx[..., None], okv[..., None].float())
    # Where k isn't drawn, every candidate keeps the frame as decoded.
    G = torch.where(present[:, :, None, None], G, F[:, :, None])
    return G


@torch.no_grad()
def smooth(b, F, k, looks=False):
    """Frames with thing k's history (its place, or with ``looks`` its appearance) re-inferred
    exactly."""
    Tm, Be = b.real.shape
    G = candidates(F, k, looks)
    Sn = G.shape[2]
    ok = D.valid(G.reshape(-1, d))
    ok = torch.ones(Be * Tm * Sn, dtype=torch.bool, device=dev) if ok is None else ok
    logok = torch.where(ok, 0.0, -1e9).reshape(Be, Tm, Sn)
    ar = torch.arange(Be, device=dev)
    # Summary densities, per (episode, sentence, state).
    lz = torch.zeros(Be, Tm, Sn, device=dev)
    if b.z is not None and fit.gauss is not None:
        for t in range(Tm):
            u = D.view(G[:, t].reshape(-1, d))
            z = b.z[t].repeat_interleave(Sn, 0)
            lz[:, t] = kappa * fit.gauss.logp(u, z).reshape(Be, Sn)
    la = lz[:, 0] + logok[:, 0]  # start: summaries and validity
    alphas = [la]
    psis = []
    rows = ar.repeat_interleave(Sn * Sn)
    for t in range(1, Tm):
        P = G[:, t - 1][:, :, None].expand(Be, Sn, Sn, d).reshape(-1, d)
        C = G[:, t][:, None, :].expand(Be, Sn, Sn, d).reshape(-1, d)
        moved = (P != C).any(1)
        ew = fit.emit_words(
            b.tok[t], P, C, moved, D.actor_word(C, fit.places_t), rows=rows, focus=focus
        )
        h = D.fm.context(P[:, None])[:, 0]
        lt = torch.where(
            moved, np.log(1 - q) + D.fm.log_prob_next(h, C), torch.tensor(np.log(q), device=dev)
        )
        lp = (torch.log(ew.clamp(min=1e-30)) + lt).reshape(Be, Sn, Sn)
        lp = lp + lz[:, t][:, None, :] + logok[:, t][:, None, :]
        real = b.real[t][:, None, None]
        eye = torch.where(torch.eye(Sn, dtype=torch.bool, device=dev), 0.0, -1e9)[None]
        lp = torch.where(real, lp, eye)
        psis.append(lp)
        la = torch.logsumexp(alphas[-1][:, :, None] + lp, 1)
        alphas.append(la)
    beta = torch.zeros(Be, Sn, device=dev)
    post = [None] * Tm
    post[Tm - 1] = alphas[-1] + beta
    for t in range(Tm - 1, 0, -1):
        beta = torch.logsumexp(psis[t - 1] + beta[:, None, :], 2)
        post[t - 1] = alphas[t - 1] + beta
    pick = torch.stack(post, 1).argmax(-1)  # Be x Tm
    return G[ar[:, None], torch.arange(Tm, device=dev)[None], pick]


def score(per_batch):
    frames, positions = [], []
    for b, F in per_batch:
        keep = b.real.T.cpu().numpy()
        fr = F.to(torch.uint8).cpu().numpy()
        for e in range(F.shape[0]):
            frames.append(fr[e][keep[e]])
            positions.append(b.ends[e])
    frames, positions = np.concatenate(frames), np.concatenate(positions)
    pred = np.stack([state_to_array(decode(world, x, key)) for x in frames])
    r = score_predictions(world, group, pred, lang_truth(data, positions))
    return r["modulo_symmetry"], {k: round(v, 3) for k, v in r["components"].items()}


dec = decoded(S["batches"]["score"])
ex, comp = score(dec)
print(f"SMOOTH as decoded exact {ex:.3f} {json.dumps(comp)}", flush=True)
cur = dec
for k in objects:
    cur = [(b, smooth(b, F, k)) for b, F in cur]
    ex, comp = score(cur)
    print(f"SMOOTH after object {k} exact {ex:.3f} {json.dumps(comp)}", flush=True)
for k in sorted({u[1] for u in units if u[0] == "looks"}):
    cur = [(b, smooth(b, F, k, looks=True)) for b, F in cur]
    ex, comp = score(cur)
    print(f"SMOOTH after appearance of thing {k} exact {ex:.3f} {json.dumps(comp)}", flush=True)
