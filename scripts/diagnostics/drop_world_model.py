"""Diagnostic (entry 108): does the learned world model rule out a dropped, broken glass landing
away from the dropper? At real drops in a facts pair's frame stream (a thing held in a carrier's
cell, then broken at a place), compares the world model's log probability of the true next frame
with the same frame with the broken thing moved to each other place: with one frame of context
(as grounding.align.objects uses) and with the episode's history so far.

    uv run python scripts/diagnostics/drop_world_model.py <facts pair> <thing> [max events]
"""

import sys
from pathlib import Path

import numpy as np
import torch

from grounding.align.units import find_units
from grounding.data.load import load_obs
from grounding.models.train import load_model

pair, k = Path(sys.argv[1]), int(sys.argv[2])
cap = int(sys.argv[3]) if len(sys.argv) > 3 else 300
obs = load_obs(pair / "data")
u = find_units(obs)
n_ch = u.shape[2]
pc = np.array(u.place_of_cell)
vals = list(u.things[k])
chans = sorted({v % n_ch for v in vals})
broken = max(chans)  # the channel it gains when it breaks (its rarer one)
cell_of = {(int(pc[v // n_ch]), v % n_ch): v for v in vals}
fm = load_model(pair / "models" / "obs-joint", "cuda").eval()
ctx = fm.backbone.cfg.context
starts = np.append(obs.episode_start, len(obs.frames))
others = [vals2 for vals2 in u.things if vals2 != u.things[k]]
res = {"one frame": [], "history": []}
n = 0
for e in range(len(obs.episode_start)):
    if n >= cap:
        break
    fr = obs.frames[starts[e] : starts[e + 1]]
    for t in range(1, len(fr)):
        a, b = fr[t - 1][vals], fr[t][vals]
        if not a.any() or not b.any():
            continue
        va, vb = vals[int(a.argmax())], vals[int(b.argmax())]
        held = any(fr[t - 1][[x for x in o if x // n_ch == va // n_ch]].any() for o in others)
        if not (held and vb % n_ch == broken and va % n_ch != broken):
            continue
        p_true = int(pc[vb // n_ch])
        alts = []
        for (p, c), v in cell_of.items():
            if c == broken and p != p_true:
                f = fr[t].copy()
                f[vb] = 0
                f[v] = 1
                alts.append(f)
        nxt = torch.tensor(np.stack([fr[t]] + alts), dtype=torch.float32, device="cuda")
        with torch.no_grad():
            for label, hist in (("one frame", fr[t - 1 : t]), ("history", fr[max(0, t - ctx) : t])):
                h = torch.tensor(hist[None], dtype=torch.float32, device="cuda")
                hc = fm.context(h)[:, -1].repeat(len(nxt), 1)
                lp = fm.log_prob_next(hc, nxt).cpu().numpy()
                res[label].append((lp[0] - lp[1:].max(), lp[0] > lp[1:].max()))
        n += 1
        if n >= cap:
            break
for label, r in res.items():
    gap = np.array([x[0] for x in r])
    win = np.array([x[1] for x in r])
    print(
        f"DROP {label:9s}: {len(r)} drops; true landing most likely {win.mean():.3f}; "
        f"log-probability margin over the best wrong place median {np.median(gap):.2f}, "
        f"10th percentile {np.percentile(gap, 10):.2f}",
        flush=True,
    )
