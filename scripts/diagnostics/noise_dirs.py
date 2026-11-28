"""Diagnostic (entry 105; uses hidden/): could split moves ever reach the true-frame summary model's
rows for a thing's place facts? Compares those rows with the main directions of the fit's
left-over noise, the only directions split moves try.

Loads a saved fit (a checkpoint of refit_restarts.py) on a facts pair, and a model fitted to the
TRUE frames (redrawn as the facts pair is: oracle_objective_lib, units.canonical_frames). Then
decodes the score episodes four ways: the fit as saved; with the true-frame word tables; with the
true-frame summary model; and the true-frame model itself. Whichever swap restores hunger shows
which half the fit gets wrong.

    GRID_CFG=<config> uv run python scripts/diagnostics/hunger_halves.py <facts pair> <original \
        pair> <checkpoint>
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from oracle_objective_lib import setup  # noqa: E402

from grounding.align.decipher import make_fit  # noqa: E402
from grounding.align.units import canonical_frames, find_units  # noqa: E402
from grounding.analysis.score import lang_truth  # noqa: E402
from grounding.data.load import load_obs  # noqa: E402
from grounding.render.grid.frames import EpisodeRenderer  # noqa: E402
from grounding.sim.world import state_from_array  # noqa: E402

pair, original, ck = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
S = setup(pair)
cfg, D, data, key, world = S["cfg"], S["D"], S["data"], S["key"], S["world"]
raw_units = find_units(load_obs(original / "data"))
saved = torch.load(ck, map_location="cuda", weights_only=False)
m = saved["state"]["m"]
print(
    f"HALVES {pair.name} from {ck.name} (iteration {saved['iteration']}), matching {m}", flush=True
)

# The true-frame model, on frames redrawn from their facts as the facts pair's are.
rng = np.random.default_rng(0)
N = cfg.particles
paths = []
for b in S["batches"]["fit"]:
    Tm, Be = b.real.shape
    traj = torch.zeros(Be * N, Tm, D.d, dtype=torch.uint8, device="cuda")
    mv = torch.zeros(Be * N, Tm, dtype=torch.bool, device="cuda")
    for e in range(Be):
        n = int(b.real[:, e].sum())
        truth = lang_truth(data, b.ends[e][:n])
        pen = EpisodeRenderer(world, key, rng)
        frames = [pen.start(state_from_array(world, truth[0]))]
        frames += [pen.step(state_from_array(world, truth[t])) for t in range(1, n)]
        fr = canonical_frames(np.stack(frames), raw_units)
        fr = torch.tensor(fr, dtype=torch.uint8, device="cuda")
        moved = torch.zeros(Tm, dtype=torch.bool, device="cuda")
        moved[1:n] = (fr[1:] != fr[:-1]).any(1)
        rows = torch.arange(N, device="cuda") + e * N
        traj[rows, :n] = fr
        mv[rows] = moved
    paths.append((b, traj, mv))
oracle = make_fit(D, m)
for it in range(max(cfg.summary_from, 6)):
    oracle.update(paths, it)
del paths

fit = make_fit(D, m)
fit.__dict__.update(saved["state"])
from grounding.align.latent import _noise_directions  # noqa: E402

g, o = fit.gauss, oracle.gauss
N = _noise_directions(g, 64)  # 64 x D, largest first
Q, _ = torch.linalg.qr(N.T)  # D x 64, orthonormal, in order
for j, u in enumerate(D.units.units):
    if u[0] != "at":
        continue
    w_true = o.Wb[j] - o.Wb[
        [k for k, v in enumerate(D.units.units) if v[0] == "at" and v[1] == u[1]]
    ].mean(0)
    w_fit = g.Wb[j] - g.Wb[
        [k for k, v in enumerate(D.units.units) if v[0] == "at" and v[1] == u[1]]
    ].mean(0)
    share = [float((Q[:, :k].T @ w_true).norm() ** 2 / w_true.norm() ** 2) for k in (4, 8, 16, 32)]
    cos = float(torch.nn.functional.cosine_similarity(w_true, w_fit, dim=0))
    print(
        f"DIRS {str(u):18s} true size {float(w_true.norm()):.2f} "
        f"fit size {float(w_fit.norm()):.2f} "
        f"cos {cos:+.2f} share in top 4/8/16/32 noise directions "
        + " ".join(f"{x:.2f}" for x in share),
        flush=True,
    )
