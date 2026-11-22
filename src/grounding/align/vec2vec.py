"""A vec2vec-style aligner (after Jha et al., 2025), for two unpaired clouds of points.

Learns ``f``: language space -> observation space and ``g``: back again, from
unpaired samples only. No labels and no paired examples are used. Losses:

- adversarial: a discriminator on each side tries to tell mapped points from
  real ones, and the maps try to fool it (least-squares GAN);
- cycle: mapping there and back should return the starting point;
- geometry: the cosine similarities within a batch should survive mapping
  (vec2vec's "vector space preservation").

Each space is centred and every point scaled to unit length first. The fit is
restarted many times with different seeds; each restart draws its own
initialisation and its own random order of points (technical decision 11).
Restarts are compared by an unsupervised score, the maximum mean discrepancy
(MMD) between mapped and real points in both directions, and the lowest is
kept. All restarts are recorded.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class Vec2VecConfig:
    hidden: int = 256
    steps: int = 3000
    batch: int = 256
    lr: float = 1e-3
    disc_lr: float = 1e-3
    adv: float = 1.0
    cycle: float = 10.0
    geometry: float = 100.0
    restarts: int = 20
    score_sample: int = 2000  # points per side for the MMD selection score


def mlp(d_in: int, d_out: int, hidden: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d_in, hidden),
        nn.SiLU(),
        nn.LayerNorm(hidden),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, d_out),
    )


class Normaliser:
    """Centre on the training mean, then scale each point to unit length."""

    def __init__(self, x: torch.Tensor):
        self.mean = x.mean(0)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x - self.mean.to(x.device), dim=-1)


def mmd(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared MMD with a Gaussian kernel, bandwidth from the median distance."""
    z = torch.cat([a, b])
    d = torch.cdist(z, z).pow(2)
    bandwidth = d[d > 0].median().clamp_min(1e-8)
    k = torch.exp(-d / bandwidth)
    n = len(a)
    return (k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean()).item()


def _gram_loss(x: torch.Tensor, fx: torch.Tensor) -> torch.Tensor:
    xn, fn = F.normalize(x, dim=-1), F.normalize(fx, dim=-1)
    return (xn @ xn.T - fn @ fn.T).pow(2).mean()


@dataclass
class Fitted:
    f: nn.Module
    g: nn.Module
    norm_x: Normaliser
    norm_y: Normaliser
    seed: int
    score: float  # MMD selection score, lower is better
    history: list[dict]

    @torch.no_grad()
    def to_obs(self, x: torch.Tensor) -> torch.Tensor:
        """Map raw language points into the normalised observation space."""
        dev = next(self.f.parameters()).device
        return F.normalize(self.f(self.norm_x(x.to(dev))), dim=-1)

    def state(self) -> dict:
        return {
            "f": self.f.state_dict(),
            "g": self.g.state_dict(),
            "mean_x": self.norm_x.mean,
            "mean_y": self.norm_y.mean,
            "seed": self.seed,
            "score": self.score,
        }


def fit(
    x: torch.Tensor, y: torch.Tensor, cfg: Vec2VecConfig, seed: int, device: str = "cpu"
) -> Fitted:
    """One restart: align language points ``x`` with observation points ``y``."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    norm_x, norm_y = Normaliser(x), Normaliser(y)
    xs, ys = norm_x(x).to(device), norm_y(y).to(device)
    dx, dy = xs.shape[1], ys.shape[1]

    f, g = mlp(dx, dy, cfg.hidden).to(device), mlp(dy, dx, cfg.hidden).to(device)
    disc_y, disc_x = mlp(dy, 1, cfg.hidden).to(device), mlp(dx, 1, cfg.hidden).to(device)
    opt = torch.optim.Adam([*f.parameters(), *g.parameters()], lr=cfg.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(
        [*disc_y.parameters(), *disc_x.parameters()], lr=cfg.disc_lr, betas=(0.5, 0.999)
    )

    def unit(z: torch.Tensor) -> torch.Tensor:
        return F.normalize(z, dim=-1)

    history = []
    for step in range(cfg.steps):
        bx = xs[torch.from_numpy(rng.integers(len(xs), size=cfg.batch)).to(device)]
        by = ys[torch.from_numpy(rng.integers(len(ys), size=cfg.batch)).to(device)]

        # Discriminators: real -> 1, mapped -> 0.
        with torch.no_grad():
            fx, gy = unit(f(bx)), unit(g(by))
        d_loss = (
            (disc_y(by) - 1).pow(2).mean()
            + disc_y(fx).pow(2).mean()
            + (disc_x(bx) - 1).pow(2).mean()
            + disc_x(gy).pow(2).mean()
        )
        opt_d.zero_grad(set_to_none=True)
        d_loss.backward()
        opt_d.step()

        # Maps: fool the discriminators, survive a round trip, keep the geometry.
        fx, gy = unit(f(bx)), unit(g(by))
        adv = (disc_y(fx) - 1).pow(2).mean() + (disc_x(gy) - 1).pow(2).mean()
        cycle = (unit(g(fx)) - bx).pow(2).sum(-1).mean() + (unit(f(gy)) - by).pow(2).sum(-1).mean()
        geometry = _gram_loss(bx, fx) + _gram_loss(by, gy)
        loss = cfg.adv * adv + cfg.cycle * cycle + cfg.geometry * geometry
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % max(1, cfg.steps // 10) == 0 or step == cfg.steps - 1:
            history.append(
                {
                    "step": step,
                    "adv": adv.item(),
                    "cycle": cycle.item(),
                    "geometry": geometry.item(),
                }
            )

    f.eval(), g.eval()
    with torch.no_grad():
        ix = torch.from_numpy(rng.choice(len(xs), min(cfg.score_sample, len(xs)), replace=False))
        iy = torch.from_numpy(rng.choice(len(ys), min(cfg.score_sample, len(ys)), replace=False))
        sx, sy = xs[ix.to(device)], ys[iy.to(device)]
        score = mmd(unit(f(sx)), sy) + mmd(unit(g(sy)), sx)
    return Fitted(f, g, norm_x, norm_y, seed, score, history)


def config_dict(cfg: Vec2VecConfig) -> dict:
    return asdict(cfg)
