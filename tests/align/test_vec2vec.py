import numpy as np
import pytest
import torch

from grounding.align.vec2vec import Vec2VecConfig, fit, mmd


def clusters(n: int, d: int, seed: int) -> tuple[torch.Tensor, np.ndarray]:
    """Points from five clusters of unequal size and spread, so only one matching fits."""
    rng = np.random.default_rng(seed)
    centres = np.random.default_rng(99).normal(size=(5, d)) * 3
    sizes = np.array([0.35, 0.25, 0.2, 0.12, 0.08])
    label = rng.choice(5, size=n, p=sizes)
    spread = np.array([0.2, 0.3, 0.4, 0.5, 0.6])[label, None]
    return torch.tensor(
        centres[label] + spread * rng.normal(size=(n, d)), dtype=torch.float32
    ), label


def test_mmd_separates_distributions():
    rng = torch.Generator().manual_seed(0)
    a, b = torch.randn(300, 4, generator=rng), torch.randn(300, 4, generator=rng)
    assert mmd(a, b) < mmd(a, b + 2)


@pytest.fixture
def one_thread():
    # These networks are tiny: spreading them over many CPU threads costs more than it saves.
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_recovers_a_hidden_rotation_without_pairs(one_thread):
    # Y is X's distribution rotated, sampled separately: no point in Y is paired with one in X.
    # Restarts land on different answers; the one with the lowest selection score (MMD) must
    # be the right one. That needs strong geometry preservation (the default): at weight 1,
    # the lowest-scoring restart was wrong.
    d = 8
    # The smallest setting found that still selects the right restart: with 600 points the
    # selection score is too noisy, and with 400 steps or width 16 it picks a wrong restart.
    x, x_label = clusters(800, d, seed=1)
    y_raw, y_label = clusters(800, d, seed=2)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(3)))
    y = y_raw @ q
    cfg = Vec2VecConfig(hidden=32, steps=600, batch=128, restarts=6, score_sample=300)
    best = min((fit(x, y, cfg, seed=s) for s in range(cfg.restarts)), key=lambda f: f.score)
    mapped = best.to_obs(x)
    targets = best.norm_y(y)
    nearest = (mapped @ targets.T).argmax(1).numpy()
    assert (y_label[nearest] == x_label).mean() > 0.95


def test_fit_is_reproducible_on_cpu():
    x, _ = clusters(300, 4, seed=0)
    y, _ = clusters(300, 4, seed=1)
    cfg = Vec2VecConfig(hidden=16, steps=20, batch=32, score_sample=100)
    assert fit(x, y, cfg, seed=5).score == fit(x, y, cfg, seed=5).score
