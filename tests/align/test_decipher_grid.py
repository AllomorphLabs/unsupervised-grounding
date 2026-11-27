"""Grid-video decipherment on a tiny pair, end to end: structure and word kinds found from each
stream alone, every order searched, and scoring through the hidden drawing key. CPU, seconds."""

import json
from dataclasses import replace

import pytest

from grounding.align.decipher import (
    DecipherConfig,
    GridDecipherer,
    GridSentenceFit,
    decipher_pair,
    make_fit,
)
from grounding.analysis.score import score_decipher
from grounding.data.config import CorpusConfig
from grounding.data.corpus import generate_pair
from grounding.models.train import ModelsConfig, TrainConfig, train
from grounding.models.transformer import ModelConfig

CORPUS = CorpusConfig(world="tiny", episodes=400, min_len=8, max_len=14, observation="grid")
QUICK = TrainConfig(steps=30, batch_size=16, lr=3e-3, warmup=5, eval_every=30, max_minutes=1)
LANG = ModelsConfig(ModelConfig(layers=1, width=16, heads=2, future=6, summary=8), QUICK)
FRAMES = ModelsConfig(ModelConfig(layers=1, width=16, heads=2, joint=True, joint_hidden=64), QUICK)
TINY = DecipherConfig(
    train_episodes=16,
    select_episodes=8,
    score_episodes=8,
    batch=8,
    particles=4,
    candidates=2,
    iterations=2,
    summary_from=1,
    freeze_mixture=1,
    update_paths=2,
    shortlist=2,
    word_kinds="classes",
    screen=3,
    screen_episodes=8,
    screen_iterations=1,
)


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs") / "pilot-grid" / "pair-000"
    generate_pair(CORPUS, 0, root / "data")
    train(root, "lang", LANG, device="cpu", out=root / "models" / "lang-ps")
    train(root, "obs", FRAMES, device="cpu", out=root / "models" / "obs-joint")
    return root


def test_structure_and_word_kinds_found(pair):
    D = GridDecipherer(pair, TINY, "cpu")
    assert len(D.units.things) == 4 and D.units.places == 3  # tiny: 2 agents, 2 objects
    assert D.roles  # at least one way for word classes to name what the frames show


def test_search_covers_both_agent_orders(pair):
    D = GridDecipherer(pair, TINY, "cpu")
    ms = D.matchings()
    seen = {(m.things, m.places) for m in ms}
    for m in ms:
        for i in range(len(m.things)):
            for j in range(i + 1, len(m.things)):
                t = list(m.things)
                t[i], t[j] = t[j], t[i]
                assert (tuple(t), m.places) in seen  # every reordering is a candidate too


def test_end_to_end(pair):
    summary = decipher_pair(pair, TINY, "cpu", name="t", log=lambda s: None)
    assert len(summary["screened"]) == 3 and len(summary["finalists"]) == 2
    report = score_decipher(pair, "t")
    assert 0 <= report["result"]["modulo_symmetry"] <= 1
    assert json.dumps(report)


def test_grid_fit_is_grid_kind(pair):
    D = GridDecipherer(pair, replace(TINY, screen=0), "cpu")
    assert isinstance(make_fit(D, D.matchings()[0]), GridSentenceFit)


def test_split_moves_run_inside_a_fit(pair):
    cfg = replace(TINY, screen=0, shortlist=1, iterations=3, summary_from=1, split_at=2)
    summary = decipher_pair(pair, cfg, "cpu", name="split", log=lambda s: None)
    assert len(summary["finalists"]) == 1


def test_context_cache_gives_the_same_filter(pair):
    """The cached context changes only the order of floating-point sums (entry 79)."""
    import torch

    from grounding.align.decipher import make_batches, sample, standardise
    from grounding.models.train import load_model
    from grounding.runs.record import read_record

    out = []
    for cached in (False, True):
        cfg = replace(TINY, screen=0, context_cache=cached)
        D = GridDecipherer(pair, cfg, "cpu")
        eps = sample(D.lang, read_record(pair / "data").seed, cfg)
        b = make_batches(
            D.lang, eps["fit"], cfg, load_model(pair / "models" / cfg.lang_model, "cpu"), "cpu"
        )
        standardise(b)
        f = make_fit(D, D.matchings()[0])
        paths, ll = f.run(b, seed=3)
        out.append((ll, torch.cat([p[1].flatten() for p in paths])))
    assert abs(out[0][0] - out[1][0]) < 1e-4 * max(1.0, abs(out[0][0]))
    assert (out[0][1] == out[1][1]).float().mean() > 0.99
