"""Decipherment on a tiny pair: structure discovery, the model's parts, and end to end.
CPU only, a few seconds."""

import json
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from grounding.align.decipher import (
    COUNTS,
    CountedFrames,
    DecipherConfig,
    Decipherer,
    Gauss,
    SentenceFit,
    decipher_pair,
    make_batches,
    make_fit,
)
from grounding.align.structure import _transitions, frame_layout, word_types
from grounding.analysis.labels import load_key
from grounding.analysis.score import score_decipher
from grounding.data.config import CorpusConfig
from grounding.data.corpus import generate_pair
from grounding.data.load import load_lang, load_obs
from grounding.data.tokens import STOP
from grounding.models.train import ModelsConfig, TrainConfig, train
from grounding.models.transformer import ModelConfig

CORPUS = CorpusConfig(world="tiny", episodes=300, min_len=5, max_len=10, held_out=0.2)
QUICK = TrainConfig(steps=30, batch_size=16, lr=3e-3, warmup=5, eval_every=30, max_minutes=1)
LANG = ModelsConfig(ModelConfig(layers=1, width=16, heads=2, future=6, summary=8), QUICK)
FRAMES = ModelsConfig(ModelConfig(layers=1, width=16, heads=2, joint=True, joint_hidden=32), QUICK)
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
)


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    # Scoring reads the condition from the folder name, so build under a "pilot" folder.
    root = tmp_path_factory.mktemp("runs") / "pilot" / "pair-000"
    generate_pair(CORPUS, 0, root / "data")
    train(root, "lang", LANG, device="cpu", out=root / "models" / "lang-ps")
    train(root, "obs", FRAMES, device="cpu", out=root / "models" / "obs-joint")
    return root


def true_desc(v: int, a: int, p: int):
    """What frame value v means in the unshuffled layout (vector.encode)."""
    if v < a * (p + 2):
        k, r = divmod(v, p + 2)
        return ("place", k, r) if r < p else (("hungry", k) if r == p else ("acted", k))
    o, r = divmod(v - a * (p + 2), 2 * p + a + 1)
    if r < p:
        return ("obj_at", o, r)
    if r < p + a:
        return ("obj_held", o, r - p)
    return ("obj_broken", o, r - p - a) if r < 2 * p + a else ("obj_gone", o)


def test_frame_layout_reads_every_value_correctly(pair):
    key = load_key(pair / "data")
    a, p = len(key["world"]["agents"]), len(key["world"]["places"])
    lay = frame_layout(load_obs(pair / "data"))
    assert (len(lay.agents), len(lay.objects), lay.places) == (a, len(key["world"]["objects"]), p)
    unit = {k: true_desc(flag, a, p)[1] for k, (_, flag, _) in enumerate(lay.agents)}
    # Place ids are positions in agent 0's block; the shuffle is the same for every block.
    for v, (kind, e, x) in lay.values.items():
        td = true_desc(v, a, p)
        assert td[0] == kind, (v, kind, td)
        if kind in ("acted", "hungry", "place"):
            assert td[1] == unit[e]
        if kind in ("place", "obj_at", "obj_broken"):
            assert td[2] == x
        if kind == "obj_held":
            assert td[2] == unit[x]


def test_word_types_match_the_vocabulary(pair):
    key = load_key(pair / "data")
    lang = load_lang(pair / "data")
    name = {stem: concept for concept, stem in key["lang"]["stems"].items()}
    wt = word_types(lang)
    for kind, concepts in (("agents", "agents"), ("objects", "objects"), ("places", "places")):
        got = sorted(name[lang.words[t]] for t in getattr(wt, kind))
        assert got == sorted(key["world"][concepts])


@pytest.fixture(scope="module")
def decipherer(pair):
    return Decipherer(pair, TINY, "cpu")


@pytest.mark.parametrize("emission", ["word", "sentence"])
def test_tables_and_start_odds_are_consistent(decipherer, emission):
    D = Decipherer.__new__(Decipherer)
    D.__dict__.update(decipherer.__dict__)
    D.cfg = replace(decipherer.cfg, emission=emission)
    f = make_fit(D, D.matchings()[0])
    assert isinstance(f, SentenceFit) == (emission == "sentence")
    assert torch.allclose(f.t.ps.sum(-1), torch.ones(f.t.ps.shape[:2]), atol=1e-5)
    assert torch.allclose(f.t.pc.sum(-1), torch.ones(f.t.pc.shape[:2]), atol=1e-5)
    assert 0 < D.q < 1
    # The fast start-frame odds must equal the general emission at each start frame.
    eps = np.flatnonzero(~D.lang.held_out)[:8]
    b = make_batches(D.lang, eps, D.cfg, None, "cpu")[0]
    tok = b.tok[0]
    fast = f.start_emit(f.start_tables(), tok)
    for k in (0, 7, len(D.start) - 1):
        fr = D.start[k][None].repeat(len(tok), 1)
        no = torch.zeros(len(tok), dtype=torch.bool)
        slow = f.emit_words(tok, fr, fr, no, torch.full((len(tok),), D.V))
        assert torch.allclose(fast[:, k], slow, rtol=1e-4)


def test_gaussian_score_includes_the_width_penalty():
    torch.manual_seed(0)
    X = torch.randint(0, 2, (500, 5)).float()
    Y = X @ torch.randn(5, 3) + 0.3 * torch.randn(500, 3)
    g = Gauss.fit(X, Y)
    X1 = torch.cat([X, torch.ones(len(X), 1)], 1)
    cov = torch.linalg.inv(g.A @ g.A.T)
    ref = torch.distributions.MultivariateNormal(X1 @ g.Wb, covariance_matrix=cov).log_prob(Y)
    assert torch.allclose(g.logp(X, Y), ref + 0.5 * 3 * math.log(2 * math.pi), atol=1e-3)


def test_gaussian_with_pairs_matches_a_direct_fit():
    torch.manual_seed(0)
    X = torch.randint(0, 2, (3000, 6)).float()
    pairs = torch.triu_indices(6, 6, 1)
    Y = (X[:, 0] * X[:, 1])[:, None] * torch.tensor([[1.0, -2.0]]) + 0.1 * torch.randn(3000, 2)
    g = Gauss.fit(X, Y, pairs=pairs, chunk=700)  # several chunks
    P = Gauss.features(X, pairs)
    Wb = torch.linalg.solve(P.T @ P + 1e-2 * torch.eye(P.shape[1]), P.T @ Y)
    assert torch.allclose(g.Wb, Wb, atol=1e-3)
    R = Y - P @ Wb
    cov = R.T @ R / len(R) + 1e-3 * torch.eye(2)
    assert torch.allclose(torch.linalg.inv(g.A @ g.A.T), cov, atol=1e-3)
    # The product is what carries Y, so the pairs model explains it far better than the linear one.
    assert g.logp(X, Y).mean() > Gauss.fit(X, Y).logp(X, Y).mean() + 1


def test_decipher_end_to_end_writes_and_scores(pair):
    summary = decipher_pair(pair, TINY, device="cpu", log=lambda s: None)
    assert len(summary["finalists"]) == 2 and summary["selected"] in (0, 1)
    out = pair / "align" / "decipher"
    lang = load_lang(pair / "data")
    with np.load(out / "finalist-0.npz") as z:
        frames, positions = z["frames"], z["positions"]
    assert len(frames) == len(positions)
    assert frames.shape[1] == load_obs(pair / "data").frames.shape[1]
    assert np.all(lang.tokens[positions] == lang.words.index(STOP))
    report = score_decipher(pair)
    assert 0 <= report["result"]["exact"] <= 1 and len(report["finalists"]) == 2
    assert json.loads((out / "run.json").read_text())["config"]["decipher"]["particles"] == 4


def test_counted_frames_propose_only_seen_transitions(pair):
    obs = load_obs(pair / "data")
    cf = CountedFrames(obs, "cpu")
    f0, f1, _ = _transitions(obs, ~obs.held_out)
    seen = {
        (a.tobytes(), b.tobytes())
        for a, b in zip(f0.astype(np.uint8), f1.astype(np.uint8), strict=True)
    }
    h = torch.tensor(f0[:50], dtype=torch.float32).repeat(4, 1)
    nxt = cf.sample_next(h, torch.Generator().manual_seed(0))
    for a, b in zip(h.numpy().astype(np.uint8), nxt.numpy().astype(np.uint8), strict=True):
        assert (a.tobytes(), b.tobytes()) in seen
    # A frame never seen as a source still gets a 0/1 next frame, by the nearest seen one.
    odd = torch.ones(3, f0.shape[1])
    out = cf.sample_next(odd, torch.Generator().manual_seed(0))
    assert set(out.unique().tolist()) <= {0.0, 1.0}


def test_decipher_runs_without_a_frame_model(pair):
    cfg = replace(TINY, frame_model=COUNTS, summary_from=99, shortlist=1)
    summary = decipher_pair(pair, cfg, device="cpu", name="counts", log=lambda s: None)
    assert len(summary["finalists"]) == 1
