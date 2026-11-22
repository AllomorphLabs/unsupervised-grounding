"""End to end on a tiny pair: build, train, align, score. CPU only, a few seconds."""

import json

import numpy as np
import pytest

from grounding.align.features import model_points, raw_points
from grounding.align.run import AlignConfig, align_pair, restart_seeds
from grounding.align.vec2vec import Vec2VecConfig
from grounding.analysis.score import score_pair
from grounding.data.config import CorpusConfig
from grounding.data.corpus import generate_pair
from grounding.data.load import load_lang, load_obs
from grounding.data.tokens import STOP
from grounding.models.train import ModelsConfig, TrainConfig, load_model, train
from grounding.models.transformer import ModelConfig

CORPUS = CorpusConfig(world="tiny", episodes=60, min_len=5, max_len=10, held_out=0.25)
MODELS = ModelsConfig(
    ModelConfig(layers=1, width=16, heads=2),
    TrainConfig(steps=30, batch_size=16, lr=3e-3, warmup=5, eval_every=30, max_minutes=1),
)
ALIGN = AlignConfig(
    vec2vec=Vec2VecConfig(hidden=16, steps=20, batch=32, restarts=2, score_sample=100)
)


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    # Scoring reads the condition from the folder name, so build under a "pilot" folder.
    root = tmp_path_factory.mktemp("runs") / "pilot" / "pair-000"
    generate_pair(CORPUS, 0, root / "data")
    for stream in ("lang", "obs"):
        train(root, stream, MODELS, device="cpu")
    return root


def test_language_points_sit_at_sentence_ends(pair):
    lang = load_lang(pair / "data")
    pts = model_points(pair / "data", "lang", load_model(pair / "models" / "lang"))
    assert np.all(lang.tokens[pts.position] == lang.words.index(STOP))
    assert pts.x.shape == (len(pts.position), MODELS.model.width)


def test_observation_points_cover_every_frame(pair):
    obs = load_obs(pair / "data")
    pts = model_points(pair / "data", "obs", load_model(pair / "models" / "obs"))
    assert np.array_equal(pts.position, np.arange(len(obs.frames)))
    assert pts.held_out.sum() == sum(
        (np.append(obs.episode_start[1:], len(obs.frames)) - obs.episode_start)[obs.held_out]
    )


def test_raw_points_have_the_baseline_shape(pair):
    lang, obs = raw_points(pair / "data", "lang"), raw_points(pair / "data", "obs")
    words = len(load_lang(pair / "data").words)
    assert lang.x.shape[1] == 2 * words
    assert np.allclose(lang.x[:, words:].sum(1), 1)  # shares of tokens so far
    assert obs.x.shape[1] == load_obs(pair / "data").frames.shape[1]


@pytest.mark.parametrize("kind", ["model", "raw"])
def test_align_then_score(pair, kind):
    summary = align_pair(pair, kind, ALIGN, device="cpu")
    scores = [r["score"] for r in summary["restarts"]]
    assert summary["selected"] == int(np.argmin(scores))
    out = pair / "align" / kind
    assert {p.name for p in out.iterdir()} == {
        "restart-00.pt",
        "restart-01.pt",
        "summary.json",
        "run.json",
    }

    result = score_pair(pair, kind, ALIGN, device="cpu")
    assert len(result["restarts"]) == ALIGN.vec2vec.restarts
    r = result["result"]
    for k in ("exact", "modulo_symmetry"):
        assert 0 <= r[k] <= 1 and 0 <= r["chance"][k] <= 1
    assert r["modulo_symmetry"] >= r["exact"]
    assert set(r["by_symmetry"]) == set(result["symmetries"]) == {"identity", "agents A->B, B->A"}
    saved = json.loads((pair / "analysis" / f"score-{kind}.json").read_text())
    assert saved["result"] == r


def test_alignment_refuses_to_overwrite(pair):
    if not (pair / "align" / "raw").exists():
        align_pair(pair, "raw", ALIGN, device="cpu")
    with pytest.raises(FileExistsError):
        align_pair(pair, "raw", ALIGN, device="cpu")


def test_restart_seeds_differ_by_kind_and_pair():
    seeds = restart_seeds(0, "model", 5) + restart_seeds(0, "raw", 5) + restart_seeds(1, "model", 5)
    assert len(set(seeds)) == 15
    assert restart_seeds(0, "model", 5) == restart_seeds(0, "model", 5)


def test_alignment_and_scoring_samples_never_mix(pair):
    from grounding.align.run import sample_episodes

    cfg = AlignConfig(episodes=20, score_episodes=8)
    s = sample_episodes(pair, cfg)
    for side, stream in (("lang", load_lang(pair / "data")), ("obs", load_obs(pair / "data"))):
        assert not stream.held_out[s["align"][side]].any()
        assert stream.held_out[s["score"][side]].all()
        assert len(s["align"][side]) == 20 and len(s["score"][side]) == 8
    again = sample_episodes(pair, cfg)
    assert all(np.array_equal(s[u][k], again[u][k]) for u in s for k in s[u])
