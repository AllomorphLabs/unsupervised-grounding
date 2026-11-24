import json
from pathlib import Path

import pytest
import torch

from grounding.data.config import CorpusConfig
from grounding.data.corpus import generate_pair
from grounding.models.train import (
    Batches,
    ModelsConfig,
    TrainConfig,
    load_model,
    train,
    training_seed,
)
from grounding.models.transformer import ModelConfig
from grounding.runs.record import read_record

CORPUS = CorpusConfig(world="tiny", episodes=80, min_len=5, max_len=10, held_out=0.25)
TINY = ModelsConfig(
    ModelConfig(layers=1, width=32, heads=2),
    TrainConfig(steps=150, batch_size=16, lr=3e-3, warmup=10, eval_every=50, max_minutes=1),
)


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    root = tmp_path_factory.mktemp("pair")
    generate_pair(CORPUS, 0, root / "data")
    return root


@pytest.fixture(scope="module", params=["lang", "obs"])
def trained(request, pair):
    return request.param, train(pair, request.param, TINY, device="cpu")


def metrics(pair, stream):
    lines = (Path(pair) / "models" / stream / "metrics.jsonl").read_text().splitlines()
    return [json.loads(x) for x in lines]


def test_validation_loss_falls(pair, trained):
    stream, summary = trained
    rows = metrics(pair, stream)
    assert summary["validation_loss"] < 0.7 * rows[0]["validation_loss"]


def test_the_best_step_is_the_one_kept(pair, trained):
    stream, summary = trained
    rows = metrics(pair, stream)
    best = min(rows, key=lambda r: r["validation_loss"])
    assert summary["best_step"] == best["step"]
    assert summary["validation_loss"] == pytest.approx(best["validation_loss"])


def test_held_out_episodes_never_reach_training_logs(pair, trained):
    stream, _ = trained
    assert all("held_out_loss" not in row for row in metrics(pair, stream))


def test_train_validation_and_held_out_are_disjoint(pair):
    data = Batches("lang", pair / "data", torch.device("cpu"), 0.2, seed=1)
    sets = [set(data.train), set(data.validation), set(data.held_out)]
    assert all(sets) and sum(map(len, sets)) == len(set().union(*sets))
    assert len(data.validation) == round(0.2 * (len(data.train) + len(data.validation)))


def test_outputs_and_record(pair, trained):
    stream, summary = trained
    out = pair / "models" / stream
    assert {p.name for p in out.iterdir()} == {
        "model.pt",
        "metrics.jsonl",
        "run.json",
        "summary.json",
    }
    record = read_record(out)
    assert record.seed == training_seed(0, stream)
    assert record.config["stream"] == stream and summary["steps"] == TINY.train.steps


def test_saved_model_reloads(pair, trained):
    stream, _ = trained
    model = load_model(pair / "models" / stream)
    x = torch.randint(0, 5, (1, 6)) if stream == "lang" else torch.zeros(1, 6, model.frame_size)
    assert model(x).shape[:2] == (1, 6)


def test_refuses_to_overwrite(pair, trained):
    stream, _ = trained
    with pytest.raises(FileExistsError):
        train(pair, stream, TINY, device="cpu")


def test_training_is_reproducible_on_cpu(pair, tmp_path):
    a = train(pair, "lang", TINY, device="cpu", out=tmp_path / "a")
    b = train(pair, "lang", TINY, device="cpu", out=tmp_path / "b")
    assert a["held_out_loss"] == b["held_out_loss"]


def test_streams_and_pairs_get_different_training_seeds():
    seeds = {training_seed(k, s) for k in range(5) for s in ("lang", "obs")}
    assert len(seeds) == 10


def test_time_budget_stops_training(pair, tmp_path):
    cfg = ModelsConfig(TINY.model, TrainConfig(steps=10**6, max_minutes=0.0))
    summary = train(pair, "obs", cfg, device="cpu", out=tmp_path / "t")
    assert summary["stopped_by_time"] and summary["steps"] == 0


def test_pilot_models_config_loads():
    cfg = ModelsConfig.from_yaml(
        Path(__file__).resolve().parents[2] / "configs" / "pilot-models.yaml"
    )
    assert (cfg.model.layers, cfg.model.width, cfg.model.heads) == (4, 128, 4)
    assert cfg.train.max_minutes == 10


def test_pilot_language_config_loads():
    cfg = ModelsConfig.from_yaml(
        Path(__file__).resolve().parents[2] / "configs" / "pilot-lang.yaml"
    )
    assert (cfg.model.layers, cfg.model.width, cfg.model.heads) == (6, 256, 8)
    assert cfg.train.amp


def test_predictive_state_language_model_trains_and_reloads(pair, tmp_path):
    cfg = ModelsConfig(
        ModelConfig(layers=1, width=32, heads=2, future=6, summary=8),
        TrainConfig(steps=40, batch_size=8, lr=3e-3, warmup=5, eval_every=20, max_minutes=1),
    )
    train(pair, "lang", cfg, device="cpu", out=tmp_path / "ps")
    model = load_model(tmp_path / "ps")
    x = torch.randint(0, 5, (1, 12))
    assert model.summaries(x).shape == (1, 12, 8)


def test_joint_frame_model_trains_and_reloads(pair, tmp_path):
    cfg = ModelsConfig(
        ModelConfig(layers=1, width=32, heads=2, joint=True, joint_hidden=64),
        TrainConfig(steps=40, batch_size=8, lr=3e-3, warmup=5, eval_every=20, max_minutes=1),
    )
    train(pair, "obs", cfg, device="cpu", out=tmp_path / "joint")
    model = load_model(tmp_path / "joint")
    assert model.joint is not None
    h = model.context(torch.zeros(1, 3, model.frame_size))[:, -1]
    assert model.sample_next(h).shape == (1, model.frame_size)
