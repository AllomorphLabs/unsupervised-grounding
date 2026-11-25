"""The statistics-only parser for state vectors (units.find_units_vector), checked against the
hidden truth, and decipherment through it end to end. CPU, seconds."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from grounding.align.decipher import DecipherConfig, GridDecipherer, decipher_pair
from grounding.align.units import UnitView, find_units_vector
from grounding.analysis.labels import load_key, load_obs_labels
from grounding.analysis.score import score_decipher
from grounding.data.config import CorpusConfig
from grounding.data.corpus import generate_pair
from grounding.data.load import load_obs
from grounding.models.train import ModelsConfig, TrainConfig, train
from grounding.models.transformer import ModelConfig
from grounding.render.state.vector import encode
from grounding.sim.symmetry import Relabel
from grounding.sim.world import AT, BROKEN, HELD, state_from_array
from grounding.sim.worlds import tiny

CORPUS = CorpusConfig(world="tiny", episodes=3000, min_len=8, max_len=14, held_out=0.2)
QUICK = TrainConfig(steps=30, batch_size=16, lr=3e-3, warmup=5, eval_every=30, max_minutes=1)
LANG = ModelsConfig(ModelConfig(layers=1, width=16, heads=2, future=6, summary=8), QUICK)
FRAMES = ModelsConfig(ModelConfig(layers=1, width=16, heads=2, joint=True, joint_hidden=32), QUICK)


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs") / "pilot" / "pair-000"
    generate_pair(CORPUS, 0, root / "data")
    return root


def test_finds_things_and_places(pair):
    u = find_units_vector(load_obs(pair / "data"))
    world = tiny()
    assert len(u.things) == len(world.agents) + len(world.objects)
    assert u.places == len(world.places)
    assert all(u.always)


def test_units_match_the_true_state(pair):
    """Each frame shows one "at" unit per agent and per object lying at a place, a "with" unit
    exactly when something is held, and a hunger flag per hungry agent."""
    data = pair / "data"
    obs, labels = load_obs(data), load_obs_labels(data)
    u = find_units_vector(obs)
    x = UnitView(u, "cpu")(torch.tensor(obs.frames[:3000], dtype=torch.float32)).numpy()
    kind = np.array([d[0] for d in u.units])
    world = tiny()
    for row, truth in zip(x, labels.states[:3000], strict=True):
        s = state_from_array(world, truth)
        lying = sum(k in (AT, BROKEN) for k, _ in s.loc)
        held = sum(k == HELD for k, _ in s.loc)
        assert row[kind == "at"].sum() == len(s.pos) + lying
        assert row[kind == "with"].sum() == held


def test_place_groups_match_the_hidden_shuffle(pair):
    """Values grouped into one place really are one place, for every agent and object."""
    data = pair / "data"
    u = find_units_vector(load_obs(data))
    slots = load_key(data)["obs_slots"]
    shuffle = Relabel(tuple(slots["agents"]), tuple(slots["objects"]), tuple(slots["places"]))
    world = tiny()
    agents = sorted({d[1] for d in u.units if d[0] == "looks"})  # things with flags
    assert len(agents) == len(world.agents)
    for p in range(u.places):
        members = [m for d, m in zip(u.units, u.members, strict=True) if d[0] == "at" and d[2] == p]
        values = {v for m in members for v in m}
        # Draw states with agent 0 at each true place in turn: exactly one true place puts an
        # agent's "at" value inside this group.
        hits = set()
        for true_p in range(len(world.places)):
            s = state_from_array(world, np.array([true_p, true_p, 0, 0, AT, 0, AT, 0]))
            on = set(np.flatnonzero(encode(world, s, None, shuffle)))
            if on & values & {v for k in agents for v in u.things[k]}:
                hits.add(true_p)
        assert len(hits) == 1


def test_decipherment_through_statistics_runs(pair):
    train(pair, "lang", LANG, device="cpu", out=pair / "models" / "lang-ps")
    train(pair, "obs", FRAMES, device="cpu", out=pair / "models" / "obs-joint")
    cfg = DecipherConfig(
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
        structure="statistics",
        word_kinds="classes",
        valid_only=True,
    )
    D = GridDecipherer(pair, cfg, "cpu")
    assert D.units.shape is None and D.roles
    decipher_pair(pair, replace(cfg), "cpu", name="stats", log=lambda s: None)
    report = score_decipher(pair, "stats")
    assert 0 <= report["result"]["modulo_symmetry"] <= 1
