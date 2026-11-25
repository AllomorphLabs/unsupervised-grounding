import numpy as np
import pytest
import torch

from grounding.align.units import UnitView, find_units
from grounding.analysis.labels import load_key, load_obs_labels
from grounding.data.config import CorpusConfig
from grounding.data.corpus import generate_pair
from grounding.data.load import load_obs
from grounding.render.grid.frames import GridKey, place_cells
from grounding.sim.world import GONE, HELD, state_from_array
from grounding.sim.worlds import pilot

CFG = CorpusConfig(world="pilot", episodes=1500, min_len=20, max_len=40, observation="grid")


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    return generate_pair(CFG, 0, tmp_path_factory.mktemp("units") / "data")


@pytest.fixture(scope="module")
def units(pair):
    return find_units(load_obs(pair))


def test_finds_the_places(pair, units):
    world, key = pilot(), GridKey.from_dict(load_key(pair)["grid"])
    assert units.places == len(world.places)
    found = np.array(units.place_of_cell)
    for p in range(len(world.places)):
        assert len(set(found[place_cells(world, p, key)])) == 1  # each patch is one place


def test_finds_one_thing_per_entity(pair, units):
    key = GridKey.from_dict(load_key(pair)["grid"])
    n_ch = load_obs(pair).shape[2]
    channels = sorted(tuple(sorted({v % n_ch for v in t})) for t in units.things)
    truth = sorted(
        [tuple(sorted(a)) for a in key.agent]
        + [tuple(sorted(c for c in o if c >= 0)) for o in key.obj]
    )
    assert channels == truth


def test_unit_view_matches_the_true_state(pair, units):
    """Each frame shows exactly one "at" unit per thing on screen, and a "with" unit exactly when
    something is carried."""
    obs, labels = load_obs(pair), load_obs_labels(pair)
    view = UnitView(units, "cpu")
    u = view(torch.tensor(obs.frames[:3000], dtype=torch.float32)).numpy()
    kinds = [x[0] for x in units.units]
    at = np.array([k == "at" for k in kinds])
    with_ = np.array([k == "with" for k in kinds])
    world = pilot()
    for row, truth in zip(u, labels.states[:3000], strict=True):
        s = state_from_array(world, truth)
        on_screen = len(s.pos) + sum(kind != GONE for kind, _ in s.loc)
        carried = sum(kind == HELD for kind, _ in s.loc)
        assert row[at].sum() == on_screen
        assert row[with_].sum() == carried
        assert set(np.unique(row)) <= {0.0, 1.0}


def test_valid_accepts_real_frames_and_rejects_broken_ones(pair, units):
    obs = load_obs(pair)
    view = UnitView(units, "cpu")
    real = torch.tensor(obs.frames[:2000], dtype=torch.float32)
    assert view.valid(real).all()
    broken = real.clone()
    thing = units.things[0]
    on = broken[:, list(thing)].argmax(1)
    other = [thing[(int(i) + 1) % len(thing)] for i in on]
    broken[torch.arange(len(broken)), torch.tensor(other)] = 1  # draw thing 0 twice
    assert not view.valid(broken).any()
    always = [k for k, a in enumerate(units.always) if a][0]
    missing = real.clone()
    missing[:, list(units.things[always])] = 0  # an always-present thing vanishes
    assert not view.valid(missing).any()


def test_joint_units_name_the_channel(pair):
    u = find_units(load_obs(pair), split=False)
    assert all(len(x) == 4 for x in u.units if x[0] == "at")
    assert not any(x[0] == "looks" for x in u.units)
