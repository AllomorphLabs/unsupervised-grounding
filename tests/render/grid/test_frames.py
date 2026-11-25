import numpy as np
import pytest

from grounding.analysis.labels import load_key, load_obs_labels
from grounding.data.config import CorpusConfig
from grounding.data.corpus import generate_pair
from grounding.data.load import load_lang, load_obs
from grounding.render.grid.frames import GridKey, decode, grid_shape, place_cells
from grounding.sim.world import state_from_array
from grounding.sim.worlds import pilot, tiny

CFG = CorpusConfig(world="tiny", episodes=200, min_len=5, max_len=15, observation="grid")


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    return generate_pair(CFG, 0, tmp_path_factory.mktemp("grid") / "data")


def test_every_frame_decodes_to_its_true_state(pair):
    world = tiny()
    key = GridKey.from_dict(load_key(pair)["grid"])
    obs, labels = load_obs(pair), load_obs_labels(pair)
    for frame, truth in zip(obs.frames, labels.states, strict=True):
        assert decode(world, frame, key) == state_from_array(world, truth)


def test_shape_is_public_and_consistent(pair):
    obs = load_obs(pair)
    key = GridKey.from_dict(load_key(pair)["grid"])
    assert obs.shape == grid_shape(tiny(), key)
    assert obs.frames.shape[1] == int(np.prod(obs.shape))


def test_only_a_carried_object_shares_a_cell(pair):
    obs = load_obs(pair)
    h, w, c = obs.shape
    per_cell = obs.frames.reshape(len(obs.frames), h * w, c).sum(-1)
    assert per_cell.max() <= 2  # a carrier and what it carries
    world = tiny()
    labels = load_obs_labels(pair)
    for frame, truth in zip(obs.frames[:500], labels.states[:500], strict=True):
        s = state_from_array(world, truth)
        doubles = int((frame.reshape(h * w, c).sum(-1) == 2).sum())
        assert doubles == sum(kind == 1 for kind, _ in s.loc)  # HELD == 1


def test_nothing_is_drawn_outside_the_patches(pair):
    world, key = tiny(), GridKey.from_dict(load_key(pair)["grid"])
    obs = load_obs(pair)
    h, w, c = obs.shape
    live = np.zeros(h * w, bool)
    for p in range(len(world.places)):
        live[place_cells(world, p, key)] = True
    drawn = obs.frames.reshape(len(obs.frames), h * w, c).any(-1).any(0)
    assert not drawn[~live].any()


def test_agents_are_drawn_alike():
    # Each agent gets two channels and the same treatment; which is which is random per pair.
    world = pilot()
    keys = [GridKey.random(world, np.random.default_rng(s)) for s in range(200)]
    first = [k.agent[0][0] < k.agent[1][0] for k in keys]
    assert 0.35 < np.mean(first) < 0.65
    assert all(len({c for pair in k.agent for c in pair}) == 4 for k in keys)


def test_language_stream_is_unchanged_by_the_observation_format(pair, tmp_path):
    state_cfg = CorpusConfig(world="tiny", episodes=200, min_len=5, max_len=15)
    other = generate_pair(state_cfg, 0, tmp_path / "data")
    a, b = load_lang(pair), load_lang(other)
    assert np.array_equal(a.tokens, b.tokens) and a.words == b.words
