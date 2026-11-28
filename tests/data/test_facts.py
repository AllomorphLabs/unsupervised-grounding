"""Frames redrawn from their facts (technical decision 29): same facts, one drawing per state."""

import numpy as np
import pytest

from grounding.align.units import canonical_frames, find_units
from grounding.data.config import CorpusConfig
from grounding.data.corpus import generate_pair
from grounding.data.facts import check_views, thing_map
from grounding.data.load import ObsStream, load_obs

CFG = CorpusConfig(world="pilot", episodes=1500, min_len=20, max_len=40, observation="grid")


@pytest.fixture(scope="module")
def redrawn(tmp_path_factory):
    obs = load_obs(generate_pair(CFG, 0, tmp_path_factory.mktemp("facts") / "data"))
    units = find_units(obs)
    new = canonical_frames(obs.frames, units)
    again = find_units(ObsStream(new, obs.episode_start, obs.held_out, obs.shape))
    return obs, units, new, again


def test_redrawn_frames_parse_into_the_same_facts(redrawn):
    obs, units, new, again = redrawn
    perm = thing_map(units, again)  # asserts the same places and facts
    check_views(obs, new, units, again, perm)  # asserts every frame's facts are unchanged


def test_redrawing_leaves_fewer_drawings(redrawn):
    obs, _, new, _ = redrawn
    assert len(np.unique(new, axis=0)) < len(np.unique(obs.frames, axis=0))


def test_each_thing_is_drawn_once(redrawn):
    obs, units, new, _ = redrawn
    for vals in units.things:
        drawn = new[:, list(vals)].sum(1)
        assert drawn.max() <= 1
        assert (drawn == (obs.frames[:, list(vals)].sum(1) > 0)).all()
