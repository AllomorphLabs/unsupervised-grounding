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


def _patch_contents(world, frame, key):
    """Each patch's cells as a sorted list of channel sets: the frame with where in its patch each
    thing sits left out (that is drawn at random)."""
    from grounding.render.grid.frames import place_cells

    v = frame.reshape(-1, key.channels)
    out = []
    for p in range(len(world.places)):
        cells = [tuple(np.flatnonzero(v[c])) for c in place_cells(world, p, key)]
        out.append(sorted(c for c in cells if c))
    return out


@pytest.mark.parametrize("name", ["pilot", "lured"])
def test_renderer_check_the_agent_swap_survives_grid_video(name):
    # Proposal, "Renderer checks": relabelling a trace and then rendering it must match rendering
    # it with the relabelled key, and the relabelled key must be just as likely. Then the grid
    # renderer can't tell the interchangeable agents apart.
    from grounding.render.grid.frames import EpisodeRenderer
    from grounding.sim.symmetry import Relabel, automorphisms
    from grounding.sim.world import sample_initial, step
    from grounding.sim.worlds import make_world

    world = make_world(name)
    swap = Relabel((1, 0), tuple(range(len(world.objects))), tuple(range(len(world.places))))
    assert swap in automorphisms(world)
    rng = np.random.default_rng(3)
    for trial in range(20):
        key = GridKey.random(world, rng)
        swapped = GridKey(key.slot, (key.agent[1], key.agent[0]), key.obj, key.actor)
        states = [sample_initial(world, rng)]
        for _ in range(30):
            states.append(step(world, states[-1], rng)[1])
        a = EpisodeRenderer(world, key, np.random.default_rng(trial))
        b = EpisodeRenderer(world, swapped, np.random.default_rng(trial))
        fa = [a.start(states[0])] + [a.step(s) for s in states[1:]]
        fb = [b.start(swap.state(states[0]))] + [b.step(swap.state(s)) for s in states[1:]]
        for x, y in zip(fa, fb, strict=True):
            assert _patch_contents(world, x, key) == _patch_contents(world, y, swapped)
    # No fixed screen positions: within its patch, each agent lands on each cell equally often.
    from grounding.render.grid.frames import place_cells

    # Many short independent episodes, one reading each (readings within an episode are
    # correlated: an agent stays put for several steps).
    where = [[], []]
    for _ in range(4000):
        key = GridKey.random(world, rng)
        r = EpisodeRenderer(world, key, rng)
        s = sample_initial(world, rng)
        r.start(s)
        for _ in range(8):
            s = step(world, s, rng)[1]
            r.step(s)
        for ag in (0, 1):
            where[ag].append(place_cells(world, s.pos[ag], key).index(r.agent_cell[ag]))
    n = max(max(w) for w in where) + 1
    h0, h1 = (np.bincount(w, minlength=n) for w in where)
    assert np.all(np.abs(h0 - h1) < 5 * np.sqrt(h0 + h1 + 1))
    # The swapped key is as likely as the key: each agent is equally likely to get each channel
    # pair, so drawing keys at random never ties a channel to one agent.
    keys = [GridKey.random(world, np.random.default_rng(s)) for s in range(6000)]
    n_ch = keys[0].channels
    for part in (0, 1):  # normal, hungry
        a0 = np.bincount([k.agent[0][part] for k in keys], minlength=n_ch)
        a1 = np.bincount([k.agent[1][part] for k in keys], minlength=n_ch)
        assert np.all(np.abs(a0 - a1) < 5 * np.sqrt(a0 + a1 + 1))
