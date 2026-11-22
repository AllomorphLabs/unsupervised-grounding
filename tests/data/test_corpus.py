import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from grounding.analysis.labels import load_key, load_lang_labels, load_obs_labels
from grounding.data.config import CorpusConfig
from grounding.data.corpus import HIDDEN, generate_pair
from grounding.data.load import load_lang, load_obs
from grounding.data.tokens import EPISODE, STOP
from grounding.render.lang.grammar import KINDS
from grounding.render.state.vector import encode
from grounding.runs.record import read_record
from grounding.sim.symmetry import Relabel
from grounding.sim.world import state_from_array
from grounding.sim.worlds import tiny

CFG = CorpusConfig(world="tiny", episodes=300, min_len=5, max_len=15, held_out=0.2, recap=0.2)
SRC = Path(__file__).resolve().parents[2] / "src" / "grounding"


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    return generate_pair(CFG, 0, tmp_path_factory.mktemp("pair") / "data")


def lang_episode_lengths(data_dir) -> np.ndarray:
    labels = load_lang_labels(data_dir)
    ends = np.append(labels.state_start[1:], len(labels.states))
    return ends - labels.state_start - 1


def obs_episode_lengths(data_dir) -> np.ndarray:
    obs = load_obs(data_dir)
    ends = np.append(obs.episode_start[1:], len(obs.frames))
    return ends - obs.episode_start - 1


def test_same_seed_same_corpus(pair, tmp_path):
    again = generate_pair(CFG, 0, tmp_path / "data")
    for name in ("lang.npz", "obs.npz", f"{HIDDEN}/lang_labels.npz", f"{HIDDEN}/obs_labels.npz"):
        with np.load(pair / name) as a, np.load(again / name) as b:
            assert a.files == b.files
            assert all(np.array_equal(a[k], b[k]) for k in a.files)
    assert load_key(pair) == load_key(again)


def test_refuses_to_overwrite(pair):
    with pytest.raises(FileExistsError):
        generate_pair(CFG, 0, pair)


def test_lengths_are_random_within_bounds(pair):
    for lengths in (lang_episode_lengths(pair), obs_episode_lengths(pair)):
        assert lengths.min() >= CFG.min_len and lengths.max() <= CFG.max_len
        assert len(set(lengths)) > 5


def test_streams_share_no_episodes_or_timing(pair):
    # Leakage check: the two streams come from separate episodes with separate lengths.
    lang_len, obs_len = lang_episode_lengths(pair), obs_episode_lengths(pair)
    assert not np.array_equal(lang_len, obs_len)
    r = np.corrcoef(lang_len, obs_len)[0, 1]
    assert abs(r) < 4 / np.sqrt(len(lang_len))
    lang_starts = load_lang_labels(pair).states[load_lang_labels(pair).state_start]
    obs_starts = load_obs_labels(pair).states[load_obs(pair).episode_start]
    same = np.all(lang_starts == obs_starts, axis=1).mean()
    assert same < 0.1  # 81 possible starts in the tiny world, so chance is about 1 in 81


def test_held_out_share(pair):
    assert load_lang(pair).held_out.sum() == load_obs(pair).held_out.sum() == 60


def test_training_loaders_work_without_the_hidden_directory(pair, tmp_path):
    public = tmp_path / "public"
    shutil.copytree(pair, public, ignore=shutil.ignore_patterns(HIDDEN))
    assert load_lang(public).tokens.size and load_obs(public).frames.size


# Code that reaches the ground truth: the HIDDEN constant, a quoted "hidden" path, or an
# import of the labels module. Mentions in prose (docstrings, comments) don't count.
TOUCHES_HIDDEN = re.compile(
    r"\bHIDDEN\b|[\"']hidden[/\"']|^\s*(from|import)\s+grounding\.analysis(\.labels|\s+import\s+labels)",
    re.MULTILINE,
)


def test_only_analysis_and_corpus_building_touch_hidden_files():
    allowed = {SRC / "analysis" / "labels.py", SRC / "data" / "corpus.py"}
    offenders = [
        p.relative_to(SRC)
        for p in SRC.rglob("*.py")
        if p not in allowed and TOUCHES_HIDDEN.search(p.read_text("utf-8"))
    ]
    assert offenders == []


def test_the_hidden_file_check_catches_real_uses():
    assert TOUCHES_HIDDEN.search("from grounding.analysis.labels import load_key")
    assert TOUCHES_HIDDEN.search('path = d / "hidden" / "key.json"')
    assert not TOUCHES_HIDDEN.search("These loaders never open ``hidden/``.")


def test_frames_match_their_hidden_states(pair):
    world, obs, labels = tiny(), load_obs(pair), load_obs_labels(pair)
    slots = load_key(pair)["obs_slots"]
    shuffle = Relabel(tuple(slots["agents"]), tuple(slots["objects"]), tuple(slots["places"]))
    for i in range(0, len(obs.frames), 37):
        s = state_from_array(world, labels.states[i])
        actor = None if labels.actors[i] < 0 else int(labels.actors[i])
        assert np.array_equal(obs.frames[i], encode(world, s, actor, shuffle))


def test_every_step_has_one_event_sentence(pair):
    labels = load_lang_labels(pair)
    kinds = labels.sentences[:, 2]
    events = np.bincount(labels.sentences[kinds == KINDS.index("event"), 0])
    assert np.array_equal(events, lang_episode_lengths(pair))
    assert (kinds == KINDS.index("recap")).sum() > 0


def test_sentence_offsets_point_at_sentences(pair):
    lang, labels = load_lang(pair), load_lang_labels(pair)
    stop, ep = lang.words.index(STOP), lang.words.index(EPISODE)
    starts = labels.sentences[:, 4]
    before = lang.tokens[starts - 1]
    assert np.all((before == stop) | (before == ep))


def test_run_record_is_written(pair):
    record = read_record(pair)
    assert record.seed == 0 and record.config["world"] == "tiny"


def test_entity_order_is_shuffled_per_pair(tmp_path):
    # Decision 11: across pairs, agent A's token number is below B's about half the time,
    # and A takes the first state-vector slot about half the time.
    small = CorpusConfig(world="tiny", episodes=1, min_len=1, max_len=1)
    a_first_token = a_first_slot = 0
    n = 40
    for k in range(n):
        out = generate_pair(small, k, tmp_path / str(k))
        words, key = json.loads((out / "lang_tokens.json").read_text()), load_key(out)
        stems = key["lang"]["stems"]
        a_first_token += words.index(stems["A"]) < words.index(stems["B"])
        a_first_slot += key["obs_slots"]["agents"][0] == 0
    assert 8 < a_first_token < 32
    assert 8 < a_first_slot < 32


def test_pilot_config_loads():
    cfg = CorpusConfig.from_yaml(Path(__file__).resolve().parents[2] / "configs" / "pilot.yaml")
    assert cfg.world == "pilot"
    assert sum(cfg.lengths().values()) == pytest.approx(1)
    assert min(cfg.lengths()) == cfg.min_len and max(cfg.lengths()) == cfg.max_len


@pytest.mark.parametrize(
    "bad",
    [
        dict(episodes=0),
        dict(min_len=0),
        dict(min_len=10, max_len=5),
        dict(held_out=1.0),
        dict(recap=2),
    ],
)
def test_bad_configs_are_rejected(bad):
    with pytest.raises(ValueError):
        CorpusConfig(world="tiny", **{"episodes": 1, **bad})
