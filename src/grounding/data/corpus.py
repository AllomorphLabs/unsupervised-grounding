"""Building one independent corpus pair.

A pair is a language stream and an observation stream from the same world,
built from separate episodes with separate seeds (technical decision 6), so
nothing about episode boundaries, lengths or timing lines up between them.

Layout of a pair's data directory:

    lang.npz            tokens, episode_start, held_out
    lang_tokens.json    token strings, in token-number order
    obs.npz             frames, episode_start, held_out
    run.json            config, seed, git commit, package versions
    hidden/             ground truth: read only by grounding.analysis
      lang_labels.npz   what each sentence reports, and every episode's states
      obs_labels.npz    every frame's true state and acting agent
      key.json          the vocabulary and the slot shuffle

Training code reads only the top level (``grounding.data.load``). Everything
that would reveal the true meaning of a token or slot is under ``hidden/``.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from grounding.data.config import CorpusConfig
from grounding.data.tokens import EPISODE, STOP, make_tokens
from grounding.render.lang.grammar import KINDS, Language, make_vocab
from grounding.render.state.vector import encode
from grounding.runs.record import make_record, write_record
from grounding.runs.seeds import part_rngs
from grounding.sim.symmetry import Relabel
from grounding.sim.world import Event, State, World, sample_initial, state_to_array, step
from grounding.sim.worlds import make_world

# Named random parts of each stream. Only ever append (see part_rngs).
LANG_PARTS = ("vocab", "tokens", "sim", "render", "split", "train")
OBS_PARTS = ("slots", "sim", "split", "train")
PARTS = {"lang": LANG_PARTS, "obs": OBS_PARTS}
HIDDEN = "hidden"

Episode = tuple[State, list[tuple[Event, State]]]


def simulate(world: World, cfg: CorpusConfig, rng: np.random.Generator) -> list[Episode]:
    """``cfg.episodes`` independent episodes, each with its own random length."""
    out = []
    for _ in range(cfg.episodes):
        length = int(rng.integers(cfg.min_len, cfg.max_len + 1))
        start = s = sample_initial(world, rng)
        steps = []
        for _ in range(length):
            e, s = step(world, s, rng)
            steps.append((e, s))
        out.append((start, steps))
    return out


def split(n: int, held_out: float, rng: np.random.Generator) -> np.ndarray:
    """A mask marking ``round(held_out * n)`` episodes, chosen at random, as held out."""
    mask = np.zeros(n, dtype=bool)
    mask[rng.choice(n, size=round(held_out * n), replace=False)] = True
    return mask


def episode_states(start: State, steps: list[tuple[Event, State]]) -> np.ndarray:
    return np.stack([state_to_array(start)] + [state_to_array(s) for _, s in steps])


@dataclass
class _Built:
    public: dict[str, np.ndarray]
    labels: dict[str, np.ndarray]


def _offsets(lengths: list[int]) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)


def build_lang(world: World, cfg: CorpusConfig, root: int):
    r = part_rngs(root, "lang", LANG_PARTS)
    vocab = make_vocab(world, r["vocab"])
    tokens = make_tokens(vocab, r["tokens"])
    lang = Language(world, vocab)
    episodes = simulate(world, cfg, r["sim"])

    ids: list[int] = []
    ep_start, n_states = [], []
    said_rows = []  # episode, step, kind, of_step, token_start
    states = []
    for i, (start, steps) in enumerate(episodes):
        ep_start.append(len(ids))
        ids.append(tokens.index[EPISODE])
        for sentence, said in lang.labelled_episode(start, steps, r["render"], cfg.recap):
            said_rows.append((i, said.step, KINDS.index(said.kind), said.of_step, len(ids)))
            ids.extend(tokens.index[m] for m in sentence)
            ids.append(tokens.index[STOP])
        states.append(episode_states(start, steps))
        n_states.append(len(steps) + 1)

    built = _Built(
        public={
            "tokens": np.array(ids, dtype=np.int32),
            "episode_start": np.array(ep_start, dtype=np.int64),
            "held_out": split(len(episodes), cfg.held_out, r["split"]),
        },
        labels={
            "sentences": np.array(said_rows, dtype=np.int64).reshape(-1, 5),
            "states": np.concatenate(states),
            "state_start": _offsets(n_states),
        },
    )
    key = {
        "stems": vocab.stems,
        "suffixes": vocab.suffixes,
        "inflections": {k: list(v) for k, v in vocab.inflections.items()},
    }
    return built, tokens, key


def build_obs(world: World, cfg: CorpusConfig, root: int):
    r = part_rngs(root, "obs", OBS_PARTS)
    shuffle = Relabel.random(world, r["slots"])
    episodes = simulate(world, cfg, r["sim"])

    frames, states, actors, lengths = [], [], [], []
    for start, steps in episodes:
        frames.append(encode(world, start, None, shuffle))
        actors.append(-1)
        for e, s in steps:
            frames.append(encode(world, s, e.agent, shuffle))
            actors.append(e.agent)
        states.append(episode_states(start, steps))
        lengths.append(len(steps) + 1)

    built = _Built(
        public={
            "frames": np.stack(frames),
            "episode_start": _offsets(lengths),
            "held_out": split(len(episodes), cfg.held_out, r["split"]),
        },
        labels={"states": np.concatenate(states), "actors": np.array(actors, dtype=np.int8)},
    )
    key = {"agents": shuffle.agents, "objects": shuffle.objects, "places": shuffle.places}
    return built, key


def generate_pair(cfg: CorpusConfig, root: int, out: Path) -> Path:
    """Build both streams of one pair from root seed ``root`` and write them to ``out``."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; refusing to overwrite a corpus")
    (out / HIDDEN).mkdir(parents=True, exist_ok=True)
    world = make_world(cfg.world, **cfg.world_args)

    lang, tokens, lang_key = build_lang(world, cfg, root)
    obs, slot_key = build_obs(world, cfg, root)

    np.savez_compressed(out / "lang.npz", **lang.public)
    (out / "lang_tokens.json").write_text(json.dumps(list(tokens.words)), encoding="utf-8")
    np.savez_compressed(out / "obs.npz", **obs.public)
    np.savez_compressed(out / HIDDEN / "lang_labels.npz", **lang.labels)
    np.savez_compressed(out / HIDDEN / "obs_labels.npz", **obs.labels)
    key = {
        "world": {"agents": world.agents, "objects": world.objects, "places": world.places},
        "lang": lang_key,
        "obs_slots": slot_key,
    }
    (out / HIDDEN / "key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    write_record(out, make_record(cfg.as_dict(), root))
    return out
