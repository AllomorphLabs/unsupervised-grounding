"""The points to align: one per sentence end in the language stream, one per frame.

Alignment may use only public structure (technical decision 14). Sentence ends
are the ``<.>`` token, and episode boundaries are in the public data, so both
are allowed. Which sentences are recaps or opening sentences is ground truth,
so it is not used here; those points simply have no exact counterpart among
the frames.

Two kinds of points:

- ``model_points``: each model's hidden state at that moment, from one layer.
- ``raw_points``: the co-occurrence baseline, with no neural model. For
  language, the token counts of the current sentence, joined to the share of
  each token so far in the episode. For observations, the raw frame.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from grounding.data.load import load_lang, load_obs
from grounding.data.tokens import STOP


@dataclass(frozen=True)
class Points:
    """Points from one stream. ``position`` is the token index (language) or frame index
    (observations) of each point, within the whole stream."""

    x: torch.Tensor  # (n, d), on the CPU
    episode: np.ndarray
    position: np.ndarray
    held_out: np.ndarray  # per point, from its episode

    def subset(self, mask: np.ndarray) -> Points:
        return Points(
            self.x[torch.from_numpy(mask)],
            self.episode[mask],
            self.position[mask],
            self.held_out[mask],
        )

    @property
    def train(self) -> Points:
        return self.subset(~self.held_out)

    @property
    def test(self) -> Points:
        return self.subset(self.held_out)


def _spans(starts: np.ndarray, total: int) -> list[tuple[int, int]]:
    return list(zip(starts.tolist(), np.append(starts[1:], total).tolist(), strict=True))


def _episode_of(positions: np.ndarray, starts: np.ndarray) -> np.ndarray:
    return np.searchsorted(starts, positions, side="right") - 1


def _select(starts: np.ndarray, total: int, episodes: np.ndarray | None) -> np.ndarray:
    """Episode indices to use: all of them, or the given ones, in stream order."""
    return np.arange(len(starts)) if episodes is None else np.sort(np.asarray(episodes))


@torch.no_grad()
def model_points(
    data_dir: Path,
    stream: str,
    model: torch.nn.Module,
    layer: int = -1,
    episodes: np.ndarray | None = None,
    batch: int = 64,
    window: int = 1,
) -> Points:
    """Hidden states from ``layer`` (0 is the input embedding, -1 the last layer), for the
    given episodes (default: all).

    With ``window > 1``, each point is the mean of the hidden states at its own position
    and the ``window - 1`` positions before it, within the episode. A language model holds
    at each position mostly the local state it needs there; pooling gathers what it knows
    about every entity into one point. Only position, a public fact, is used.
    """
    device = next(model.parameters()).device
    s = load_lang(data_dir) if stream == "lang" else load_obs(data_dir)
    data = s.tokens if stream == "lang" else s.frames
    spans = _spans(s.episode_start, len(data))
    chosen = _select(s.episode_start, len(data), episodes)
    stop = s.words.index(STOP) if stream == "lang" else None

    xs, episode, position = [], [], []
    for i in range(0, len(chosen), batch):
        eps = chosen[i : i + batch]
        chunk = [spans[e] for e in eps]
        t = max(b - a for a, b in chunk)
        if stream == "lang":
            x = torch.full((len(chunk), t), model.pad, dtype=torch.int64)
            for j, (a, b) in enumerate(chunk):
                x[j, : b - a] = torch.from_numpy(data[a:b].astype(np.int64))
        else:
            x = torch.zeros((len(chunk), t, data.shape[1]))
            for j, (a, b) in enumerate(chunk):
                x[j, : b - a] = torch.from_numpy(data[a:b].astype(np.float32))
        h = model.hidden(x.to(device))[layer].float().cpu()
        if window > 1:
            c = torch.cumsum(h, 1)
            lag = torch.zeros_like(c)
            lag[:, window:] = c[:, :-window]
            n = torch.clamp(torch.arange(1, h.shape[1] + 1), max=window).view(1, -1, 1)
            h = (c - lag) / n
        for j, (e, (a, b)) in enumerate(zip(eps, chunk, strict=True)):
            rows = np.flatnonzero(data[a:b] == stop) if stream == "lang" else np.arange(b - a)
            xs.append(h[j, rows])
            episode.append(np.full(len(rows), e))
            position.append(a + rows)
    episode_a = np.concatenate(episode)
    return Points(torch.cat(xs).float(), episode_a, np.concatenate(position), s.held_out[episode_a])


def raw_points(data_dir: Path, stream: str, episodes: np.ndarray | None = None) -> Points:
    """The co-occurrence baseline's points: raw statistics, no neural model."""
    if stream == "obs":
        s = load_obs(data_dir)
        spans = _spans(s.episode_start, len(s.frames))
        chosen = _select(s.episode_start, len(s.frames), episodes)
        positions = np.concatenate([np.arange(*spans[e]) for e in chosen])
        episode = _episode_of(positions, s.episode_start)
        x = torch.from_numpy(s.frames[positions].astype(np.float32))
        return Points(x, episode, positions, s.held_out[episode])

    s = load_lang(data_dir)
    v = len(s.words)
    stop = s.words.index(STOP)
    spans = _spans(s.episode_start, len(s.tokens))
    rows, positions = [], []
    for e in _select(s.episode_start, len(s.tokens), episodes):
        a, b = spans[e]
        so_far = np.zeros(v)
        sentence = np.zeros(v)
        for pos in range(a, b):
            tok = s.tokens[pos]
            so_far[tok] += 1
            sentence[tok] += 1
            if tok == stop:
                rows.append(np.concatenate([sentence, so_far / so_far.sum()]))
                positions.append(pos)
                sentence = np.zeros(v)
    positions_a = np.array(positions)
    episode = _episode_of(positions_a, s.episode_start)
    x = torch.tensor(np.array(rows), dtype=torch.float32)
    return Points(x, episode, positions_a, s.held_out[episode])
