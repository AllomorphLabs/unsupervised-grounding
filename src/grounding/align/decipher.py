"""Decipherment: aligning language to frames through the two trained models.

    uv run python -m grounding.align.decipher runs/pilot/pair-001

The language stream is treated as a code for a hidden sequence of frames (experiment log,
entries 21-40; technical decision 19). Nothing pairs a sentence with a frame. Instead:

- **The frame model proposes.** For each language episode, a particle filter keeps
  ``particles`` guesses at the hidden frame history. At each sentence a guess either stays on its
  frame or moves to one of ``candidates`` next frames sampled from the trained frame model's
  joint head (technical decision 18).
- **The language side weighs the guesses**, in two ways:
  - *Words*, through shared role tables. Each kind of frame feature (a value that is on, turned
    on, turned off, or "nothing changed") has one table over roles: this agent, another agent,
    this object, another object, this place, another place, or a specific non-entity word. A
    *matching* turns roles into words, and a further component points at the word for the
    acting agent's place. With ``emission: sentence`` (``SentenceFit``) each sentence is about
    one fact, and all its words come from that fact; with ``emission: word`` (``Fit``) each word
    comes from any fact on its own.
  - *The language model's summaries* at sentence ends (technical decision 17), as a Gaussian
    whose mean is linear in the frame's values (``summary_features: pairs`` adds the product of
    every two live values; experiment log, entries 42-47), softened by ``summary_power``.
- **EM** refits the tables (and the Gaussian, from ``summary_from``) to the particles' paths.
  Before a particle first moves, each episode keeps exact odds over every start frame.

**Search over matchings.** All matchings are ranked by how far apart the two streams' raw
frequency profiles are under each, the top ``shortlist`` are each fitted separately, and the one
whose fit best explains held-out *selection* episodes is kept. Nothing here reads ``hidden/``.

Writes to ``<pair>/align/<name>/``: ``summary.json`` (shortlist, finalists' scores, selection),
``finalist-<i>.pt`` (fitted tables) and ``finalist-<i>.npz`` (the decoded frame at every sentence
end of the score episodes, for ``grounding.analysis.score --kind decipher``), and ``run.json``.
"""

import argparse
import itertools
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import yaml

from grounding.align.run import ALIGN_PARTS
from grounding.align.structure import FrameLayout, WordTypes, _transitions, frame_layout, word_types
from grounding.align.units import UnitView, find_units, find_units_vector
from grounding.align.words import Roles, role_options, word_classes
from grounding.data.load import LangStream, load_lang, load_obs
from grounding.data.tokens import EPISODE, STOP
from grounding.models.train import load_model
from grounding.runs.record import make_record, read_record, write_record
from grounding.runs.seeds import part_rngs


@dataclass(frozen=True)
class DecipherConfig:
    lang_model: str = "lang-ps"  # folders under <pair>/models/
    frame_model: str = "obs-joint"
    train_episodes: int = 512  # language episodes for fitting
    select_episodes: int = 128  # language episodes for choosing between finalists
    score_episodes: int = 300  # held-out language episodes decoded for scoring
    batch: int = 64  # episodes per particle-filter batch
    max_sentences: int = 100  # per episode
    sentence_tokens: int = 6  # word positions per sentence
    particles: int = 32
    candidates: int = 4  # next frames sampled per particle per sentence
    # Also weigh, exactly, every frame one appearance change or one disappearance away from each
    # particle's frame, scored by the frame model: rare single changes such as an agent turning
    # hungry are then never missed for want of a sample (entry 94). Grid structure only.
    edit_proposals: bool = False
    iterations: int = 12
    own_prior: float = 20.0  # pseudo-counts on each value's own entity words, to start
    # Pseudo-counts tying each other word to the changes that happen about as often as it is said,
    # from each stream's own rates (0: off; entry 93). Grid structure only.
    rate_prior: float = 0.0
    smooth: float = 0.01
    summary_power: float = 0.2
    summary_features: str = "linear"  # "linear": the frame; "pairs": also products of live values
    emission: str = "sentence"  # "word": each word from any fact; "sentence": all from one fact
    summary_from: int = 5  # iteration from which the summaries count
    freeze_mixture: int = 8  # iteration from which the mixture weights stay fixed
    update_paths: int = 8  # particle paths per episode used in each update
    shortlist: int = 6
    # Split moves on the summary model after this iteration (0: off; units only, entry 69).
    split_at: int = 0
    # Keep the frame model's keys and values for each particle's history, so a step puts only the
    # new frame through the model instead of the whole history. Checked bit-identical to the full
    # recompute on the GPU at a real search's size, with and without the sampler graph (entry 91).
    context_cache: bool = True
    # Replay the frame model's sampler as a CUDA graph on the GPU: the same operations in the same
    # order, launched at once, so the same numbers (entry 83); no effect on the CPU.
    sampler_graph: bool = True
    split_weakest: int = 0  # facts of smallest effect to try (0: those below a quarter of median)
    # Also try every appearance unit of every thing, whatever its effect size (entry 92).
    split_looks: bool = False
    split_rounds: int = 1
    split_orders: int = 1  # orders of the weak facts to try, keeping the likeliest (entry 74)
    # Also restart one thing's appearances together while both stay weak (entry 72).
    split_together: bool = False
    # Screening (0: off): the ``screen`` matchings closest by raw counts each get a short fit
    # (``screen_episodes``, ``screen_iterations``), and the ``shortlist`` that best explain the
    # selection episodes go on to the full fit. Raw counts alone can't tell some matchings apart.
    screen: int = 0
    screen_episodes: int = 128
    screen_iterations: int = 4
    # How words are sorted into kinds (grid video only): "rules", the pilot's grammar rules
    # (structure.word_types), or "classes", found from the language stream alone (align.words).
    word_kinds: str = "rules"
    # State vectors: "fixed" keeps the agents in one order (the pilot, decision 19); "all" searches
    # every order (decision 22). Grid video always searches every order.
    agent_orders: str = "fixed"
    # State vectors: "rules" uses the pilot's hand-written parser (structure.frame_layout);
    # "statistics" finds the structure from the frame stream's statistics alone
    # (units.find_units_vector) and deciphers through units, as for grid video.
    structure: str = "rules"
    # Grid video only. grid_units: "split" keeps where a thing is apart from how it looks;
    # "joint" makes each place-and-appearance one unit (align.units). valid_only: candidate frames
    # that break the frame stream's own structure (a thing drawn twice) get no weight.
    grid_units: str = "split"
    valid_only: bool = False
    seed: int = 0  # restart seed for a fit's random draws; restarts measure variance

    @classmethod
    def from_yaml(cls, path: Path) -> DecipherConfig:
        return cls(**(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}))


@dataclass(frozen=True)
class Matching:
    """Which word names each frame entity: per agent, per object block, per place id."""

    agents: tuple[int, ...]
    objects: tuple[int, ...]
    places: tuple[int, ...]


# ---------------------------------------------------------------------------------------------
# Language batches
# ---------------------------------------------------------------------------------------------


@dataclass
class Batch:
    tok: torch.Tensor  # sentences x episodes x positions; V marks padding
    real: torch.Tensor  # sentences x episodes
    z: torch.Tensor | None  # sentences x episodes x summary size
    ends: list[np.ndarray]  # per episode, token positions of its sentence ends


def make_batches(lang: LangStream, eps: np.ndarray, cfg: DecipherConfig, lm, dev) -> list[Batch]:
    starts = np.append(lang.episode_start, len(lang.tokens))
    stop, epi, V = lang.words.index(STOP), lang.words.index(EPISODE), len(lang.words)
    out = []
    for i in range(0, len(eps), cfg.batch):
        chunk = eps[i : i + cfg.batch]
        sents, ends, zs = [], [], []
        for e in chunk:
            toks = lang.tokens[starts[e] : starts[e + 1]]
            end = np.flatnonzero(toks == stop)[: cfg.max_sentences]
            b = 0
            s = []
            for x in end:
                s.append([t for t in toks[b:x] if t != epi][: cfg.sentence_tokens])
                b = x + 1
            sents.append(s)
            ends.append(starts[e] + end)
            if lm is not None:
                with torch.no_grad():
                    t_ = torch.tensor(toks.astype(np.int64), device=dev)[None]
                    zs.append(lm.summaries(t_)[0, torch.tensor(end, device=dev)].float())
        T = max(map(len, sents))
        tok = np.full((T, len(chunk), cfg.sentence_tokens), V, np.int64)
        real = np.zeros((T, len(chunk)), bool)
        for b, s in enumerate(sents):
            for t, w in enumerate(s):
                tok[t, b, : len(w)] = w
                real[t, b] = True
        z = None
        if lm is not None:
            z = torch.zeros(T, len(chunk), zs[0].shape[1], device=dev)
            for b, zz in enumerate(zs):
                z[: len(zz), b] = zz
        out.append(Batch(torch.tensor(tok, device=dev), torch.tensor(real, device=dev), z, ends))
    return out


def standardise(batches: list[Batch], mu=None, sd=None):
    if batches[0].z is None:
        return None, None
    if mu is None:
        allz = torch.cat([b.z[b.real] for b in batches])
        mu, sd = allz.mean(0), allz.std(0) + 1e-6
    for b in batches:
        b.z = (b.z - mu) / sd
    return mu, sd


# ---------------------------------------------------------------------------------------------
# The model: tables, tying, emissions
# ---------------------------------------------------------------------------------------------


class Gauss:
    """z ~ N(W phi(f) + b, Sigma): the summary as a noisy linear view of features of the frame.

    phi(f) is the frame's values and, if ``pairs`` is given, the product of each listed pair of
    values. A language model may store a fact (where an object is) only in combination with others
    (where the agents are), so that no straight-line read-out of single values reaches it; the
    products let the Gaussian use such facts (experiment log, entry 42).
    """

    def __init__(self, Wb: torch.Tensor, A: torch.Tensor, pairs: torch.Tensor | None = None):
        self.Wb, self.A, self.pairs = Wb, A, pairs
        # -0.5 log det(Sigma) = sum log diag(A): without this width penalty a worse fit (wider
        # Sigma) scores better whenever fits are compared (experiment log, entry 40).
        self.norm = torch.log(torch.diagonal(A).abs()).sum()

    @staticmethod
    def features(f: torch.Tensor, pairs: torch.Tensor | None) -> torch.Tensor:
        parts = [f]
        if pairs is not None:
            parts.append(f[:, pairs[0]] * f[:, pairs[1]])
        parts.append(torch.ones(len(f), 1, device=f.device))
        return torch.cat(parts, 1)

    @classmethod
    def fit(
        cls,
        X: torch.Tensor,
        Y: torch.Tensor,
        reg: float = 1e-2,
        pairs: torch.Tensor | None = None,
        chunk: int = 32768,
    ) -> Gauss:
        """Ridge regression of Y on phi(X), accumulated in chunks to bound memory."""
        n = len(X)
        XtX = XtY = None
        YtY = Y.T @ Y
        for i in range(0, n, chunk):
            P = cls.features(X[i : i + chunk], pairs)
            a, b = P.T @ P, P.T @ Y[i : i + chunk]
            XtX, XtY = (a, b) if XtX is None else (XtX + a, XtY + b)
        eye = torch.eye(XtX.shape[0], device=X.device)
        Wb = torch.linalg.solve(XtX + reg * eye, XtY)
        # Residual covariance from the sums: (Y - P Wb)^T (Y - P Wb) / n.
        RtR = YtY - Wb.T @ XtY - XtY.T @ Wb + Wb.T @ XtX @ Wb
        Sig = (RtR + RtR.T) / (2 * n) + 1e-3 * torch.eye(Y.shape[1], device=X.device)
        return cls(Wb, torch.linalg.inv(torch.linalg.cholesky(Sig)).T, pairs)

    def mean(self, f: torch.Tensor) -> torch.Tensor:
        return self.features(f, self.pairs) @ self.Wb

    def logp(self, f: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Log density of each z given frame f, up to the constant -0.5 D log(2 pi)."""
        r = (z - self.mean(f)) @ self.A
        return -0.5 * (r * r).sum(1) + self.norm


COUNTS = "counts"  # frame_model value: counted transitions instead of a trained frame model


class CountedFrames:
    """A frame "model" with no neural network, for the no-neural-model baseline: next frames are
    drawn from the frame stream's own counted transitions out of the current frame. A frame never
    seen as a source borrows the change from the nearest one that was (fewest values different).
    It has the two calls the particle filter makes of a trained frame model; its "context" is just
    the current frame, so it sees no further back than one frame."""

    joint = True

    def __init__(self, obs, device: str, chunk: int = 4096):
        f0, f1, w = _transitions(obs, ~obs.held_out)
        d = f0.shape[1]
        self.dev, self.chunk = device, chunk
        self.bits = 2 ** torch.arange(d, device=device, dtype=torch.int64)
        code0 = (torch.tensor(f0, device=device).long() * self.bits).sum(1)
        order = torch.argsort(code0)
        self.f0 = torch.tensor(f0, dtype=torch.float32, device=device)[order]
        self.f1 = torch.tensor(f1, dtype=torch.float32, device=device)[order]
        code0 = code0[order]
        self.codes, first = torch.unique_consecutive(code0, return_inverse=True)
        self.src = torch.zeros(len(self.codes), d, device=device)
        self.src[first] = self.f0
        self.cw = torch.cumsum(torch.tensor(w[order.cpu().numpy()], device=device), 0)
        ends = torch.searchsorted(first, torch.arange(len(self.codes), device=device), right=True)
        self.row_end = self.cw[ends - 1]
        self.row_start = torch.cat([torch.zeros(1, device=device), self.row_end[:-1]])
        starts = np.append(obs.episode_start, len(obs.frames))
        self.ctx = int(np.diff(starts).max())

    def context(self, frames: torch.Tensor) -> torch.Tensor:
        return frames

    def _source(self, h: torch.Tensor) -> torch.Tensor:
        code = (h.long() * self.bits).sum(1)
        idx = torch.searchsorted(self.codes, code).clamp(max=len(self.codes) - 1)
        miss = torch.nonzero(self.codes[idx] != code).flatten()
        for i in range(0, len(miss), self.chunk):
            m = miss[i : i + self.chunk]
            x = h[m]
            dist = x.sum(1, keepdim=True) + self.src.sum(1)[None] - 2 * x @ self.src.T
            idx[m] = dist.argmin(1)
        return idx

    @torch.no_grad()
    def sample_next(self, h: torch.Tensor, generator: torch.Generator | None = None):
        idx = self._source(h)
        lo, hi = self.row_start[idx], self.row_end[idx]
        r = torch.rand(len(h), device=self.dev, generator=generator, dtype=lo.dtype)
        u = lo + r * (hi - lo)
        pos = torch.searchsorted(self.cw, u, right=True).clamp(max=len(self.cw) - 1)
        return (h + self.f1[pos] - self.f0[pos]).clamp(0, 1)


class Decipherer:
    """Everything that depends on one pair's streams and models, but not on the matching."""

    def __init__(self, pair: Path, cfg: DecipherConfig, device: str):
        self.pair, self.cfg, self.dev = Path(pair), cfg, device
        self.lang, obs = load_lang(self.pair / "data"), load_obs(self.pair / "data")
        self.layout: FrameLayout = frame_layout(obs)
        self.types: WordTypes = word_types(self.lang)
        self.V, self.d, self.J = len(self.lang.words), self.layout.size, cfg.sentence_tokens
        # Emissions read a frame through ``view``; for state vectors its units are the values.
        self.U, self.view, self.own_classes = self.d, (lambda f: f), (0, 2, 4)
        if cfg.frame_model == COUNTS:
            self.fm = CountedFrames(obs, device)
            self.ctx = self.fm.ctx
        else:
            self.fm = load_model(self.pair / "models" / cfg.frame_model, device)
            self.fm.use_graph = cfg.sampler_graph
            if self.fm.joint is None:
                raise ValueError("decipherment needs a joint-head frame model (decision 18)")
            for p in self.fm.parameters():
                p.requires_grad_(False)
            self.ctx = self.fm.backbone.cfg.context
        stop = self.lang.words.index(STOP)
        # Share of sentences that advance the frame, from each stream's mean episode length.
        frames_per_ep = len(obs.frames) / len(obs.episode_start)
        sents_per_ep = (self.lang.tokens == stop).sum() / len(self.lang.episode_start)
        self.q = float(1 - (frames_per_ep - 1) / (sents_per_ep - 1))
        self._count_stats(obs)
        self._start_frames()
        self.pairs = None
        if cfg.summary_features == "pairs":
            live = torch.tensor(sorted(self.layout.values), device=device)
            iu = torch.triu_indices(len(live), len(live), 1, device=device)
            self.pairs = live[iu]
        elif cfg.summary_features != "linear":
            raise ValueError(f"unknown summary_features {cfg.summary_features!r}")
        self._actor_tensors()

    # -- raw statistics, for the shortlist ---------------------------------------------------
    def _count_stats(self, obs) -> None:
        f0, f1, w = _transitions(obs, ~obs.held_out)
        on = np.clip(f1 - f0, 0, 1)
        lay = self.layout
        self.obj_changes = np.array([(w[on[:, list(b)].sum(1) > 0]).sum() for b in lay.objects])
        self.arrivals = np.zeros(lay.places)
        for v, (kind, _, p) in lay.values.items():
            if kind == "place":
                self.arrivals[p] += (w * on[:, v]).sum()
        self.word_counts = np.bincount(self.lang.tokens, minlength=self.V).astype(float)

    def count_mismatch(self, m: Matching) -> float:
        """How far apart the streams' frequency profiles are under m (lower is closer)."""

        def share(x):
            x = np.asarray(x, float)
            return np.log(x / x.sum())

        wc = self.word_counts
        return float(
            ((share([wc[w] for w in m.objects]) - share(self.obj_changes)) ** 2).sum()
            + ((share([wc[w] for w in m.places]) - share(self.arrivals)) ** 2).sum()
        )

    def matchings(self) -> list[Matching]:
        """Every matching. With ``agent_orders: fixed`` (the pilot) the agents keep one order, so
        their swap is decided by numbering; with ``all`` every order is a candidate."""
        t = self.types
        if self.cfg.agent_orders == "all":
            agent_orders = [tuple(a) for a in itertools.permutations(t.agents)]
        elif self.cfg.agent_orders == "fixed":
            agent_orders = [tuple(t.agents)]
        else:
            raise ValueError(f"unknown agent_orders {self.cfg.agent_orders!r}")
        return [
            Matching(a, tuple(o), tuple(p))
            for a in agent_orders
            for o in itertools.permutations(t.objects)
            for p in itertools.permutations(t.places)
        ]

    # -- layout as tensors ---------------------------------------------------------------------
    def _start_frames(self) -> None:
        combos = list(itertools.product(*[list(b) for b in self.layout.blocks]))
        S = torch.zeros(len(combos), self.d, device=self.dev)
        for i, c in enumerate(combos):
            S[i, list(c)] = 1
        self.start, self.start_n = S, S / S.sum(1, keepdim=True)
        self.start_u = S

    def _actor_tensors(self) -> None:
        lay = self.layout
        self.acted_v = [a for _, a, _ in lay.agents]
        self.blk_v = [list(b) for b, _, _ in lay.agents]
        self.pid_v = [
            torch.tensor([lay.values[v][2] for v in b], device=self.dev) for b, _, _ in lay.agents
        ]

    def valid(self, c: torch.Tensor) -> torch.Tensor | None:
        """Which candidate frames may be used (None: all). State vectors: all."""
        return None

    def actor_place(self, c: torch.Tensor) -> torch.Tensor:
        """Place id of the acting agent in each frame, or -1 if none acted."""
        out = torch.full((c.shape[0],), -1, device=self.dev, dtype=torch.long)
        for k in range(len(self.acted_v)):
            on = c[:, self.acted_v[k]] > 0.5
            out = torch.where(on, self.pid_v[k][c[:, self.blk_v[k]].argmax(1)], out)
        return out

    def actor_word(self, c: torch.Tensor, places_t: torch.Tensor) -> torch.Tensor:
        p = self.actor_place(c)
        return torch.where(p >= 0, places_t[p.clamp(min=0)], torch.full_like(p, self.V))

    # -- tying ---------------------------------------------------------------------------------
    def tying(self, m: Matching):
        """For every feature: its kind (shared table) and the role class of every word."""
        d, V, lay, t = self.d, self.V, self.layout, self.types
        ent = set(t.agents) | set(t.objects) | set(t.places)
        other = [w for w in range(V) if w not in ent]
        n_cls = 6 + len(other)
        F = 3 * d + 1
        kinds: dict[tuple, int] = {}
        g = np.zeros(F, np.int64)
        K = np.zeros((F, V), np.int64)
        for f in range(F):
            if f == 3 * d:
                desc, typ = ("none", None, None), "none"
            else:
                v = f % d
                typ = ("state", "on", "off")[f // d]
                desc = lay.values.get(v, (f"dead{v}", None, None))
            kind, e, x = desc
            g[f] = kinds.setdefault((typ, kind), len(kinds))
            sa = (
                {m.agents[e]}
                if kind in ("acted", "hungry", "place")
                else ({m.agents[x]} if kind == "obj_held" else set())
            )
            so = {m.objects[e]} if kind.startswith("obj") else set()
            sp = {m.places[x]} if kind in ("place", "obj_at", "obj_broken") else set()
            for w in range(V):
                if w in t.agents:
                    K[f, w] = 0 if w in sa else 1
                elif w in t.objects:
                    K[f, w] = 2 if w in so else 3
                elif w in t.places:
                    K[f, w] = 4 if w in sp else 5
                else:
                    K[f, w] = 6 + other.index(w)
        size = np.stack([np.bincount(K[f], minlength=n_cls) for f in range(F)])
        dev = self.dev
        return (
            torch.tensor(g, device=dev),
            torch.tensor(K, device=dev),
            torch.tensor(size, dtype=torch.float32, device=dev),
            len(kinds),
            n_cls,
        )

    def tie(self, counts: torch.Tensor, f_idx: torch.Tensor, tying) -> torch.Tensor:
        """Pool word counts into role classes per kind, then expand back into words."""
        gs, Ks, ss, G, n_cls = tying
        g, K, size = gs[f_idx], Ks[f_idx], ss[f_idx]
        J = counts.shape[0]
        S = torch.zeros(J, G * n_cls, device=self.dev)
        S.index_add_(1, (g[:, None] * n_cls + K).reshape(-1), counts.reshape(J, -1))
        # Smooth only the role classes that have words for a kind (an object's features have no
        # "this agent" word, for example), so every table sums to exactly 1.
        present = torch.zeros(G * n_cls, device=self.dev)
        present[(g[:, None] * n_cls + K).reshape(-1)] = 1.0
        S = (S.reshape(J, G, n_cls) + self.cfg.smooth) * present.reshape(1, G, n_cls)
        S = S / S.sum(-1, keepdim=True).clamp(min=1e-30)
        rows = torch.arange(len(f_idx), device=self.dev)[:, None]
        return S[:, g[:, None], K] / size[rows, K].clamp(min=1)


@dataclass
class Tables:
    p0: torch.Tensor  # J x V: background word frequencies
    ps: torch.Tensor  # J x d x V: words from a value that is on
    pc: torch.Tensor  # J x (2d+1) x V: words from a change
    lam: torch.Tensor  # J x 4: background, state, change, actor's place
    cs: torch.Tensor | None = None  # last counts, for the tying projection
    cc: torch.Tensor | None = None


class Fit:
    """One matching's model: tables, summary Gaussian, and the particle filter."""

    def __init__(self, D: Decipherer, m: Matching):
        self.D, self.m = D, m
        self.places_t = torch.tensor(m.places, device=D.dev)
        self.tying_ = D.tying(m)
        self.gauss: Gauss | None = None
        self.frozen = False
        J, V, dev = D.J, D.V, D.dev
        wc = D.word_counts.copy()
        wc[[D.lang.words.index(STOP), D.lang.words.index(EPISODE)]] = 0  # never inside a sentence
        unigram = torch.tensor(wc / wc.sum(), dtype=torch.float32, device=dev)
        own = torch.isin(self.tying_[1], torch.tensor(D.own_classes, device=dev)).float()
        gen = torch.Generator(device=dev).manual_seed(D.cfg.seed)
        U = D.U
        noise = 1 + 0.1 * torch.rand(J, 3 * U + 1, V, device=dev, generator=gen)
        cnt = (1 + D.cfg.own_prior * own)[None] * noise
        if D.cfg.rate_prior and hasattr(D, "rate_matches"):
            # Words that are neither entity nor place words (role 4 on), tied by rate to changes,
            # in the sentence positions where each word is said, in proportion (entry 93).
            other = [w for w in range(V) if int(self.tying_[1][0, w]) >= 4]
            share = torch.tensor(D.position_share(), dtype=torch.float32, device=dev)  # J x V
            for f, w in D.rate_matches(other):
                cnt[:, f, w] += D.cfg.rate_prior * share[:, w]
        lam = torch.tensor([0.45, 0.25, 0.2, 0.1], device=dev)[None].repeat(J, 1)
        self.t = Tables(unigram[None].repeat(J, 1), None, None, lam, cnt[:, :U], cnt[:, U:])
        self.project()

    def project(self) -> None:
        D = self.D
        self.t.ps = D.tie(self.t.cs, torch.arange(D.U, device=D.dev), self.tying_)
        self.t.pc = D.tie(self.t.cc, torch.arange(D.U, 3 * D.U + 1, device=D.dev), self.tying_)

    # -- emissions -------------------------------------------------------------------------------
    def feats(self, prev: torch.Tensor, cur: torch.Tensor, views=None):
        """``views``: the unit views of (prev, cur), if already computed."""
        prev, cur = views if views is not None else (self.D.view(prev), self.D.view(cur))
        fs = cur / cur.sum(1, keepdim=True).clamp(min=1)
        on, off = (cur - prev).clamp(min=0), (prev - cur).clamp(min=0)
        dc = torch.cat([on, off, ((on.sum(1) + off.sum(1)) == 0).float()[:, None]], 1)
        return fs, dc / dc.sum(1, keepdim=True)

    def emit_words(
        self, tok, prev, cur, moved, actw, views=None, rows=None, focus=None
    ) -> torch.Tensor:
        """Probability of each sentence given (prev -> cur); ``moved`` says whether it stepped.

        ``views``: the unit views of (prev, cur), if already computed. ``rows``: ``tok`` holds
        each distinct sentence once, and item i reads sentence ``rows[i]``. ``focus``: unused
        here (``SentenceFit``)."""
        V, t = self.D.V, self.t
        if rows is not None:
            tok = tok[rows]
        fs, dc = self.feats(prev, cur, views)
        out = torch.ones(len(tok), device=self.D.dev)
        for j in range(tok.shape[1]):
            l0, ls, lc, la = t.lam[j].tolist()
            w = tok[:, j]
            ok, wc = w < V, w.clamp(max=V - 1)
            ps_w = (fs * t.ps[j][:, wc].T).sum(1)
            pc_w = (dc * t.pc[j][:, wc].T).sum(1)
            mv = l0 * t.p0[j][wc] + ls * ps_w + lc * pc_w + la * (actw == wc).float()
            st = (l0 * t.p0[j][wc] + ls * ps_w) / (l0 + ls)
            out = out * torch.where(ok, torch.where(moved, mv, st), torch.ones_like(mv))
        return out

    def start_tables(self):
        D, t = self.D, self.t
        out = []
        for j in range(D.J):
            l0, ls, _, _ = t.lam[j].tolist()
            s = (l0 * t.p0[j][None] + ls * (D.start_n @ t.ps[j])) / (l0 + ls)
            out.append(torch.cat([s, torch.ones(len(D.start), 1, device=D.dev)], 1))
        return out

    @staticmethod
    def start_emit(TS, tok_t):
        out = torch.ones(tok_t.shape[0], TS[0].shape[0], device=TS[0].device)
        for j in range(tok_t.shape[1]):
            out = out * TS[j][:, tok_t[:, j]].T
        return out

    # -- the particle filter -----------------------------------------------------------------------
    @torch.no_grad()
    def filter(self, b: Batch, gen: torch.Generator, use_summary: bool = True):
        """Per-episode log-likelihood, and each particle's frame and move flag per sentence."""
        D, cfg, dev = self.D, self.D.cfg, self.D.dev
        tok, real = b.tok, b.real
        Tm, Be = real.shape
        N, M, d = cfg.particles, cfg.candidates, D.d
        R = Be * N
        ep = torch.arange(Be, device=dev).repeat_interleave(N)
        ar = torch.arange(R, device=dev)
        G = self.gauss if (use_summary and b.z is not None) else None
        kappa = cfg.summary_power
        TS = self.start_tables()
        e0 = self.start_emit(TS, tok[0])
        ll = torch.log(e0.mean(1))
        tiny = torch.finfo(e0.dtype).tiny
        e0 = torch.nan_to_num(e0, nan=0.0, posinf=0.0) + tiny
        alpha0 = e0 / e0.sum(1, keepdim=True)  # exact odds over start frames, until a first move
        pick = torch.multinomial(alpha0, N, replacement=True, generator=gen).reshape(-1)
        hist = torch.zeros(R, D.ctx, d, device=dev)
        hist[:, 0] = D.start[pick]
        L = torch.ones(R, dtype=torch.long, device=dev)
        traj = torch.zeros(R, Tm, d, dtype=torch.uint8, device=dev)
        traj[:, 0] = hist[:, 0].to(torch.uint8)
        moved_t = torch.zeros(R, Tm, dtype=torch.bool, device=dev)
        inc = cfg.context_cache and hasattr(D.fm, "context_at")
        if inc:
            cache = D.fm.context_cache(R, D.ctx)
            h = D.fm.context_at(cache, hist[:, 0], torch.zeros_like(L), ar)
        else:
            h = D.fm.context(hist[:, :1])[:, 0]
        if G is not None:
            MS = G.mean(D.start_u)
        no = torch.zeros(R, dtype=torch.bool, device=dev)
        # Fixed for the whole pass, so made once (each was a small host-to-GPU copy or a wait
        # for the GPU at every step). The numbers are the same.
        q_real, q_one = torch.tensor(D.q, device=dev), torch.tensor(1.0, device=dev)
        no_actor = torch.full((R,), D.V, device=dev)
        all_moved = torch.ones(R * M, dtype=torch.bool, device=dev)
        ep_m = ep.repeat_interleave(M, 0)  # the episode of each candidate
        edits_on = cfg.edit_proposals and hasattr(D, "edits")
        focus = (self.sig / self.sig.sum()).tolist() if hasattr(self, "sig") else None
        for t in range(1, Tm):
            rt = real[t][ep]
            if not rt.any():
                continue
            un = L == 1
            any_un = bool(un.any())  # one wait for the GPU per step, not five
            if any_un:
                E0 = self.start_emit(TS, tok[t])
                draw = torch.multinomial(alpha0, N, replacement=True, generator=gen).reshape(-1)
                hist[:, 0] = torch.where(un[:, None], D.start[draw], hist[:, 0])
                if inc:
                    idx = torch.nonzero(un).flatten()
                    h = h.clone()
                    h[idx] = D.fm.context_at(cache, hist[idx, 0], torch.zeros_like(idx), idx)
                else:
                    h = torch.where(un[:, None], D.fm.context(hist[:, :1])[:, 0], h)
            cur = hist[ar, L - 1]
            q = torch.where(rt, q_real, q_one)
            q = torch.where(L >= D.ctx, torch.ones_like(q), q)
            # Each frame's unit view, computed once per step (it was computed up to four times).
            u_cur = D.view(cur)
            e_stay = self.emit_words(
                tok[t], cur, cur, no, no_actor, views=(u_cur, u_cur), rows=ep, focus=focus
            )
            cand = D.fm.sample_next(h.repeat_interleave(M, 0), gen)
            ok_c = D.valid(cand)
            aw = D.actor_word(cand, self.places_t)
            u_cand = D.view(cand)
            e_mv = self.emit_words(
                tok[t],
                cur.repeat_interleave(M, 0),
                cand,
                all_moved,
                aw,
                views=(u_cur.repeat_interleave(M, 0), u_cand),
                rows=ep_m,
                focus=focus,
            ).reshape(R, M)
            ne = 0
            if edits_on:
                # Stratified: the edited frames are weighed exactly (frame model probability times
                # emission), and the samples estimate the rest, so a sample equal to an edit is
                # dropped. Unbiased, and rare single changes always get their due weight.
                E, okE = D.edits(cur)  # R x ne x d, R x ne
                ne = E.shape[1]
                Ef = E.reshape(R * ne, d)
                lpE = D.fm.log_prob_next(h.repeat_interleave(ne, 0), Ef).reshape(R, ne)
                vE = D.valid(Ef)
                if vE is not None:
                    okE = okE & vE.reshape(R, ne)
                u_E = D.view(Ef)
                e_E = self.emit_words(
                    tok[t],
                    cur.repeat_interleave(ne, 0),
                    Ef,
                    torch.ones(R * ne, dtype=torch.bool, device=dev),
                    D.actor_word(Ef, self.places_t),
                    views=(u_cur.repeat_interleave(ne, 0), u_E),
                    rows=ep.repeat_interleave(ne, 0),
                    focus=focus,
                ).reshape(R, ne)
                dup = (cand.reshape(R, M, 1, d) == E[:, None]).all(-1) & okE[:, None]
                e_mv = e_mv * (~dup.any(-1)).float()
            E0z = E0 if any_un else None
            if G is not None:
                zt = b.z[t][ep]
                ls_ = kappa * G.logp(u_cur, zt)
                lm_ = kappa * G.logp(u_cand, zt.repeat_interleave(M, 0)).reshape(R, M)
                mx = torch.maximum(
                    ls_.reshape(Be, N).max(1).values, lm_.reshape(Be, -1).max(1).values
                )
                if ne:
                    lE = kappa * G.logp(u_E, zt.repeat_interleave(ne, 0)).reshape(R, ne)
                    mx = torch.maximum(mx, lE.reshape(Be, -1).max(1).values)
                if any_un:
                    r0 = (b.z[t][:, None, :] - MS[None]) @ G.A
                    lp0 = kappa * (-0.5 * (r0 * r0).sum(-1) + G.norm)
                    mx = torch.maximum(mx, lp0.max(1).values)
                    E0z = E0 * torch.exp(lp0 - mx[:, None])
                e_stay = e_stay * torch.exp(ls_ - mx[ep])
                e_mv = e_mv * torch.exp(lm_ - mx[ep][:, None])
                if ne:
                    e_E = e_E * torch.exp(lE - mx[ep][:, None])
                ll = ll + mx * real[t]
            if any_un:
                e_stay = torch.where(un, (alpha0 * E0z).sum(1)[ep], e_stay)
            if ok_c is not None:
                e_mv = e_mv * ok_c.reshape(R, M).float()
            parts = [(q * e_stay)[:, None], (1 - q)[:, None] * e_mv / M]
            if ne:
                parts.append((1 - q)[:, None] * torch.exp(lpE) * e_E * okE.float())
            contrib = torch.cat(parts, 1)
            # The floor must be representable in float32 (1e-300 rounds to 0): without it a sharp
            # summary model can leave a particle with no option of nonzero weight (entry 69).
            contrib = torch.nan_to_num(contrib, nan=0.0, posinf=0.0).clamp(min=0)
            # The floor goes on "stay" only, which is always allowed: on every option it would let
            # a particle move past the frame model's context, or onto an excluded frame.
            contrib[:, 0] = contrib[:, 0] + tiny
            w = torch.where(rt, contrib.sum(1), torch.ones(R, device=dev))
            wb = w.reshape(Be, N)
            ll = ll + torch.log(wb.mean(1).clamp(min=tiny)) * real[t]
            choice = torch.multinomial(contrib, 1, generator=gen)[:, 0]
            mv = (choice > 0) & rt
            pool = cand.reshape(R, M, d) if not ne else torch.cat([cand.reshape(R, M, d), E], 1)
            newf = torch.where(mv[:, None], pool[ar, (choice - 1).clamp(min=0)], cur)
            at = L.clamp(max=D.ctx - 1)
            hist[ar, at] = torch.where(mv[:, None], newf, hist[ar, at])
            first = mv & un
            if first.any():
                traj[first, :t] = hist[first, 0][:, None].to(torch.uint8)
            L = L + mv.long()
            traj[:, t] = newf.to(torch.uint8)
            moved_t[:, t] = mv
            if any_un:
                # Same float32 floor as below: if every start view's weight underflows, the odds
                # would become 0/0 (entry 69).
                alpha0 = torch.nan_to_num(alpha0 * E0z, nan=0.0, posinf=0.0) + tiny
                alpha0 = alpha0 / alpha0.sum(1, keepdim=True)
            wb = torch.nan_to_num(wb, nan=0.0, posinf=0.0) + tiny
            idx = torch.multinomial(
                wb / wb.sum(1, keepdim=True), N, replacement=True, generator=gen
            )
            src = (idx + torch.arange(Be, device=dev)[:, None] * N).reshape(-1)
            hist, L, traj, moved_t = hist[src], L[src], traj[src], moved_t[src]
            if inc:
                cache = [[K[src], V[src]] for K, V in cache]
                h = h[src]
                idx = torch.nonzero(mv[src]).flatten()
                if len(idx):
                    p_ = L[idx] - 1
                    h[idx] = D.fm.context_at(cache, hist[idx, p_], p_, idx)
            else:
                h = D.fm.context(hist[:, : int(L.max())])[ar, L - 1] if mv.any() else h[src]
        return ll, traj, moved_t

    def run(self, batches: list[Batch], seed: int, use_summary: bool = True):
        gen = torch.Generator(device=self.D.dev).manual_seed(seed)
        paths, total, n = [], 0.0, 0
        for b in batches:
            ll, traj, mv = self.filter(b, gen, use_summary)
            total += ll.sum().item()
            n += int(b.real.sum())
            paths.append((b, traj, mv))
        return paths, total / n

    # -- EM update ---------------------------------------------------------------------------------
    def _items(self, paths):
        N = self.D.cfg.particles
        for b, traj, mv in paths:
            Be = b.real.shape[1]
            for k in range(self.D.cfg.update_paths):
                one = torch.arange(Be, device=self.D.dev) * N + k
                tr, m_ = traj[one].float(), mv[one]
                prev = torch.cat([tr[:, :1], tr[:, :-1]], 1)
                keep = b.real.T
                cur = tr[keep]
                yield (
                    b.tok.permute(1, 0, 2)[keep],
                    prev[keep],
                    cur,
                    m_[keep],
                    self.D.actor_word(cur, self.places_t),
                    b.z.permute(1, 0, 2)[keep] if b.z is not None else None,
                )

    def update(self, paths, it: int) -> None:
        D, t = self.D, self.t
        J, V, d, dev = D.J, D.V, D.U, D.dev
        c0 = torch.zeros(J, V, device=dev)
        cs = torch.zeros(J, d, V, device=dev)
        cc = torch.zeros(J, 2 * d + 1, V, device=dev)
        lamc = torch.zeros(J, 4, device=dev)
        X, Y = [], []
        for tok, prev, cur, m, actw, z in self._items(paths):
            fs, dc = self.feats(prev, cur)
            if z is not None:
                X.append(D.view(cur))
                Y.append(z)
            for j in range(J):
                l0, ls, lc, la = t.lam[j].tolist()
                w = tok[:, j]
                ok, wc = w < V, w.clamp(max=V - 1)
                l0_ = torch.where(m, l0, l0 / (l0 + ls))
                ls_ = torch.where(m, ls, ls / (l0 + ls))
                lc_ = torch.where(m, lc, 0.0)
                r0 = l0_ * t.p0[j][wc]
                rs = ls_[:, None] * fs * t.ps[j][:, wc].T
                rc = lc_[:, None] * dc * t.pc[j][:, wc].T
                ra = torch.where(m & (actw == wc), la, 0.0)
                tot = r0 + rs.sum(1) + rc.sum(1) + ra
                okf = ok.float() / tot
                c0[j].index_add_(0, wc, r0 * okf)
                cs[j].index_add_(1, wc, (rs * okf[:, None]).T)
                cc[j].index_add_(1, wc, (rc * okf[:, None]).T)
                mm = (m & ok).float()
                lamc[j] += torch.stack(
                    [
                        (r0 * okf * mm).sum(),
                        (rs.sum(1) * okf * mm).sum(),
                        (rc.sum(1) * okf * mm).sum(),
                        (ra * okf * mm).sum(),
                    ]
                )
        sm = D.cfg.smooth
        t.cs, t.cc = cs, cc
        t.p0 = (c0 + sm) / (c0 + sm).sum(-1, keepdim=True)
        if it < D.cfg.freeze_mixture:
            t.lam = (lamc + 1.0) / (lamc + 1.0).sum(-1, keepdim=True)
        self.project()
        if X and it + 1 >= D.cfg.summary_from:
            self.gauss = Gauss.fit(torch.cat(X), torch.cat(Y), pairs=D.pairs)

    def fit(self, batches: list[Batch], log=print) -> None:
        t0 = time.time()
        for it in range(self.D.cfg.iterations):
            paths, ll = self.run(batches, seed=it + 1000 * self.D.cfg.seed)
            self.update(paths, it)
            log(
                f"    iteration {it + 1}/{self.D.cfg.iterations}: "
                f"fit {ll:.4f} ({time.time() - t0:.0f}s)"
            )
            if self.D.cfg.split_at and it + 1 == self.D.cfg.split_at and self.gauss is not None:
                # The filter's particle paths are no longer needed. Release them and the
                # allocator's cache first: on state vectors the split step otherwise ran with
                # the GPU's memory full and spilling into shared memory (entry 82).
                del paths
                _free_gpu()
                self.split(batches, log)

    def split(self, batches: list[Batch], log=print) -> None:
        """Split moves on the summary Gaussian (align.latent.split_moves): facts it has learned
        no effect for get a chance to start, from the summaries' own distribution and the frame
        stream's prior over unit views. Uses no pairing (entry 69)."""
        from grounding.align.latent import prune, split_moves

        D = self.D
        z = torch.cat([b.z[b.real] for b in batches])
        # The rarest unit views (0.1% of frames) are dropped for speed, as in entries 70-74.
        views, prior = prune(*D.unit_prior())
        cfg = D.cfg
        together = None
        if cfg.split_together:
            # One thing's appearances exclude each other, so their effects may only be placed
            # jointly (entry 72). Read from the frame structure alone.
            looks = [(j, u[1]) for j, u in enumerate(D.units.units) if u[0] == "looks"]
            together = [(a, b) for (a, ta), (b, tb) in itertools.combinations(looks, 2) if ta == tb]
        extra = None
        if cfg.split_looks:
            extra = [j for j, u in enumerate(D.units.units) if u[0] == "looks"]
        out = split_moves(
            z,
            views,
            prior,
            self.gauss,
            weakest=cfg.split_weakest,
            extra=extra,
            rounds=cfg.split_rounds,
            together=together,
            orders=cfg.split_orders,
            log=log,
        ).gauss
        fine = torch.isfinite(out.Wb).all() and torch.isfinite(out.A).all()
        if fine and torch.isfinite(out.norm):
            self.gauss = out

    @torch.no_grad()
    def decode(self, batches: list[Batch], seed: int | None = None):
        """The commonest particle frame at every sentence end, with its token position."""
        bits = 2 ** torch.arange(self.D.d, device=self.D.dev, dtype=torch.int64)
        N = self.D.cfg.particles
        frames, positions = [], []
        seed = 1000 * self.D.cfg.seed if seed is None else seed
        for b, traj, _ in self.run(batches, seed)[0]:
            Tm, Be = b.real.shape
            code = (traj.long() * bits).sum(-1).reshape(Be, N, Tm)
            best = torch.mode(code, 1).values
            fr = ((best[..., None] // bits) % 2).to(torch.uint8).cpu().numpy()
            keep = b.real.T.cpu().numpy()
            for e in range(Be):
                frames.append(fr[e][keep[e]])
                positions.append(b.ends[e])
        return np.concatenate(frames), np.concatenate(positions)

    def state(self) -> dict:
        t = self.t
        return {
            "matching": asdict(self.m),
            "p0": t.p0.cpu(),
            "ps": t.ps.cpu(),
            "pc": t.pc.cpu(),
            "lam": t.lam.cpu(),
            "gauss": None
            if self.gauss is None
            else (
                self.gauss.Wb.cpu(),
                self.gauss.A.cpu(),
                None if self.gauss.pairs is None else self.gauss.pairs.cpu(),
            ),
        }


class SentenceFit(Fit):
    """The word model with each sentence about one fact (experiment log, entry 43).

    ``Fit`` explains every word of a sentence separately, each from any fact in the frame, so
    nothing ties "stone" and "river" in "the stone is at the river" to the same fact: a frame with
    the stone elsewhere and an agent at the river explains the words almost as well. Here a
    sentence first picks one *focus* feature (a value that is on, or one change), and every word
    then comes from that feature's table, the background, or (after a step) the acting agent's
    place:

        P(sentence) = sum_f pi(f) prod_j [mu0 p0_j(w_j) + mu1 T_j(w_j | f) + mua [w_j = A]]

    where A is the word for the acting agent's place.

    pi spreads ``sig[0]`` over the values that are on and ``sig[1]`` over the changes (by their
    share), or puts everything on the values that are on if the sentence didn't step. ``t.lam`` is
    (mu0, mu1, mua) per position. The tables T are the same tied role tables as ``Fit``'s.
    """

    def __init__(self, D: Decipherer, m: Matching):
        super().__init__(D, m)
        self.t.lam = torch.tensor([0.45, 0.45, 0.1], device=D.dev)[None].repeat(D.J, 1)
        self.sig = torch.tensor([0.55, 0.45], device=D.dev)

    def _per_feature(self, tok, actw, moved, rows=None):
        """For each item, feature and position: the three parts of P_j(w_j | f), and pads.

        ``rows``: ``tok`` holds each distinct sentence once and item i reads ``rows[i]``; the
        word look-ups are then done once per sentence and copied, which gives the same numbers."""
        V, t = self.D.V, self.t
        ok = tok < V
        wc = tok.clamp(max=V - 1)
        mu = t.lam  # J x 3
        a = mu[:, 0][None] * torch.stack([t.p0[j][wc[:, j]] for j in range(tok.shape[1])], 1)
        T = torch.stack(
            [
                torch.cat([t.ps[j][:, wc[:, j]], t.pc[j][:, wc[:, j]]], 0).T
                for j in range(tok.shape[1])
            ],
            2,
        )  # n x F x J
        b = mu[:, 1][None, None] * T
        if rows is not None:
            ok, wc, a, b = ok[rows], wc[rows], a[rows], b[rows]
        c = mu[:, 2][None] * (wc == actw[:, None]).float() * moved[:, None].float()
        scale = torch.where(moved[:, None], 1.0, 1.0 / (mu[:, 0] + mu[:, 1])[None])  # n x J
        return a, b, c, scale, ok

    def _focus(self, prev, cur, moved, views=None, focus=None):
        """``focus``: the two shares of ``sig``, if already read (they are fixed during a pass)."""
        fs, dc = self.feats(prev, cur, views)
        s0, s1 = focus if focus is not None else (self.sig / self.sig.sum()).tolist()
        pi_mv = torch.cat([s0 * fs, s1 * dc], 1)
        pi_st = torch.cat([fs, torch.zeros_like(dc)], 1)
        return torch.where(moved[:, None], pi_mv, pi_st)

    def _weights(self, tok, prev, cur, moved, actw, views=None, rows=None, focus=None):
        a, b, c, scale, ok = self._per_feature(tok, actw, moved, rows)
        P = (a[:, None] + b + c[:, None]) * scale[:, None]  # n x F x J
        P = torch.where(ok[:, None], P, torch.ones_like(P))
        pi = self._focus(prev, cur, moved, views, focus)
        return pi * P.prod(-1), (a, b, c, scale, ok, P)

    def emit_words(
        self, tok, prev, cur, moved, actw, views=None, rows=None, focus=None
    ) -> torch.Tensor:
        return self._weights(tok, prev, cur, moved, actw, views, rows, focus)[0].sum(1)

    def start_tables(self):
        """Per position: each value's stay probability of each word (d x V+1, pad column 1)."""
        D, t = self.D, self.t
        out = []
        for j in range(D.J):
            m0, m1, _ = t.lam[j].tolist()
            s = (m0 * t.p0[j][None] + m1 * t.ps[j]) / (m0 + m1)
            out.append(torch.cat([s, torch.ones(D.U, 1, device=D.dev)], 1))
        return out

    def start_emit(self, TS, tok_t):
        G = torch.ones(tok_t.shape[0], self.D.U, device=self.D.dev)
        for j in range(tok_t.shape[1]):
            G = G * TS[j][:, tok_t[:, j]].T
        return G @ self.D.start_n.T  # Be x K: the average over each start frame's values

    def update(self, paths, it: int) -> None:
        D, t = self.D, self.t
        J, V, d, dev = D.J, D.V, D.U, D.dev
        F = 3 * d + 1
        c0 = torch.zeros(J, V, device=dev)
        cT = torch.zeros(J, F, V, device=dev)
        lamc = torch.zeros(J, 3, device=dev)
        sigc = torch.zeros(2, device=dev)
        X, Y = [], []
        for tok, prev, cur, m, actw, z in self._items(paths):
            if z is not None:
                X.append(D.view(cur))
                Y.append(z)
            w, (a, b, c, scale, ok, P) = self._weights(tok, prev, cur, m, actw)
            r = w / w.sum(1, keepdim=True).clamp(min=1e-300)  # n x F: which feature is the focus
            r = torch.nan_to_num(r)
            sigc += torch.stack([r[m][:, :d].sum(), r[m][:, d:].sum()])
            Pn = torch.where(ok[:, None], (a[:, None] + b + c[:, None]), torch.ones_like(P))
            rb = r[..., None] * b / Pn  # n x F x J: word from the focus's table
            ra = (r[..., None] * a[:, None] / Pn).sum(1)  # n x J: background
            rc = (r[..., None] * c[:, None] / Pn).sum(1)
            okf = ok.float()
            wc = tok.clamp(max=V - 1)
            for j in range(J):
                c0[j].index_add_(0, wc[:, j], ra[:, j] * okf[:, j])
                cT[j].index_add_(1, wc[:, j], (rb[:, :, j] * okf[:, j, None]).T)
                mm = (m & ok[:, j]).float()
                lamc[j] += torch.stack(
                    [(ra[:, j] * mm).sum(), (rb[:, :, j].sum(1) * mm).sum(), (rc[:, j] * mm).sum()]
                )
        sm = D.cfg.smooth
        t.cs, t.cc = cT[:, :d], cT[:, d:]
        t.p0 = (c0 + sm) / (c0 + sm).sum(-1, keepdim=True)
        if it < D.cfg.freeze_mixture:
            t.lam = (lamc + 1.0) / (lamc + 1.0).sum(-1, keepdim=True)
            self.sig = (sigc + 1.0) / (sigc + 1.0).sum()
        self.project()
        if X and it + 1 >= D.cfg.summary_from:
            self.gauss = Gauss.fit(torch.cat(X), torch.cat(Y), pairs=D.pairs)

    def state(self) -> dict:
        return {**super().state(), "sig": self.sig.cpu()}


# ---------------------------------------------------------------------------------------------
# Grid video: frames read through units found from the frame stream alone
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GridMatching:
    """Which word names each thing found in the frames, and each place."""

    things: tuple[int, ...]
    places: tuple[int, ...]


class GridDecipherer(Decipherer):
    """Decipherment through units found from the frame stream alone (``grounding.align.units``):
    grid video always, and state vectors with ``structure: statistics``.

    Emissions and the summary Gaussian read each frame as units: a thing at a place, a thing's
    appearance, two things in one cell, a thing gone. A matching names the things and the places,
    with no frame-side notion of agent or object: any entity word may name any thing, so the
    agents' order is free. Nothing marks who acted in grid video, so there is no actor component.
    """

    def __init__(self, pair: Path, cfg: DecipherConfig, device: str):
        self.pair, self.cfg, self.dev = Path(pair), cfg, device
        self.lang, obs = load_lang(self.pair / "data"), load_obs(self.pair / "data")
        if cfg.grid_units not in ("split", "joint"):
            raise ValueError(f"unknown grid_units {cfg.grid_units!r}")
        if obs.shape is None:
            self.units = find_units_vector(obs)
        else:
            self.units = find_units(obs, split=cfg.grid_units == "split")
        self.view = UnitView(self.units, device)
        n_things, n_places = len(self.units.things), self.units.places
        if cfg.word_kinds == "classes":
            self.classes = word_classes(self.lang)
            self.roles = role_options(self.classes, n_things, n_places) or role_options(
                self.classes, n_things - 1, n_places
            )  # allowing one thing that no word names, such as a mark of who acted (entry 66)
        elif cfg.word_kinds == "rules":
            self.types: WordTypes = word_types(self.lang)
            t = self.types
            self.roles = [Roles(tuple(t.agents) + tuple(t.objects), tuple(t.places))]
        else:
            raise ValueError(f"unknown word_kinds {cfg.word_kinds!r}")
        if not self.roles:
            raise ValueError(f"no word classes fit {n_things} things and {n_places} places")
        self.V, self.d, self.J = len(self.lang.words), self.units.size, cfg.sentence_tokens
        self.U, self.own_classes = self.units.n, (0, 2)
        if cfg.frame_model == COUNTS:
            raise ValueError("the counted-transitions control isn't implemented for grid video yet")
        self.fm = load_model(self.pair / "models" / cfg.frame_model, device)
        self.fm.use_graph = cfg.sampler_graph
        for p in self.fm.parameters():
            p.requires_grad_(False)
        self.ctx = self.fm.backbone.cfg.context
        stop = self.lang.words.index(STOP)
        frames_per_ep = len(obs.frames) / len(obs.episode_start)
        sents_per_ep = (self.lang.tokens == stop).sum() / len(self.lang.episode_start)
        self.q = float(1 - (frames_per_ep - 1) / (sents_per_ep - 1))
        self._grid_counts(obs)
        self._grid_starts(obs)
        self.pairs = None
        if cfg.summary_features == "pairs":
            self.pairs = torch.triu_indices(self.U, self.U, 1, device=device)
        elif cfg.summary_features != "linear":
            raise ValueError(f"unknown summary_features {cfg.summary_features!r}")

    def _grid_counts(self, obs) -> None:
        """Raw frequency profiles for the shortlist: how often each thing changes, and how often
        something arrives at each place."""
        f0, f1, w = _transitions(obs, ~obs.held_out)
        u0 = self.view(torch.tensor(f0, dtype=torch.float32, device=self.dev))
        u1 = self.view(torch.tensor(f1, dtype=torch.float32, device=self.dev))
        on = (u1 - u0).clamp(min=0)
        wt = torch.tensor(w, dtype=torch.float32, device=self.dev)
        units = self.units.units
        self.thing_changes = np.zeros(len(self.units.things))
        self.arrivals = np.zeros(self.units.places)
        for j, u in enumerate(units):
            if u[0] == "at":
                self.arrivals[u[2]] += float((wt * on[:, j]).sum())
        # A thing changes when its place changes or it disappears: the definition that ranks the
        # right reading of the word classes first on every pilot pair, in both formats (entry 67).
        changed = (u0 != u1).float()
        for k in range(len(self.units.things)):
            cols = [j for j, u in enumerate(units) if u[0] in ("at", "gone") and u[1] == k]
            self.thing_changes[k] = float((wt * (changed[:, cols].sum(1) > 0).float()).sum())
        self.word_counts = np.bincount(self.lang.tokens, minlength=self.V).astype(float)
        # How often each unit turns on and off per frame step (for the rate prior, entry 93).
        tot = float(wt.sum())
        off = (u0 - u1).clamp(min=0)
        self.change_rate = (
            (wt[:, None] * on).sum(0).cpu().numpy() / tot,
            (wt[:, None] * off).sum(0).cpu().numpy() / tot,
        )

    def position_share(self) -> np.ndarray:
        """For each word, the share of its occurrences at each sentence position (J x V), from
        the language stream alone, with sentences cut as ``make_batches`` cuts them."""
        if getattr(self, "_pos_share", None) is None:
            w = self.lang.words
            t = self.lang.tokens[self.lang.tokens != w.index(EPISODE)]
            stop = t == w.index(STOP)
            idx = np.arange(len(t))
            last = np.maximum.accumulate(np.where(stop, idx, -1))
            prev = np.concatenate([[-1], last[:-1]])
            pos = idx - prev - 1
            keep = ~stop & (pos < self.J)
            c = np.zeros((self.J, self.V))
            np.add.at(c, (pos[keep], t[keep]), 1.0)
            self._pos_share = c / np.maximum(c.sum(0, keepdims=True), 1.0)
        return self._pos_share

    def rate_matches(
        self, words, tol: float = 0.1, close: float = 0.05, kinds=("looks", "gone")
    ) -> list[tuple[int, int]]:
        """(feature, word) pairs for the rate prior, from each stream's own statistics.

        Every change is reported in the language, so a word that reports a kind of change is said
        about as often, per frame step, as that change happens, summed over the things it can
        happen to. Changes of one kind (a unit type turning on, or off) are grouped when their
        rates are within ``close`` of each other, and each word is tied to every group whose
        summed rate is within ``tol`` of its own (in log terms). Sentences are converted to frame
        steps with the episodes' lengths alone (1 - q). Only appearance changes and disappearances
        by default: rates alone also match places and carrying by coincidence (entry 93)."""
        U = self.U
        n_sent = float((self.lang.tokens == self.lang.words.index(STOP)).sum())
        per_step = self.word_counts / n_sent / max(1e-9, 1 - self.q)
        groups = []
        for typ, rates in ((1, self.change_rate[0]), (2, self.change_rate[1])):
            for kind in kinds:
                js = sorted(
                    (j for j, u in enumerate(self.units.units) if u[0] == kind and rates[j] > 0),
                    key=lambda j: rates[j],
                )
                cur: list[int] = []
                for j in js:
                    if cur and rates[j] > rates[cur[0]] * (1 + close):
                        groups.append(([typ * U + i for i in cur], float(rates[cur].sum())))
                        cur = []
                    cur.append(j)
                if cur:
                    groups.append(([typ * U + i for i in cur], float(rates[cur].sum())))
        out = []
        for w in words:
            if per_step[w] <= 0:
                continue
            for fs, r in groups:
                if abs(np.log(per_step[w] / r)) < np.log(1 + tol):
                    out.extend((f, w) for f in fs)
        return out

    def _grid_starts(self, obs) -> None:
        """Start frames seen in the frame stream's training episodes, one per distinct unit view."""
        starts = obs.frames[obs.episode_start[~obs.held_out]]
        S = torch.tensor(np.unique(starts, axis=0), dtype=torch.float32, device=self.dev)
        Su = self.view(S)
        _, first = np.unique(Su.cpu().numpy(), axis=0, return_index=True)
        first = torch.tensor(np.sort(first), device=self.dev)
        self.start, self.start_u = S[first], Su[first]
        self.start_n = self.start_u / self.start_u.sum(1, keepdim=True)

    def count_mismatch(self, m: GridMatching) -> float:
        def share(x):
            x = np.asarray(x, float)
            return np.log(x / x.sum())

        wc = self.word_counts
        return float(
            ((share([wc[w] for w in m.things]) - share(self.thing_changes)) ** 2).sum()
            + ((share([wc[w] for w in m.places]) - share(self.arrivals)) ** 2).sum()
        )

    def matchings(self) -> list[GridMatching]:
        """Every matching of words to things and places, for every way the word classes could name
        them, with the words in every order (so the agents' order is never fixed)."""
        return [
            GridMatching(tuple(t), tuple(p))
            for r in self.roles
            for t in itertools.permutations(r.things)
            for p in itertools.permutations(r.places)
        ]

    def valid(self, c: torch.Tensor) -> torch.Tensor | None:
        return self.view.valid(c) if self.cfg.valid_only else None

    def edits(self, cur: torch.Tensor):
        """For each frame, the frames one appearance change or one disappearance away, and
        which of them exist (entry 94). Read from the frame structure alone: a thing's appearance
        is the channel it is drawn in, and changing it keeps the cell."""
        if getattr(self, "_edit_plan", None) is None:
            n_ch = self.units.shape[2]
            plan = []
            for k, vals in enumerate(self.units.things):
                chans = sorted({v % n_ch for v in vals})
                member = torch.zeros(self.d, dtype=torch.bool, device=self.dev)
                member[list(vals)] = True
                vt = torch.tensor(sorted(vals), device=self.dev)
                for c2 in chans if len(chans) > 1 else []:
                    plan.append((vt, member, c2))
                if k in self.view.gone:
                    plan.append((vt, member, None))
            self._edit_plan = plan
        n_ch = self.units.shape[2]
        R = cur.shape[0]
        out, ok = [], []
        rows = torch.arange(R, device=self.dev)
        for vt, member, c2 in self._edit_plan:
            on = cur[:, vt] > 0.5  # R x |vals|
            present = on.any(1)
            v = vt[on.float().argmax(1)]  # the thing's drawn value (where present)
            f = cur.clone()
            if c2 is None:  # the thing disappears
                f[:, vt] = 0
                good = present
            else:  # the thing takes appearance c2 in the same cell
                v2 = (v // n_ch) * n_ch + c2
                good = present & (v % n_ch != c2) & member[v2]
                f[rows, v] = 0
                f[rows, v2] = 1
            out.append(f)
            ok.append(good)
        return torch.stack(out, 1), torch.stack(ok, 1)

    def actor_word(self, c: torch.Tensor, places_t: torch.Tensor) -> torch.Tensor:
        return torch.full((c.shape[0],), self.V, device=self.dev, dtype=torch.long)

    def unit_prior(self):
        """Distinct unit views among the frame stream's training frames, and the share of each."""
        if getattr(self, "_prior", None) is None:
            from grounding.align.latent import unit_prior

            obs = load_obs(self.pair / "data")
            ep = np.searchsorted(obs.episode_start, np.arange(len(obs.frames)), side="right") - 1
            fr = obs.frames[~obs.held_out[ep]]
            U = [
                self.view(torch.tensor(fr[i : i + 65536], dtype=torch.float32, device=self.dev))
                for i in range(0, len(fr), 65536)
            ]
            self._prior = unit_prior(torch.cat(U))
        return self._prior

    def tying(self, m: GridMatching):
        """For every unit feature: its kind (shared table) and the role class of every word.
        Roles: 0 a word naming one of the unit's things, 1 another entity word, 2 the word for the
        unit's place, 3 another place word, then one class per other word."""
        U, V = self.U, self.V
        ents, places = set(m.things), set(m.places)
        other = [w for w in range(V) if w not in ents and w not in places]
        n_cls = 4 + len(other)
        F = 3 * U + 1
        kinds: dict[tuple, int] = {}
        g = np.zeros(F, np.int64)
        K = np.zeros((F, V), np.int64)
        for f in range(F):
            if f == 3 * U:
                typ, desc = "none", ("none",)
            else:
                typ, desc = ("state", "on", "off")[f // U], self.units.units[f % U]
            if desc[0] == "at":
                key, things, place = (typ, "at", desc[1], *desc[3:]), {desc[1]}, desc[2]
            elif desc[0] == "looks":
                key, things, place = (typ, "looks", desc[1], desc[2]), {desc[1]}, None
            elif desc[0] == "with":
                key, things, place = (typ, "with", desc[1], desc[2]), {desc[1], desc[2]}, None
            elif desc[0] == "gone":
                key, things, place = (typ, "gone", desc[1]), {desc[1]}, None
            else:
                key, things, place = (typ,), set(), None
            g[f] = kinds.setdefault(key, len(kinds))
            mine = {m.things[k] for k in things if m.things[k] >= 0}
            here = None if place is None else m.places[place]
            for w in range(V):
                if w in ents:
                    K[f, w] = 0 if w in mine else 1
                elif w in places:
                    K[f, w] = 2 if w == here else 3
                else:
                    K[f, w] = 4 + other.index(w)
        size = np.stack([np.bincount(K[f], minlength=n_cls) for f in range(F)])
        dev = self.dev
        return (
            torch.tensor(g, device=dev),
            torch.tensor(K, device=dev),
            torch.tensor(size, dtype=torch.float32, device=dev),
            len(kinds),
            n_cls,
        )


class GridFitMixin:
    """Decoding for grid video. Particles that show the same state can differ in which cell each
    thing sits in, so the commonest *unit view* is taken, returned as one particle frame with it."""

    @torch.no_grad()
    def decode(self, batches: list[Batch], seed: int | None = None):
        D = self.D
        bits = 2 ** torch.arange(D.U, device=D.dev, dtype=torch.int64)
        N = D.cfg.particles
        frames, positions = [], []
        seed = 1000 * D.cfg.seed if seed is None else seed
        for b, traj, _ in self.run(batches, seed)[0]:
            Tm, Be = b.real.shape
            flat = traj.reshape(-1, D.d).float()
            units = D.view(flat).round().clamp(0, 1).long()
            code = (units * bits).sum(-1).reshape(Be, N, Tm)
            best = torch.mode(code, 1).values  # Be x Tm
            pick = (code == best[:, None]).float().argmax(1)  # first particle with that view
            tr = traj.reshape(Be, N, Tm, D.d)
            e_idx = torch.arange(Be, device=D.dev)[:, None]
            t_idx = torch.arange(Tm, device=D.dev)[None]
            fr = tr[e_idx, pick, t_idx].cpu().numpy()
            keep = b.real.T.cpu().numpy()
            for e in range(Be):
                frames.append(fr[e][keep[e]])
                positions.append(b.ends[e])
        return np.concatenate(frames), np.concatenate(positions)


class GridSentenceFit(GridFitMixin, SentenceFit):
    pass


class GridWordFit(GridFitMixin, Fit):
    pass


def make_fit(D: Decipherer, m) -> Fit:
    grid = isinstance(D, GridDecipherer)
    if D.cfg.emission == "sentence":
        return (GridSentenceFit if grid else SentenceFit)(D, m)
    if D.cfg.emission != "word":
        raise ValueError(f"unknown emission {D.cfg.emission!r}")
    return (GridWordFit if grid else Fit)(D, m)


# ---------------------------------------------------------------------------------------------
# The two-stage search
# ---------------------------------------------------------------------------------------------

DECIPHER_PART = "decipher"


def _free_gpu() -> None:
    """Release a finished fit's GPU memory. Without this, a grid search slowed about 3.5 times
    after its first full fit, as the GPU's memory filled (experiment log, entry 65)."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sample(lang: LangStream, root: int, cfg: DecipherConfig) -> dict[str, np.ndarray]:
    rng = part_rngs(root, "align", ALIGN_PARTS)[DECIPHER_PART]
    train = np.flatnonzero(~lang.held_out)
    fit = rng.choice(train, cfg.train_episodes, replace=False)
    sel = rng.choice(np.setdiff1d(train, fit), cfg.select_episodes, replace=False)
    held = np.flatnonzero(lang.held_out)
    score = np.sort(rng.choice(held, min(cfg.score_episodes, len(held)), replace=False))
    return {"fit": fit, "select": sel, "score": score}


def decipher_pair(
    pair: Path, cfg: DecipherConfig, device: str | None = None, name: str = "decipher", log=print
) -> dict:
    pair = Path(pair)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = pair / "align" / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if cfg.structure not in ("rules", "statistics"):
        raise ValueError(f"unknown structure {cfg.structure!r}")
    grid = load_obs(pair / "data").shape is not None or cfg.structure == "statistics"
    D = (GridDecipherer if grid else Decipherer)(pair, cfg, dev)
    if grid:
        u = D.units
        log(
            f"structure: {len(u.things)} things, {u.places} places, {u.n} units "
            f"({D.d} frame values); word kinds by {cfg.word_kinds}: {len(D.roles)} ways to name "
            f"them; {len(D.start)} start views; stay probability {D.q:.3f}"
        )
    else:
        ty = D.types
        lay = D.layout
        log(
            f"structure: {len(lay.agents)} agents, {len(lay.objects)} objects, {lay.places} "
            f"places, {len(lay.values)} live frame values; words: {len(ty.agents)} agents, "
            f"{len(ty.objects)} objects, {len(ty.places)} places, {len(ty.verbs)} verbs; "
            f"stay probability {D.q:.3f}"
        )
    root = read_record(pair / "data").seed
    eps = sample(D.lang, root, cfg)
    lm = load_model(pair / "models" / cfg.lang_model, dev)
    batches = {k: make_batches(D.lang, v, cfg, lm, dev) for k, v in eps.items()}
    mu, sd = standardise(batches["fit"])
    standardise(batches["select"], mu, sd)
    standardise(batches["score"], mu, sd)
    del lm
    log(f"language batches and summaries ready ({time.time() - t0:.0f}s)")

    ranked = sorted(D.matchings(), key=D.count_mismatch)
    screened = []
    if cfg.screen:
        pool = ranked[: cfg.screen]
        n_b = max(1, cfg.screen_episodes // cfg.batch)
        it = cfg.screen_iterations
        D.cfg = replace(
            cfg, iterations=it, summary_from=max(1, it // 2), freeze_mixture=max(1, it - 1)
        )
        log(f"screening {len(pool)} matchings: {it} iterations on {n_b} batches each")
        for j, m in enumerate(pool):
            f = make_fit(D, m)
            f.fit(batches["fit"][:n_b], log=lambda _s: None)
            sc = f.run(batches["select"], seed=11)[1]
            del f
            _free_gpu()
            screened.append(
                {"matching": asdict(m), "count_mismatch": D.count_mismatch(m), "score": sc}
            )
            log(f"  screen {j + 1}/{len(pool)}: {sc:.4f} ({time.time() - t0:.0f}s)")
        D.cfg = cfg
        order = np.argsort([-r["score"] for r in screened])
        finalists = [pool[int(i)] for i in order[: cfg.shortlist]]
    else:
        finalists = ranked[: cfg.shortlist]
    log(
        "finalists' raw-count mismatch: "
        + "; ".join(f"{D.count_mismatch(m):.4f}" for m in finalists)
    )
    rows = []
    for i, m in enumerate(finalists):
        log(f"finalist {i + 1}/{len(finalists)}: fitting")
        f = make_fit(D, m)
        f.fit(batches["fit"], log)
        own = [f.run(batches["select"], seed=s)[1] for s in (11, 12)]
        frames, positions = f.decode(batches["score"])
        torch.save(f.state(), out / f"finalist-{i}.pt")
        np.savez_compressed(out / f"finalist-{i}.npz", frames=frames, positions=positions)
        rows.append(
            {
                "finalist": i,
                "matching": asdict(m),
                "count_mismatch": D.count_mismatch(m),
                "score": float(np.mean(own)),
                "score_seeds": own,
            }
        )
        del f
        _free_gpu()
        log(
            f"finalist {i + 1}/{len(finalists)}: selection score {np.mean(own):.4f} "
            f"(seeds {own[0]:.4f}, {own[1]:.4f}) ({time.time() - t0:.0f}s)"
        )
    selected = int(np.argmax([r["score"] for r in rows]))
    summary = {
        "selected": selected,
        "finalists": rows,
        "screened": screened,
        "stay_probability": D.q,
        "words": list(D.lang.words),
        "minutes": (time.time() - t0) / 60,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_record(out, make_record({"decipher": asdict(cfg)}, root))
    log(f"selected finalist {selected + 1} ({summary['minutes']:.0f} minutes)")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("pair", type=Path)
    p.add_argument("--config", type=Path, default=Path("configs/pilot-decipher.yaml"))
    p.add_argument("--device", default=None)
    p.add_argument("--name", default="decipher", help="results folder under align/")
    args = p.parse_args()
    decipher_pair(
        args.pair,
        DecipherConfig.from_yaml(args.config),
        args.device,
        args.name,
        log=lambda s: print(s, flush=True),
    )


if __name__ == "__main__":
    main()
