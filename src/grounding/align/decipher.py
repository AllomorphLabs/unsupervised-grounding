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
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from grounding.align.run import ALIGN_PARTS
from grounding.align.structure import FrameLayout, WordTypes, _transitions, frame_layout, word_types
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
    iterations: int = 12
    own_prior: float = 20.0  # pseudo-counts on each value's own entity words, to start
    smooth: float = 0.01
    summary_power: float = 0.2
    summary_features: str = "linear"  # "linear": the frame; "pairs": also products of live values
    emission: str = "sentence"  # "word": each word from any fact; "sentence": all from one fact
    summary_from: int = 5  # iteration from which the summaries count
    freeze_mixture: int = 8  # iteration from which the mixture weights stay fixed
    update_paths: int = 8  # particle paths per episode used in each update
    shortlist: int = 6

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
        code0 = torch.tensor(f0, device=device) @ self.bits
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
        code = h.long() @ self.bits
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
        if cfg.frame_model == COUNTS:
            self.fm = CountedFrames(obs, device)
            self.ctx = self.fm.ctx
        else:
            self.fm = load_model(self.pair / "models" / cfg.frame_model, device)
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
        """Every matching, with agents in a fixed order (their swap is a symmetry)."""
        t = self.types
        return [
            Matching(t.agents, tuple(o), tuple(p))
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

    def _actor_tensors(self) -> None:
        lay = self.layout
        self.acted_v = [a for _, a, _ in lay.agents]
        self.blk_v = [list(b) for b, _, _ in lay.agents]
        self.pid_v = [
            torch.tensor([lay.values[v][2] for v in b], device=self.dev) for b, _, _ in lay.agents
        ]

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
        J, V, d, dev = D.J, D.V, D.d, D.dev
        wc = D.word_counts.copy()
        wc[[D.lang.words.index(STOP), D.lang.words.index(EPISODE)]] = 0  # never inside a sentence
        unigram = torch.tensor(wc / wc.sum(), dtype=torch.float32, device=dev)
        own = ((self.tying_[1] == 0) | (self.tying_[1] == 2) | (self.tying_[1] == 4)).float()
        gen = torch.Generator(device=dev).manual_seed(0)
        noise = 1 + 0.1 * torch.rand(J, 3 * d + 1, V, device=dev, generator=gen)
        cnt = (1 + D.cfg.own_prior * own)[None] * noise
        lam = torch.tensor([0.45, 0.25, 0.2, 0.1], device=dev)[None].repeat(J, 1)
        self.t = Tables(unigram[None].repeat(J, 1), None, None, lam, cnt[:, :d], cnt[:, d:])
        self.project()

    def project(self) -> None:
        D = self.D
        self.t.ps = D.tie(self.t.cs, torch.arange(D.d, device=D.dev), self.tying_)
        self.t.pc = D.tie(self.t.cc, torch.arange(D.d, 3 * D.d + 1, device=D.dev), self.tying_)

    # -- emissions -------------------------------------------------------------------------------
    @staticmethod
    def feats(prev: torch.Tensor, cur: torch.Tensor):
        fs = cur / cur.sum(1, keepdim=True).clamp(min=1)
        on, off = (cur - prev).clamp(min=0), (prev - cur).clamp(min=0)
        dc = torch.cat([on, off, ((on.sum(1) + off.sum(1)) == 0).float()[:, None]], 1)
        return fs, dc / dc.sum(1, keepdim=True)

    def emit_words(self, tok, prev, cur, moved, actw) -> torch.Tensor:
        """Probability of each sentence given (prev -> cur); ``moved`` says whether it stepped."""
        V, t = self.D.V, self.t
        fs, dc = self.feats(prev, cur)
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
        alpha0 = e0 / e0.sum(1, keepdim=True)  # exact odds over start frames, until a first move
        pick = torch.multinomial(alpha0, N, replacement=True, generator=gen).reshape(-1)
        hist = torch.zeros(R, D.ctx, d, device=dev)
        hist[:, 0] = D.start[pick]
        L = torch.ones(R, dtype=torch.long, device=dev)
        traj = torch.zeros(R, Tm, d, dtype=torch.uint8, device=dev)
        traj[:, 0] = hist[:, 0].to(torch.uint8)
        moved_t = torch.zeros(R, Tm, dtype=torch.bool, device=dev)
        h = D.fm.context(hist[:, :1])[:, 0]
        if G is not None:
            MS = G.mean(D.start)
        no = torch.zeros(R, dtype=torch.bool, device=dev)
        for t in range(1, Tm):
            rt = real[t][ep]
            if not rt.any():
                continue
            tk = tok[t][ep]
            un = L == 1
            if un.any():
                E0 = self.start_emit(TS, tok[t])
                draw = torch.multinomial(alpha0, N, replacement=True, generator=gen).reshape(-1)
                hist[un, 0] = D.start[draw[un]]
                h = torch.where(un[:, None], D.fm.context(hist[:, :1])[:, 0], h)
            cur = hist[ar, L - 1]
            q = torch.where(rt, torch.tensor(D.q, device=dev), torch.tensor(1.0, device=dev))
            q = torch.where(L >= D.ctx, torch.ones_like(q), q)
            e_stay = self.emit_words(tk, cur, cur, no, torch.full((R,), D.V, device=dev))
            cand = D.fm.sample_next(h.repeat_interleave(M, 0), gen)
            aw = D.actor_word(cand, self.places_t)
            e_mv = self.emit_words(
                tk.repeat_interleave(M, 0),
                cur.repeat_interleave(M, 0),
                cand,
                torch.ones(R * M, dtype=torch.bool, device=dev),
                aw,
            ).reshape(R, M)
            E0z = E0 if un.any() else None
            if G is not None:
                zt = b.z[t][ep]
                ls_ = kappa * G.logp(cur, zt)
                lm_ = kappa * G.logp(cand, zt.repeat_interleave(M, 0)).reshape(R, M)
                mx = torch.maximum(
                    ls_.reshape(Be, N).max(1).values, lm_.reshape(Be, -1).max(1).values
                )
                if un.any():
                    r0 = (b.z[t][:, None, :] - MS[None]) @ G.A
                    lp0 = kappa * (-0.5 * (r0 * r0).sum(-1) + G.norm)
                    mx = torch.maximum(mx, lp0.max(1).values)
                    E0z = E0 * torch.exp(lp0 - mx[:, None])
                e_stay = e_stay * torch.exp(ls_ - mx[ep])
                e_mv = e_mv * torch.exp(lm_ - mx[ep][:, None])
                ll = ll + mx * real[t]
            if un.any():
                e_stay = torch.where(un, (alpha0 * E0z).sum(1)[ep], e_stay)
            contrib = torch.cat([(q * e_stay)[:, None], (1 - q)[:, None] * e_mv / M], 1)
            contrib = torch.nan_to_num(contrib, nan=0.0, posinf=0.0).clamp(min=0) + 1e-300
            w = torch.where(rt, contrib.sum(1), torch.ones(R, device=dev))
            wb = w.reshape(Be, N)
            ll = ll + torch.log(wb.mean(1).clamp(min=1e-300)) * real[t]
            choice = torch.multinomial(contrib, 1, generator=gen)[:, 0]
            mv = (choice > 0) & rt
            newf = torch.where(
                mv[:, None], cand.reshape(R, M, d)[ar, (choice - 1).clamp(min=0)], cur
            )
            at = L.clamp(max=D.ctx - 1)
            hist[ar, at] = torch.where(mv[:, None], newf, hist[ar, at])
            first = mv & un
            if first.any():
                traj[first, :t] = hist[first, 0][:, None].to(torch.uint8)
            L = L + mv.long()
            traj[:, t] = newf.to(torch.uint8)
            moved_t[:, t] = mv
            if un.any():
                alpha0 = alpha0 * E0z
                alpha0 = alpha0 / alpha0.sum(1, keepdim=True)
            wb = torch.nan_to_num(wb, nan=0.0, posinf=0.0) + 1e-300
            idx = torch.multinomial(
                wb / wb.sum(1, keepdim=True), N, replacement=True, generator=gen
            )
            src = (idx + torch.arange(Be, device=dev)[:, None] * N).reshape(-1)
            hist, L, traj, moved_t = hist[src], L[src], traj[src], moved_t[src]
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
        J, V, d, dev = D.J, D.V, D.d, D.dev
        c0 = torch.zeros(J, V, device=dev)
        cs = torch.zeros(J, d, V, device=dev)
        cc = torch.zeros(J, 2 * d + 1, V, device=dev)
        lamc = torch.zeros(J, 4, device=dev)
        X, Y = [], []
        for tok, prev, cur, m, actw, z in self._items(paths):
            fs, dc = self.feats(prev, cur)
            if z is not None:
                X.append(cur)
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
            paths, ll = self.run(batches, seed=it)
            self.update(paths, it)
            log(
                f"    iteration {it + 1}/{self.D.cfg.iterations}: "
                f"fit {ll:.4f} ({time.time() - t0:.0f}s)"
            )

    @torch.no_grad()
    def decode(self, batches: list[Batch], seed: int = 0):
        """The commonest particle frame at every sentence end, with its token position."""
        bits = 2 ** torch.arange(self.D.d, device=self.D.dev, dtype=torch.int64)
        N = self.D.cfg.particles
        frames, positions = [], []
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

    def _per_feature(self, tok, actw, moved):
        """For each item, feature and position: the three parts of P_j(w_j | f), and pads."""
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
        c = mu[:, 2][None] * (wc == actw[:, None]).float() * moved[:, None].float()
        scale = torch.where(moved[:, None], 1.0, 1.0 / (mu[:, 0] + mu[:, 1])[None])  # n x J
        return a, b, c, scale, ok

    def _focus(self, prev, cur, moved):
        fs, dc = self.feats(prev, cur)
        s0, s1 = (self.sig / self.sig.sum()).tolist()
        pi_mv = torch.cat([s0 * fs, s1 * dc], 1)
        pi_st = torch.cat([fs, torch.zeros_like(dc)], 1)
        return torch.where(moved[:, None], pi_mv, pi_st)

    def _weights(self, tok, prev, cur, moved, actw):
        a, b, c, scale, ok = self._per_feature(tok, actw, moved)
        P = (a[:, None] + b + c[:, None]) * scale[:, None]  # n x F x J
        P = torch.where(ok[:, None], P, torch.ones_like(P))
        return self._focus(prev, cur, moved) * P.prod(-1), (a, b, c, scale, ok, P)

    def emit_words(self, tok, prev, cur, moved, actw) -> torch.Tensor:
        return self._weights(tok, prev, cur, moved, actw)[0].sum(1)

    def start_tables(self):
        """Per position: each value's stay probability of each word (d x V+1, pad column 1)."""
        D, t = self.D, self.t
        out = []
        for j in range(D.J):
            m0, m1, _ = t.lam[j].tolist()
            s = (m0 * t.p0[j][None] + m1 * t.ps[j]) / (m0 + m1)
            out.append(torch.cat([s, torch.ones(D.d, 1, device=D.dev)], 1))
        return out

    def start_emit(self, TS, tok_t):
        G = torch.ones(tok_t.shape[0], self.D.d, device=self.D.dev)
        for j in range(tok_t.shape[1]):
            G = G * TS[j][:, tok_t[:, j]].T
        return G @ self.D.start_n.T  # Be x K: the average over each start frame's values

    def update(self, paths, it: int) -> None:
        D, t = self.D, self.t
        J, V, d, dev = D.J, D.V, D.d, D.dev
        F = 3 * d + 1
        c0 = torch.zeros(J, V, device=dev)
        cT = torch.zeros(J, F, V, device=dev)
        lamc = torch.zeros(J, 3, device=dev)
        sigc = torch.zeros(2, device=dev)
        X, Y = [], []
        for tok, prev, cur, m, actw, z in self._items(paths):
            if z is not None:
                X.append(cur)
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


def make_fit(D: Decipherer, m: Matching) -> Fit:
    if D.cfg.emission == "sentence":
        return SentenceFit(D, m)
    if D.cfg.emission != "word":
        raise ValueError(f"unknown emission {D.cfg.emission!r}")
    return Fit(D, m)


# ---------------------------------------------------------------------------------------------
# The two-stage search
# ---------------------------------------------------------------------------------------------

DECIPHER_PART = "decipher"


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
    D = Decipherer(pair, cfg, dev)
    lay, ty = D.layout, D.types
    log(
        f"structure: {len(lay.agents)} agents, {len(lay.objects)} objects, {lay.places} places, "
        f"{len(lay.values)} live frame values; words: {len(ty.agents)} agents, {len(ty.objects)} "
        f"objects, {len(ty.places)} places, {len(ty.verbs)} verbs; stay probability {D.q:.3f}"
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
    finalists = ranked[: cfg.shortlist]
    log("shortlist by raw counts: " + "; ".join(f"{D.count_mismatch(m):.4f}" for m in finalists))
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
        log(
            f"finalist {i + 1}/{len(finalists)}: selection score {np.mean(own):.4f} "
            f"(seeds {own[0]:.4f}, {own[1]:.4f}) ({time.time() - t0:.0f}s)"
        )
    selected = int(np.argmax([r["score"] for r in rows]))
    summary = {
        "selected": selected,
        "finalists": rows,
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
