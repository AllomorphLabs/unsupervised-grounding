"""Kinds of word, found from the language stream alone (phase 3, step 2).

This replaces the pilot's rules for telling agents, objects, places and verbs apart, which were
written for the pilot language's grammar (``structure.word_types``). Here nothing about the
grammar is assumed:

- Each token is described by what comes just before it and just after it: two distributions over
  the token list, from the training episodes.
- Tokens are clustered by how alike those descriptions are (Jensen-Shannon divergence, average
  linkage). Words that can stand in for each other, such as the names of two places, end up
  together.
- The clustering is cut at its largest jump in merge distance, the natural break between words
  that are interchangeable and words that are not.

The result is a set of *classes*: groups of two or more tokens. Which classes name the things and
the places seen in the frames is not decided here. ``role_options`` lists every way the classes
could cover them, by size alone, and decipherment's search decides among them.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from grounding.data.load import LangStream


@dataclass(frozen=True)
class WordClasses:
    classes: tuple[tuple[int, ...], ...]  # groups of two or more interchangeable tokens
    heights: tuple[float, ...]  # every merge distance, in order
    cut: float  # merges at or below this distance were kept


def _js(p: np.ndarray, q: np.ndarray) -> float:
    m = (p + q) / 2

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        nz = a > 0
        return float((a[nz] * np.log(a[nz] / b[nz])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def word_classes(lang: LangStream, max_tokens: int = 3_000_000) -> WordClasses:
    per_token = np.repeat(lang.held_out, np.diff(np.append(lang.episode_start, len(lang.tokens))))
    toks = lang.tokens[~per_token][:max_tokens].astype(np.int64)
    V = len(lang.words)
    nxt = np.zeros((V, V))
    prv = np.zeros((V, V))
    np.add.at(nxt, (toks[:-1], toks[1:]), 1)
    np.add.at(prv, (toks[1:], toks[:-1]), 1)
    seen = (nxt.sum(1) > 0) & (prv.sum(1) > 0)
    X = (
        np.concatenate(
            [
                nxt / np.maximum(nxt.sum(1, keepdims=True), 1),
                prv / np.maximum(prv.sum(1, keepdims=True), 1),
            ],
            1,
        )
        / 2
    )
    idx = np.flatnonzero(seen)
    D = np.array([[_js(X[i], X[j]) for j in idx] for i in idx])

    def linkage(stop: float | None):
        cl = [[i] for i in range(len(idx))]
        heights = []
        while len(cl) > 1:
            best = None
            for a in range(len(cl)):
                for b in range(a + 1, len(cl)):
                    d = D[np.ix_(cl[a], cl[b])].mean()
                    if best is None or d < best[0]:
                        best = (d, a, b)
            d, a, b = best
            if stop is not None and d > stop:
                break
            heights.append(float(d))
            cl[a] = cl[a] + cl[b]
            del cl[b]
        return cl, heights

    _, heights = linkage(None)
    h = np.array(heights)
    jump = int(np.argmax(np.diff(h)))  # the largest gap between consecutive merges
    cut = float(h[jump])
    groups, _ = linkage(cut)
    classes = tuple(
        tuple(sorted(int(idx[i]) for i in g)) for g in sorted(groups, key=min) if len(g) > 1
    )
    return WordClasses(classes, tuple(heights), cut)


@dataclass(frozen=True)
class Roles:
    """One way the word classes could name what the frames show."""

    things: tuple[int, ...]  # the words that may name things (one class or a union)
    places: tuple[int, ...]  # the words that may name places


def role_options(wc: WordClasses, n_things: int, n_places: int) -> list[Roles]:
    """Every choice of one class for the places and a disjoint union of classes for the things,
    whose sizes match the numbers of places and things found in the frames."""
    out = []
    for pc in wc.classes:
        if len(pc) != n_places:
            continue
        rest = [c for c in wc.classes if c != pc]
        for k in range(1, len(rest) + 1):
            for combo in combinations(rest, k):
                words = tuple(sorted(w for c in combo for w in c))
                if len(words) == n_things:
                    out.append(Roles(words, pc))
    return out
