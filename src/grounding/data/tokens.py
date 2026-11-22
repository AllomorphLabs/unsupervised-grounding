"""Token numbering for the language stream.

Tokens are morphemes (technical decision 2) plus two markers: ``<ep>`` starts
an episode and ``<.>`` ends a sentence. Token numbers are assigned in a
random order per pair (technical decision 11), so nothing about a token's
number says which concept it stands for.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from grounding.render.lang.grammar import Sentence
from grounding.render.lang.vocab import Vocab

EPISODE = "<ep>"
STOP = "<.>"


@dataclass(frozen=True)
class Tokens:
    """``words[i]`` is the token with number ``i``."""

    words: tuple[str, ...]

    @cached_property
    def index(self) -> dict[str, int]:
        return {w: i for i, w in enumerate(self.words)}

    def encode_episode(self, sentences: list[Sentence]) -> list[int]:
        ids = [self.index[EPISODE]]
        for s in sentences:
            ids.extend(self.index[m] for m in s)
            ids.append(self.index[STOP])
        return ids


def make_tokens(vocab: Vocab, rng: np.random.Generator) -> Tokens:
    words = [EPISODE, STOP, *vocab.stems.values(), *("-" + s for s in vocab.suffixes.values())]
    if len(set(words)) != len(words):
        raise ValueError("token strings must be distinct")
    return Tokens(tuple(words[i] for i in rng.permutation(len(words))))
