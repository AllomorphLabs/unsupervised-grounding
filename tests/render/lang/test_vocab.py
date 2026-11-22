import re

import numpy as np
import pytest

from grounding.render.lang.grammar import ROLES, inflections
from grounding.render.lang.vocab import (
    Vocab,
    check_vocab,
    common_words,
    edit_distance,
    edits1,
    generate_vocab,
    load_glosses,
    problems_with,
)
from grounding.sim.worlds import pilot

CVCVC = re.compile(r"[bdfgklmnprstvz][aeiou][bdfgklmnprstvz][aeiou][bdfgklmnprstvz]")


@pytest.fixture(scope="module")
def table():
    return inflections(pilot())


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("glass", "glass") == 0
    assert edit_distance("", "abc") == 3


def test_edits1_matches_edit_distance():
    for w in edits1("cave"):
        assert edit_distance("cave", w) <= 1


def test_common_list_has_twenty_thousand_words():
    assert len(common_words()) <= 20_000
    assert {"river", "glass", "the"} <= common_words()


def test_english_and_near_english_forms_are_rejected():
    glosses = load_glosses()
    assert problems_with("river", glosses["river"], common_words())
    assert problems_with("rivet", (), common_words())  # one edit from common words
    assert problems_with("glasu", glosses["glass"], frozenset())  # two edits from "glass"
    assert not problems_with("glasuvo", glosses["glass"], frozenset())


def test_every_concept_with_a_meaning_has_glosses(table):
    agents = set(pilot().agents)
    missing = [c for c in table if c not in agents and c not in load_glosses()]
    assert missing == []


@pytest.mark.parametrize("seed", range(5))
def test_generated_vocabularies_pass_the_check(table, seed):
    vocab = generate_vocab(table, ROLES, np.random.default_rng(seed))
    assert check_vocab(vocab) == []
    assert all(CVCVC.fullmatch(s) for s in vocab.stems.values())
    assert set(vocab.suffixes) == set(ROLES)


def test_generation_is_reproducible(table):
    a = generate_vocab(table, ROLES, np.random.default_rng(7))
    b = generate_vocab(table, ROLES, np.random.default_rng(7))
    assert a == b


def test_check_catches_english_and_duplicates():
    vocab = Vocab(
        stems={"glass": "glass", "stone": "mazok", "apple": "mazok"},
        suffixes={"obj": "in"},
        inflections={"glass": ("obj",), "stone": ("obj",), "apple": ("obj",)},
    )
    problems = check_vocab(vocab)
    assert any("'glass'" in p for p in problems)
    assert "two concepts share a stem" in problems


def test_inflected_forms_are_checked_too():
    # The stem passes, but stem + suffix spells a common English word.
    bare = Vocab(stems={"stone": "diamet"}, suffixes={"obj": "er"}, inflections={"stone": ()})
    assert check_vocab(bare) == []
    inflected = Vocab(
        stems={"stone": "diamet"}, suffixes={"obj": "er"}, inflections={"stone": ("obj",)}
    )
    assert any("diameter" in p for p in check_vocab(inflected))
