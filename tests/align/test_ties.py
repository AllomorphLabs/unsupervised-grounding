"""Decision 11: entity order can't bias the swap rate.

A stand-in aligner that faces a perfect tie between the two agents and breaks it
the way real solvers do, by index order: it pairs the lower-numbered agent token
with the lower-numbered agent slot. With the per-pair shuffles, that choice is
right about half the time across pairs. Without them, it would be right every
time, and the swap rate would be 0% for reasons unrelated to the models.
"""

import json

from grounding.analysis.labels import load_key
from grounding.data.config import CorpusConfig
from grounding.data.corpus import generate_pair

SMALL = CorpusConfig(world="tiny", episodes=1, min_len=1, max_len=1)


def index_order_swaps(words: list[str], stems: dict[str, str], slots: list[int]) -> bool:
    """Does breaking the tie by index order pair agent A with agent B's slot?"""
    a_token_first = words.index(stems["A"]) < words.index(stems["B"])
    a_slot_first = slots[0] < slots[1]  # slots[i] is the slot holding agent i
    return a_token_first != a_slot_first


def test_shuffles_make_index_order_tie_breaking_a_coin_toss(tmp_path):
    n = 60
    swaps = 0
    for k in range(n):
        out = generate_pair(SMALL, k, tmp_path / str(k))
        key = load_key(out)
        words = json.loads((out / "lang_tokens.json").read_text())
        swaps += index_order_swaps(words, key["lang"]["stems"], key["obs_slots"]["agents"])
    assert 15 < swaps < 45  # about 30 of 60 expected


def test_without_shuffles_tie_breaking_never_swaps():
    stems = {"A": "zibon", "B": "movek"}
    unshuffled_words = ["<ep>", "<.>", "zibon", "movek"]  # concept order
    assert not index_order_swaps(unshuffled_words, stems, [0, 1])
