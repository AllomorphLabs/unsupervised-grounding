from collections import Counter

import numpy as np
import pytest

from grounding.render.lang.grammar import VERBS, Language, make_vocab, surface
from grounding.render.lang.vocab import Vocab
from grounding.sim.symmetry import Relabel
from grounding.sim.world import AT, BROKEN, GONE, HELD, Event, State, sample_initial, step
from grounding.sim.worlds import GLASS, RIVER, pilot

AGENT_SWAP = Relabel((1, 0), (0, 1, 2), (0, 1, 2, 3))
SCRAMBLE = Relabel((1, 0), (2, 0, 1), (3, 0, 2, 1))  # not a symmetry: any relabelling works


@pytest.fixture(scope="module")
def world():
    return pilot()


@pytest.fixture(scope="module")
def lang(world):
    return Language(world, make_vocab(world, np.random.default_rng(0)))


def simulate(world, seed: int, n: int = 40) -> tuple[State, list[tuple[Event, State]]]:
    rng = np.random.default_rng(seed)
    start = s = sample_initial(world, rng)
    steps = []
    for _ in range(n):
        e, s = step(world, s, rng)
        steps.append((e, s))
    return start, steps


def relabel_vocab(world, vocab: Vocab, r: Relabel) -> dict[str, str]:
    """Token map taking the stem of each entity to the stem of its image under ``r``."""
    m = {}
    for names, perm in (
        (world.agents, r.agents),
        (world.objects, r.objects),
        (world.places, r.places),
    ):
        for i, j in enumerate(perm):
            m[vocab.stems[names[i]]] = vocab.stems[names[j]]
    return m


def test_sentence_patterns(lang):
    v, s = lang.vocab.stems, lang.vocab.suffixes
    e = Event("drop", 0, obj=GLASS, place=RIVER)
    assert lang.event(e) == (
        v["A"],
        v["glass"],
        "-" + s["obj"],
        v["river"],
        "-" + s["at"],
        v["drop"],
        "-" + s["now"],
    )
    assert lang.event(Event("wait", 1), "past") == (v["B"], v["wait"], "-" + s["past"])


def test_surface_joins_suffixes(lang):
    v, s = lang.vocab.stems, lang.vocab.suffixes
    assert surface(lang.event(Event("wait", 1))) == f"{v['B']} {v['wait']}{s['now']}"


def test_side_effects_are_said(world, lang):
    v = lang.vocab.stems
    before = State((RIVER, RIVER), (False, False), ((HELD, 0), (AT, RIVER), (AT, RIVER)))
    after = State((RIVER, RIVER), (True, False), ((BROKEN, RIVER), (AT, RIVER), (AT, RIVER)))
    said = lang.side_effects(before, after)
    assert [x[:2] for x in said] == [(v["glass"], v["break"]), (v["A"], v["hungry"])]


@pytest.mark.parametrize("r", [AGENT_SWAP, SCRAMBLE])
def test_relabelling_the_world_only_relabels_the_sentences(world, lang, r):
    # The renderer check: rendering a relabelled episode gives the original sentences with
    # entity words swapped, and nothing else changes. The intro is shuffled, so compare it
    # as a multiset.
    start, steps = simulate(world, seed=1)
    original = lang.episode(start, steps, np.random.default_rng(5), recap=0.3)
    moved = lang.episode(
        r.state(start),
        [(r.event(e), r.state(s)) for e, s in steps],
        np.random.default_rng(5),
        recap=0.3,
    )
    m = relabel_vocab(world, lang.vocab, r)
    expected = [tuple(m.get(tok, tok) for tok in sent) for sent in original]
    n_intro = len(world.agents) + len(world.objects)
    assert Counter(moved[:n_intro]) == Counter(expected[:n_intro])
    assert moved[n_intro:] == expected[n_intro:]


def test_intro_order_does_not_favour_any_entity(world, lang):
    start, steps = simulate(world, seed=2, n=1)
    a = lang.vocab.stems["A"]
    first = Counter(
        lang.episode(start, steps, np.random.default_rng(i))[0][0] == a for i in range(500)
    )
    # A is one of five intro sentences, so it should come first about 1 time in 5.
    assert 60 < first[True] < 140


def test_recaps_are_past_tense_and_repeat_earlier_events(world, lang):
    start, steps = simulate(world, seed=3)
    past = "-" + lang.vocab.suffixes["past"]
    assert not any(
        past in s for s in lang.episode(start, steps, np.random.default_rng(0), recap=0.0)
    )
    every = lang.episode(start, steps, np.random.default_rng(0), recap=1.0)
    recaps = [s for s in every if s[-1] == past]
    assert len(recaps) == len(steps) - 1
    earlier = {lang.event(e, "past") for e, _ in steps}
    assert set(recaps) <= earlier


def replay(world, lang, sentences) -> list[State]:
    """Rebuild the state after every step from the sentences alone, without the world's rules."""
    concept = {stem: c for c, stem in lang.vocab.stems.items()}
    role = {"-" + suf: r for r, suf in lang.vocab.suffixes.items()}
    agents, objects, places = (
        {n: i for i, n in enumerate(names)} for names in (world.agents, world.objects, world.places)
    )
    pos = [None] * len(agents)
    loc = [None] * len(objects)
    hungry = [False] * len(agents)
    states: list[State] = []

    def snapshot():
        states.append(State(tuple(pos), tuple(hungry), tuple(loc)))

    first_event = True
    for sent in sentences:
        words = [(concept[t], None) if t in concept else (None, role[t]) for t in sent]
        # Attach suffix roles to the preceding stem.
        parsed: list[tuple[str, str | None]] = []
        for c, r in words:
            if c is not None:
                parsed.append((c, None))
            else:
                parsed[-1] = (parsed[-1][0], r)
        (verb, tense), subj, rest = parsed[-1], parsed[0][0], parsed[1:-1]
        assert verb in VERBS
        if tense == "past":
            continue
        args = dict((r, c) for c, r in rest)
        if verb == "be":
            if subj in agents:
                pos[agents[subj]] = places[args["at"]]
            else:
                loc[objects[subj]] = (AT, places[args["at"]])
            continue
        if verb in ("break", "hungry"):
            if verb == "break":
                loc[objects[subj]] = (BROKEN, loc[objects[subj]][1])
            else:
                hungry[agents[subj]] = True
            continue
        if not first_event:
            snapshot()
        first_event = False
        a = agents[subj]
        match verb:
            case "go":
                pos[a] = places[args["to"]]
            case "take":
                loc[objects[args["obj"]]] = (HELD, a)
            case "drop":
                loc[objects[args["obj"]]] = (AT, places[args["at"]])
            case "eat":
                hungry[a] = False
                if "obj" in args:
                    loc[objects[args["obj"]]] = (GONE, -1)
    snapshot()
    return states


@pytest.mark.parametrize("seed", range(3))
def test_the_sentences_determine_every_state(world, lang, seed):
    # Every change is said: the language alone recovers the full state after every step.
    start, steps = simulate(world, seed=seed, n=60)
    sentences = lang.episode(start, steps, np.random.default_rng(seed), recap=0.3)
    assert replay(world, lang, sentences) == [s for _, s in steps]
