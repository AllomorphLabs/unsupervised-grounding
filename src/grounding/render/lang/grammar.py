"""The invented language's grammar: from episodes to sentences.

Sentences are subject-object-verb (SOV), with suffixes marking each noun's role
and each verb's tense. A sentence is a tuple of morphemes, each of which becomes
one token (technical decision 2). Suffix morphemes start with ``-``.

Roles (suffixes):

- ``obj``: the thing acted on ("the glass" in "A takes the glass")
- ``to``: where something goes
- ``at``: where something is or happens
- ``now``: present tense, for what happens at this step
- ``past``: past tense, for recaps of earlier events

Sentence patterns (glossed in English):

    A  river-to     go-now              A goes to the river
    A  glass-obj    take-now            A takes the glass
    A  glass-obj    river-at  drop-now  A drops the glass at the river
    A  bush-at      eat-now             A eats at the bush
    A  apple-obj    eat-now             A eats the apple
    A  wait-now                         A waits
    glass  break-now                    The glass breaks
    A  hungry-now                       A becomes hungry
    A  river-at  be-now                 A is at the river

An episode opens by saying where every agent and object starts, in random
order. Each step then gives the event, followed by any side effects (something
breaking, an agent becoming hungry), so every change in the state is said.
After a step, with probability ``recap``, a past-tense sentence repeats an
earlier event from the same episode, chosen uniformly.

Random choices never depend on which entity is involved, so relabelling the
world relabels the sentences and changes nothing else (``tests/render/lang``).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from grounding.render.lang.vocab import Vocab, generate_vocab
from grounding.sim.world import AT, BROKEN, Event, State, World

ROLES = ("obj", "to", "at", "now", "past")
VERBS = ("be", "go", "take", "drop", "eat", "wait", "break", "hungry")

Sentence = tuple[str, ...]
KINDS = ("intro", "event", "effect", "recap")


class Said(NamedTuple):
    """What a sentence reports: its kind, the step it belongs to (-1 for the intro), and,
    for a recap, the earlier step it repeats (-1 otherwise)."""

    kind: str
    step: int
    of_step: int


def inflections(world: World) -> dict[str, tuple[str, ...]]:
    """Which suffixes each concept can take."""
    out: dict[str, tuple[str, ...]] = {}
    for a in world.agents:
        out[a] = ()  # agents are only ever subjects, which are unmarked
    for o in world.objects:
        out[o] = ("obj",)
    for p in world.places:
        out[p] = ("to", "at")
    for v in VERBS:
        out[v] = ("now", "past")
    if len(out) != len(world.agents) + len(world.objects) + len(world.places) + len(VERBS):
        raise ValueError("entity names must be distinct from each other and from verbs")
    return out


def make_vocab(world: World, rng: np.random.Generator) -> Vocab:
    return generate_vocab(inflections(world), ROLES, rng)


def surface(sentence: Sentence) -> str:
    """The written form, with suffixes joined to their stems: for people, not models."""
    words: list[str] = []
    for m in sentence:
        if m.startswith("-"):
            words[-1] += m[1:]
        else:
            words.append(m)
    return " ".join(words)


@dataclass(frozen=True)
class Language:
    world: World
    vocab: Vocab

    def _word(self, concept: str, role: str | None = None) -> list[str]:
        stem = self.vocab.stems[concept]
        return [stem] if role is None else [stem, "-" + self.vocab.suffixes[role]]

    def _agent(self, i: int) -> list[str]:
        return self._word(self.world.agents[i])

    def _object(self, i: int, role: str | None = None) -> list[str]:
        return self._word(self.world.objects[i], role)

    def _place(self, i: int, role: str) -> list[str]:
        return self._word(self.world.places[i], role)

    def event(self, e: Event, tense: str = "now") -> Sentence:
        subj = self._agent(e.agent)
        verb = self._word(e.verb, tense)
        match e.verb:
            case "go":
                middle = self._place(e.place, "to")
            case "take":
                middle = self._object(e.obj, "obj")
            case "drop":
                middle = self._object(e.obj, "obj") + self._place(e.place, "at")
            case "eat" if e.obj is not None:
                middle = self._object(e.obj, "obj")
            case "eat":
                middle = self._place(e.place, "at")
            case "wait":
                middle = []
            case _:
                raise ValueError(f"no sentence pattern for event {e}")
        return tuple(subj + middle + verb)

    def side_effects(self, before: State, after: State) -> list[Sentence]:
        """Changes not stated by the event itself. At most one of each kind per step."""
        out = []
        for o, (b, a) in enumerate(zip(before.loc, after.loc, strict=True)):
            if a[0] == BROKEN and b[0] != BROKEN:
                out.append(tuple(self._object(o) + self._word("break", "now")))
        for i, (b, a) in enumerate(zip(before.hungry, after.hungry, strict=True)):
            if a and not b:
                out.append(tuple(self._agent(i) + self._word("hungry", "now")))
        return out

    def intro(self, start: State) -> list[Sentence]:
        """Where every agent and object starts, in a fixed order (shuffled by the caller)."""
        be = self._word("be", "now")
        out = [tuple(self._agent(i) + self._place(p, "at") + be) for i, p in enumerate(start.pos)]
        for o, (kind, p) in enumerate(start.loc):
            if kind != AT:
                raise ValueError("episodes must start with every object intact at a place")
            out.append(tuple(self._object(o) + self._place(p, "at") + be))
        return out

    def episode(
        self,
        start: State,
        steps: Sequence[tuple[Event, State]],
        rng: np.random.Generator,
        recap: float = 0.1,
    ) -> list[Sentence]:
        """Render one episode: the intro, then each step with side effects and recaps."""
        return [s for s, _ in self.labelled_episode(start, steps, rng, recap)]

    def labelled_episode(
        self,
        start: State,
        steps: Sequence[tuple[Event, State]],
        rng: np.random.Generator,
        recap: float = 0.1,
    ) -> list[tuple[Sentence, Said]]:
        """As ``episode``, with what each sentence reports. The labels are ground truth."""
        intro = self.intro(start)
        out = [(intro[i], Said("intro", -1, -1)) for i in rng.permutation(len(intro))]
        before = start
        events: list[Event] = []
        for t, (e, after) in enumerate(steps):
            out.append((self.event(e), Said("event", t, -1)))
            out.extend((s, Said("effect", t, -1)) for s in self.side_effects(before, after))
            # Draw on every step, so the random stream never depends on the content.
            if rng.random() < recap and t > 0:
                j = int(rng.integers(t))
                out.append((self.event(events[j], "past"), Said("recap", t, j)))
            events.append(e)
            before = after
        return out
