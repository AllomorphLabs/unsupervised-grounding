"""Named world specs."""

from fractions import Fraction

from grounding.sim.world import Habit, World

PLACES = ("river", "bush", "cave", "hill")
RIVER, BUSH, CAVE, HILL = range(4)
GLASS, STONE, APPLE = range(3)


def pilot(habit: Fraction | int | str | None = None, feeding: bool = True) -> World:
    """The pilot world: two interchangeable agents, distinguishable objects and places.

    Places lie on a path, river - bush - cave - hill. On its own the path has a
    mirror symmetry; feeding at the bush breaks it. The objects differ in
    their rules: glass breaks when dropped, the apple can be eaten, the stone
    does neither.

    ``habit`` gives agent A that many times the usual preference for going to
    the river, breaking the agent symmetry. Pass a ``Fraction`` or a string
    such as ``"5/4"`` for graded habits; a float would not be exact.
    ``habit=1`` keeps the symmetry. ``feeding=False`` removes feeding
    at the bush, restoring the mirror symmetry of the places.
    """
    return World(
        agents=("A", "B"),
        places=PLACES,
        edges=((RIVER, BUSH), (BUSH, CAVE), (CAVE, HILL)),
        objects=("glass", "stone", "apple"),
        fragile=frozenset({GLASS}),
        edible=frozenset({APPLE}),
        feeding=frozenset({BUSH}) if feeding else frozenset(),
        habits=() if habit is None else (Habit(agent=0, place=RIVER, weight=Fraction(habit)),),
    )


def tiny(habit: Fraction | int | str | None = None) -> World:
    """A cut-down pilot world for fast tests: three places, glass and stone.

    Places lie on a path, river - bush - cave, with feeding at the river to
    break the mirror. Its symmetry group is the agent swap, as in the pilot.
    ``habit`` works as in ``pilot``.
    """
    return World(
        agents=("A", "B"),
        places=PLACES[:3],
        edges=((RIVER, BUSH), (BUSH, CAVE)),
        objects=("glass", "stone"),
        fragile=frozenset({GLASS}),
        feeding=frozenset({RIVER}),
        habits=() if habit is None else (Habit(agent=0, place=RIVER, weight=Fraction(habit)),),
    )


WORLDS = {"pilot": pilot, "tiny": tiny}


def make_world(name: str, **args) -> World:
    """Build a named world, e.g. ``make_world("pilot", habit="5/4")``."""
    if name not in WORLDS:
        raise ValueError(f"unknown world {name!r}; expected one of {sorted(WORLDS)}")
    return WORLDS[name](**args)
