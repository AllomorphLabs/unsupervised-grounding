"""Seeding for independent data streams.

Each corpus pair has one root seed. The language and observation streams get
their own child seeds, derived from it, so the two are statistically
independent but the whole pair can be regenerated from the root alone.
"""

import numpy as np

# Order matters: a stream's index fixes its child seed. Only ever append.
# "align" seeds alignment restarts; it is not a data stream.
STREAMS = ("lang", "obs", "align")


def stream_seeds(root: int) -> dict[str, np.random.SeedSequence]:
    """Return one independent child seed per stream, derived from ``root``."""
    children = np.random.SeedSequence(root).spawn(len(STREAMS))
    return dict(zip(STREAMS, children, strict=True))


def stream_rng(root: int, stream: str) -> np.random.Generator:
    """Return a random generator for one stream of the pair with root seed ``root``."""
    if stream not in STREAMS:
        raise ValueError(f"unknown stream {stream!r}; expected one of {STREAMS}")
    return np.random.default_rng(stream_seeds(root)[stream])


def part_rngs(root: int, stream: str, parts: tuple[str, ...]) -> dict[str, np.random.Generator]:
    """Independent generators for named parts of one stream (vocabulary, simulation, ...).

    A part's generator depends only on its position in ``parts``, so keep each
    stream's tuple fixed and only ever append to it.
    """
    if stream not in STREAMS:
        raise ValueError(f"unknown stream {stream!r}; expected one of {STREAMS}")
    seq = stream_seeds(root)[stream]
    return {
        name: np.random.default_rng(
            np.random.SeedSequence(seq.entropy, spawn_key=(*seq.spawn_key, i))
        )
        for i, name in enumerate(parts)
    }
