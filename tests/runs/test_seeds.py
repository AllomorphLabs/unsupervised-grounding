import numpy as np
import pytest

from grounding.runs.seeds import STREAMS, stream_rng, stream_seeds


def draw(root: int, stream: str) -> np.ndarray:
    return stream_rng(root, stream).integers(0, 2**32, size=16)


def test_same_root_reproduces_each_stream():
    for stream in STREAMS:
        assert np.array_equal(draw(7, stream), draw(7, stream))


def test_streams_differ_within_a_pair():
    assert not np.array_equal(draw(7, "lang"), draw(7, "obs"))


def test_roots_differ():
    assert not np.array_equal(draw(7, "lang"), draw(8, "lang"))


def test_every_stream_has_a_seed():
    assert set(stream_seeds(0)) == set(STREAMS)


def test_unknown_stream_rejected():
    with pytest.raises(ValueError):
        stream_rng(0, "video")


def test_parts_are_reproducible_and_distinct():
    from grounding.runs.seeds import part_rngs

    def draws(root, stream):
        return {
            k: g.integers(0, 2**32, 8).tolist()
            for k, g in part_rngs(root, stream, ("a", "b")).items()
        }

    assert draws(3, "lang") == draws(3, "lang")
    assert draws(3, "lang")["a"] != draws(3, "lang")["b"]
    assert draws(3, "lang")["a"] != draws(3, "obs")["a"]
    # Appending a part leaves earlier parts unchanged.
    longer = part_rngs(3, "lang", ("a", "b", "c"))["a"].integers(0, 2**32, 8).tolist()
    assert longer == draws(3, "lang")["a"]
