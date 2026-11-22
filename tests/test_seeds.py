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
