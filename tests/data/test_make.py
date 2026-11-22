from grounding.data.make import parse_pairs


def test_parse_pairs():
    assert parse_pairs("3") == [3]
    assert parse_pairs("0-4") == [0, 1, 2, 3, 4]
