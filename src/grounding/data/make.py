"""Build corpus pairs from a config.

    uv run python -m grounding.data.make configs/pilot.yaml --pairs 0-9

Pair ``k`` uses root seed ``k`` and is written to
``runs/<config name>/pair-<k>/data/``.
"""

import argparse
from pathlib import Path

from grounding.data.config import CorpusConfig
from grounding.data.corpus import generate_pair


def parse_pairs(spec: str) -> list[int]:
    """``"3"`` -> [3]; ``"0-9"`` -> [0, ..., 9]."""
    lo, _, hi = spec.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("config", type=Path)
    p.add_argument("--pairs", default="0", help="a seed or an inclusive range, e.g. 0-9")
    p.add_argument("--runs", type=Path, default=Path("runs"))
    args = p.parse_args()

    cfg = CorpusConfig.from_yaml(args.config)
    for k in parse_pairs(args.pairs):
        out = generate_pair(cfg, k, args.runs / args.config.stem / f"pair-{k:03d}" / "data")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
