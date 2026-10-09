"""
Build the raw dataset parquet from the PFF World Cup 2022 data.

    # hand-annotated chains (the default dataset)
    python scripts/create_dataset.py soccerai/data/resources/raw/dataset.parquet
    # every possession of every game, with timeline targets (early warning)
    python scripts/create_dataset.py <root>/raw/dataset.parquet --possessions
"""

import argparse
from pathlib import Path

from soccerai.data.data import create_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("--possessions", action="store_true")
    parser.add_argument(
        "--players",
        default=None,
        help="parquet of players with velocities, to skip the tracking pass",
    )
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    create_dataset(
        str(args.output),
        possessions=args.possessions,
        players_path=args.players,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
