"""
Add the `period` (1-4) and `homeTeamStartLeftExtraTime` columns to an existing
`dataset.parquet` from the raw PFF event/metadata files.

The game period is the only reliable way to know which goal a team attacks
(teams swap ends after each period, including extra time); the previous rule
compared the game clock with a video timestamp. Rebuilding the whole dataset
would require re-reading ~50 GB of tracking data, so this script patches the
two columns in place instead.
"""

import argparse
import os
import shutil
from pathlib import Path

import polars as pl
from loguru import logger

from soccerai.data.data import add_period_columns


def main(args: argparse.Namespace) -> None:
    df = pl.read_parquet(args.dataset)
    logger.info("Loaded {} rows from {}", df.height, args.dataset)

    out = add_period_columns(df, args.event_dir, args.metadata_dir)

    per_period = out.unique(["gameId", "gameEventId"]).group_by("period").len()
    logger.info("Events per period:\n{}", per_period.sort("period"))
    extra_time_games = (
        out.filter(pl.col("period") > 2).select("gameId").unique().to_series().to_list()
    )
    logger.info("Games with extra time: {}", sorted(extra_time_games))

    if out.height != df.height:
        raise RuntimeError(
            f"Patching changed the row count ({df.height} -> {out.height}): "
            "dataset left untouched"
        )

    # write next to the original, keep a backup, then swap atomically
    dataset = Path(args.dataset)
    tmp = dataset.with_suffix(".parquet.tmp")
    backup = dataset.with_suffix(".parquet.bak")
    out.write_parquet(tmp)
    shutil.copy2(dataset, backup)
    os.replace(tmp, dataset)
    logger.success("Saved patched dataset to {} (backup: {})", dataset, backup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="soccerai/data/resources/raw/dataset.parquet"
    )
    parser.add_argument(
        "--event-dir", default="/home/soccerdata/FIFA_WorldCup_2022/Event Data"
    )
    parser.add_argument(
        "--metadata-dir", default="/home/soccerdata/FIFA_WorldCup_2022/Metadata"
    )
    main(parser.parse_args())
