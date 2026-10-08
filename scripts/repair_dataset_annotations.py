"""Migrate legacy annotations using the parquet's verified event identities.

Also remove invalid negative annotations and restore missing ball coordinates
from the raw events. Original files are backed up before the atomic swaps.
Tracking velocities are retained: this does not read the tracking archive.
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from soccerai.data.annotations import (
    FRAME_KEYS,
    chain_errors,
    decode_chains,
    encode_chains,
)
from soccerai.data.data import load_and_process_soccer_events


def _legacy_timeline(reference: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    # The committed parquet establishes the historical game order. This lets
    # us recover the few annotated events missing from the parquet without
    # guessing the filesystem's original directory enumeration.
    order = (
        reference.group_by("gameId")
        .agg(pl.col("index").min())
        .sort("index")["gameId"]
        .to_list()
    )
    if set(order) != set(events["gameId"].to_list()):
        raise ValueError(
            "Cannot reconstruct legacy indices: reference and raw games differ"
        )
    legacy = (
        events.with_columns(
            pl.col("gameId")
            .replace_strict(dict(zip(order, range(len(order)))))
            .alias("_game_order")
        )
        .sort(["_game_order", "index"])
        .drop("index", "_game_order")
        .with_row_index()
    )
    original = reference.select("index", *FRAME_KEYS).unique()
    check = original.join(
        legacy.select("index", *FRAME_KEYS),
        on="index",
        how="left",
        suffix="_raw",
        validate="m:1",
    )
    matches = pl.all_horizontal(
        [pl.col(c).eq_missing(pl.col(c + "_raw")) for c in FRAME_KEYS]
    )
    if check.filter(~matches).height:
        raise ValueError(
            "Raw events do not match the historical parquet indices; migration refused"
        )
    return legacy


def repair_dataset_annotations(
    reference: pl.DataFrame,
    events: pl.DataFrame,
    players: pl.DataFrame,
    positive: Any,
    negative: Any,
) -> tuple[pl.DataFrame, dict, dict, dict]:
    if isinstance(positive, list) != isinstance(negative, list):
        raise ValueError("Positive and negative annotations must use the same format")
    if isinstance(positive, list):
        legacy = _legacy_timeline(reference, events)
        positive = encode_chains(positive, legacy)
        negative = encode_chains(negative, legacy)
    pos_chains = decode_chains(positive, events)
    neg_chains = decode_chains(negative, events)
    invalid_pos = chain_errors(pos_chains, events, positive=True)
    if invalid_pos:
        raise ValueError(f"Positive annotations need manual review: {invalid_pos}")
    invalid_neg = chain_errors(neg_chains, events, positive=False)
    kept_neg = [chain for i, chain in enumerate(neg_chains) if i not in invalid_neg]
    if {i for c in pos_chains for i in c} & {i for c in kept_neg for i in c}:
        raise ValueError("Positive and negative annotations overlap")

    n_pos = len(pos_chains)
    old_ids = list(range(n_pos)) + [
        n_pos + i for i in range(len(neg_chains)) if i not in invalid_neg
    ]
    id_map = dict(zip(old_ids, range(len(old_ids))))
    out = reference.filter(pl.col("chain_id").is_in(old_ids)).with_columns(
        pl.col("chain_id").replace_strict(id_map).cast(pl.Int64)
    )
    selected_keys = out.select(FRAME_KEYS).unique()
    selected_events = events.join(
        selected_keys, on=FRAME_KEYS, how="semi", nulls_equal=True
    )
    selected_players = players.join(
        selected_keys, on=FRAME_KEYS, how="semi", nulls_equal=True
    )
    out = out.join(
        selected_events.select(*FRAME_KEYS, pl.col("index").alias("_new_index")),
        on=FRAME_KEYS,
        how="left",
        nulls_equal=True,
        validate="m:1",
        maintain_order="left",
    )
    if out["_new_index"].null_count():
        raise ValueError("Some parquet events are missing from the raw data")
    out = out.with_columns(pl.col("_new_index").alias("index")).drop("_new_index")

    balls = selected_players.filter(pl.col("team").is_null()).select(
        *FRAME_KEYS, *[pl.col(c).alias(f"_raw_ball_{c}") for c in ["x", "y", "z"]]
    )
    out = out.join(
        balls,
        on=FRAME_KEYS,
        how="left",
        nulls_equal=True,
        validate="m:1",
        maintain_order="left",
    )
    missing_ball_frames = (
        out.filter(
            pl.col("team").is_null()
            & (pl.col("_raw_ball_x").is_null() | pl.col("_raw_ball_y").is_null())
        )
        .select(FRAME_KEYS)
        .n_unique()
    )
    out = out.with_columns(
        [
            pl.when(pl.col("team").is_null())
            .then(pl.col(f"_raw_ball_{c}"))
            .otherwise(pl.col(c))
            .alias(c)
            for c in ["x", "y", "z"]
        ]
    ).drop([f"_raw_ball_{c}" for c in ["x", "y", "z"]])

    entity_keys = [*FRAME_KEYS, "team", "jerseyNum"]
    out = out.drop("index_right").join(
        selected_players.select(*entity_keys, pl.col("index").alias("index_right")),
        on=entity_keys,
        how="left",
        nulls_equal=True,
        validate="m:1",
        maintain_order="left",
    )
    if out["index_right"].null_count():
        raise ValueError("Some parquet players are missing from the raw events")
    report = {
        "removed_negative_chains": {
            n_pos + i: reasons for i, reasons in invalid_neg.items()
        },
        "missing_ball_frames_restored": missing_ball_frames,
        "rows_before": reference.height,
        "rows_after": out.height,
        "positive_chains": len(pos_chains),
        "negative_chains": len(kept_neg),
    }
    return (
        out,
        encode_chains(pos_chains, events),
        encode_chains(kept_neg, events),
        report,
    )


def _annotation_text(payload: dict) -> str:
    # One chain per line keeps the larger event identities readable in diffs.
    header = json.dumps({k: v for k, v in payload.items() if k != "chains"})[:-1]
    chains = ",\n".join(
        "    " + json.dumps(c, separators=(",", ":")) for c in payload["chains"]
    )
    return header + ', "chains": [\n' + chains + "\n]}\n"


def main(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset)
    annotation_dir = Path(args.annotations_dir)
    pos_path = annotation_dir / "accepted_pos_chains.json"
    neg_path = annotation_dir / "accepted_neg_chains.json"
    reference = pl.read_parquet(dataset)
    events, players = load_and_process_soccer_events(args.event_dir, True)
    out, positive, negative, report = repair_dataset_annotations(
        reference,
        events,
        players,
        json.loads(pos_path.read_text()),
        json.loads(neg_path.read_text()),
    )

    backup_dir = Path(
        tempfile.mkdtemp(prefix="soccerai-data-backup-", dir=args.backup_dir)
    )
    for path in [dataset, pos_path, neg_path]:
        shutil.copy2(path, backup_dir / path.name)
    (backup_dir / "repair_report.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Repair report: {}", report)
    logger.info("Original files backed up to {}", backup_dir)
    # Stage all three outputs before replacing any originals.
    staged = []
    for target in [dataset, pos_path, neg_path]:
        fd, name = tempfile.mkstemp(
            prefix=target.name + ".", suffix=".tmp", dir=target.parent
        )
        os.close(fd)
        staged.append((Path(name), target))
    out.write_parquet(staged[0][0])
    staged[1][0].write_text(_annotation_text(positive))
    staged[2][0].write_text(_annotation_text(negative))
    for source, target in staged:
        os.replace(source, target)
    logger.success("Repaired dataset and migrated annotations; backup: {}", backup_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="soccerai/data/resources/raw/dataset.parquet"
    )
    parser.add_argument("--annotations-dir", default="soccerai/data/resources")
    parser.add_argument(
        "--event-dir", default="/home/soccerdata/FIFA_WorldCup_2022/Event Data"
    )
    parser.add_argument("--backup-dir", default="/tmp")
    main(parser.parse_args())
