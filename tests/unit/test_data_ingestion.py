import json

import polars as pl
import pytest
from factories import make_event_df, make_raw_df

from scripts.repair_dataset_annotations import repair_dataset_annotations
from soccerai.data.annotations import FRAME_KEYS, decode_chains
from soccerai.data.data import load_and_process_soccer_events
from soccerai.data.utils import offset_x, offset_y, save_accepted_chains


def raw_event(game):
    return {
        "gameId": game,
        "gameEventId": 10 + game,
        "possessionEventId": 20 + game,
        "startTime": 1.0,
        "endTime": 2.0,
        "duration": 1.0,
        "gameEvents": {
            "gameEventType": "OTB",
            "teamName": "A",
            "playerName": "P",
            "videoUrl": "",
            "period": 1,
        },
        "possessionEvents": {
            "possessionEventType": "PA",
            "formattedGameClock": "00:01",
        },
        "homePlayers": [{"x": 0.0, "y": 0.0, "jerseyNum": "1"}],
        "awayPlayers": [{"x": 1.0, "y": 1.0, "jerseyNum": "2"}],
        "ball": {"x": None, "y": None, "z": 0.0, "visibility": "UNKNOWN"},
    }


def test_loader_is_independent_of_directory_enumeration(tmp_path, monkeypatch):
    for name, game in [("b.json", 2), ("a.json", 1)]:
        (tmp_path / name).write_text(json.dumps([raw_event(game)]))
    monkeypatch.setattr("soccerai.data.data.os.listdir", lambda _: ["b.json", "a.json"])
    df, players = load_and_process_soccer_events(str(tmp_path), True)
    assert df["gameId"].to_list() == [1, 2]
    assert players.filter(pl.col("team").is_null())["x"].null_count() == 2
    monkeypatch.setattr("soccerai.data.data.os.listdir", lambda _: ["a.json", "b.json"])
    other, _ = load_and_process_soccer_events(str(tmp_path), True)
    assert df.equals(other)
    assert offset_x(None) is None and offset_y(None) is None
    assert offset_x(0.0) == 52.5 and offset_y(0.0) == 34.0


def test_saving_annotations_keeps_identities_and_deduplicates(tmp_path):
    df = make_event_df(["A", "A"], ["PA", "SH"])
    for _ in range(2):
        save_accepted_chains([[0, 1]], str(tmp_path), True, df)
    payload = json.loads((tmp_path / "accepted_pos_chains.json").read_text())
    assert decode_chains(payload, df) == [[0, 1]]
    with pytest.raises(ValueError, match="Invalid accepted"):
        save_accepted_chains([[]], str(tmp_path), True, df)


def migration_inputs():
    raw = make_raw_df(
        [
            {
                "game_id": 1,
                "chain_id": 0,
                "label": 1,
                "n_frames": 2,
                "event_types": ["PA", "SH"],
            },
            {"game_id": 2, "chain_id": 1, "label": 0, "n_frames": 2},
        ]
    )
    timeline = (
        raw.filter(pl.col("team") == "home")
        .unique(FRAME_KEYS, maintain_order=True)
        .select(
            "index",
            *FRAME_KEYS,
            "period",
            "teamName",
            "possessionEventType",
        )
    )
    players = raw.select(
        *FRAME_KEYS, "team", "jerseyNum", "x", "y", "z"
    ).with_row_index()
    players = players.with_columns(
        pl.when(pl.col("team").is_null() & (pl.col("gameId") == 2))
        .then(None)
        .otherwise(pl.col("x"))
        .alias("x")
    )
    reference = raw.with_columns(
        pl.col("index").replace_strict({0: 2, 1: 3, 2: 0, 3: 1})
    )
    return reference, timeline, players


def test_migration_verifies_legacy_mapping_and_restores_missing_ball():
    reference, timeline, players = migration_inputs()
    out, positive, negative, report = repair_dataset_annotations(
        reference, timeline, players, [[2, 3]], [[0, 1]]
    )
    assert decode_chains(positive, timeline) == [[0, 1]]
    assert decode_chains(negative, timeline) == [[2, 3]]
    assert out.filter(pl.col("gameId") == 1)["index"].unique().sort().to_list() == [
        0,
        1,
    ]
    assert (
        out.filter(pl.col("team").is_null() & (pl.col("gameId") == 2))["x"].null_count()
        == 2
    )
    assert report["missing_ball_frames_restored"] == 2
    assert out.height == reference.height


def test_migration_refuses_mismatched_legacy_reference():
    reference, timeline, players = migration_inputs()
    broken = reference.with_columns((pl.col("gameEventId") + 100).alias("gameEventId"))
    with pytest.raises(ValueError, match="migration refused"):
        repair_dataset_annotations(broken, timeline, players, [[2, 3]], [[0, 1]])


def test_migration_ignores_duplicate_unannotated_entities():
    reference, timeline, players = migration_inputs()
    extra = players.head(1).with_columns(pl.lit(999, dtype=pl.Int64).alias("gameId"))
    out, _, _, _ = repair_dataset_annotations(
        reference, timeline, pl.concat([players, extra, extra]), [[2, 3]], [[0, 1]]
    )
    assert out.height == reference.height


def test_migration_removes_negative_shot_and_reassigns_chain_ids():
    reference, timeline, players = migration_inputs()
    reference = reference.with_columns(
        pl.when(pl.col("gameId") == 2)
        .then(2)
        .otherwise(pl.col("chain_id"))
        .alias("chain_id")
    )
    out, _, negative, report = repair_dataset_annotations(
        reference, timeline, players, [[2, 3]], [[3], [0, 1]]
    )
    assert decode_chains(negative, timeline) == [[2, 3]]
    assert set(out["chain_id"].to_list()) == {0, 1}
    assert 1 in report["removed_negative_chains"]
