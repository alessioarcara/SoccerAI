import polars as pl
import pytest
from factories import make_event_df

from soccerai.data.annotations import chain_errors, decode_chains, encode_chains
from soccerai.data.label import _is_within_range, _neg_labeling, _pos_labeling


def negatives(df, monkeypatch, positives=None):
    monkeypatch.setattr("soccerai.data.label._is_within_range", lambda *args: True)
    return _neg_labeling(
        df, pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), positives or [], 2, 25
    )


def test_negative_keeps_last_run_and_never_stitches_across_positive(monkeypatch):
    df = make_event_df(["A", "A", "B", "B", "A", "A"], ["PA"] * 6)
    assert negatives(df, monkeypatch, [[2, 3]]) == [[0, 1], [4, 5]]


@pytest.mark.parametrize("field", ["period", "gameId"])
def test_labeling_stops_at_period_and_game_boundaries(monkeypatch, field):
    df = make_event_df(["A"] * 4, ["PA", "PA", "PA", "PA"]).with_columns(
        pl.Series(field, [1, 1, 2, 2])
    )
    assert negatives(df, monkeypatch) == [[0, 1], [2, 3]]
    df = df.with_columns(pl.Series("possessionEventType", ["PA", "PA", "PA", "SH"]))
    assert _pos_labeling(df, 2, False) == [[2, 3]]


def test_negative_shot_without_eligible_positive_is_rejected(monkeypatch):
    df = make_event_df(["A", "A", "B", "B"], ["SH", "PA", "PA", "PA"])
    assert negatives(df, monkeypatch) == [[2, 3]]


def test_unknown_possession_event_cannot_label_a_negative(monkeypatch):
    df = make_event_df(["A"] * 3, ["PA", None, "PA"])
    assert negatives(df, monkeypatch) == []
    assert (
        "missing possession event" in chain_errors([[0, 1, 2]], df, positive=False)[0]
    )


def test_labeling_handles_empty_and_all_positive_events(monkeypatch):
    df = make_event_df(["A", "A"], ["PA", "SH"])
    assert negatives(df.head(0), monkeypatch) == []
    assert negatives(df, monkeypatch, [[0, 1]]) == []


def test_positive_uses_event_ids_and_stops_at_previous_shot():
    df = make_event_df(
        ["A"] * 5, ["PA", "CH", "SH", "PA", "SH"], indices=[10, 20, 30, 40, 50]
    )
    assert _pos_labeling(df, 2, True) == [[10, 30], [40, 50]]


def test_annotations_resolve_identity_after_indices_change():
    original = make_event_df(["A", "A"], ["PA", "SH"], indices=[10, 20])
    payload = encode_chains([[10, 20]], original)
    reloaded = original.with_columns(pl.Series("index", [1000, 1001])).reverse()
    assert decode_chains(payload, reloaded) == [[1000, 1001]]


def test_annotations_reject_legacy_missing_and_ambiguous_identities():
    df = make_event_df(["A", "A"], ["PA", "SH"])
    payload = encode_chains([[0, 1]], df)
    with pytest.raises(ValueError, match="Legacy"):
        decode_chains([[0, 1]], df)
    with pytest.raises(ValueError, match="missing"):
        decode_chains(payload, df.head(1))
    duplicate = df.head(1).with_columns(pl.lit(99, dtype=pl.Int64).alias("index"))
    with pytest.raises(ValueError, match="Ambiguous"):
        decode_chains(payload, pl.concat([df, duplicate]))


def test_annotation_validation_uses_unselected_timeline():
    df = make_event_df(["A", "B", "A", "A"], ["PA", "PA", "PA", "SH"])
    errors = chain_errors([[0, 2]], df, positive=False)
    assert "chain skips a possession or period boundary" in errors[0]
    assert not chain_errors([[2, 3]], df, positive=True)


def test_goal_range_uses_game_identity_and_rejects_missing_ball():
    df = make_event_df(["A"], ["PA"])
    players = pl.DataFrame(
        {
            "gameId": [2, 1],
            "gameEventId": [100, 100],
            "possessionEventId": [200, 200],
            "team": [None, None],
            "x": [0.0, 90.0],
        }
    )
    metadata = pl.DataFrame(
        {
            "gameId": [1],
            "homeTeamName": ["A"],
            "awayTeamName": ["B"],
            "homeTeamStartLeft": [True],
        }
    )
    assert _is_within_range(df, players, metadata, pl.DataFrame(), 0, "A", 25, 0, False)
    missing = players.with_columns(pl.lit(None, dtype=pl.Float64).alias("x"))
    assert not _is_within_range(
        df, missing, metadata, pl.DataFrame(), 0, "A", 25, 0, False
    )
