import json

import polars as pl
import pytest

from soccerai.data.data import add_period_columns
from soccerai.data.utils import home_attacks_right


@pytest.mark.parametrize(
    "period,start_left,start_left_et,expected",
    [
        (1, True, None, True),  # starts left, 1st half -> attacks right
        (2, True, None, False),  # swapped ends
        (1, False, None, False),
        (2, False, None, True),
        (3, True, False, False),  # extra time uses its own flag
        (4, True, False, True),
        (3, False, True, True),
        (4, False, True, False),
        (3, True, None, True),  # falls back to the regular-time flag
        (4, True, None, False),
    ],
)
def test_home_attacks_right(period, start_left, start_left_et, expected):
    assert home_attacks_right(period, start_left, start_left_et) is expected


def test_add_period_columns_joins_raw_events(tmp_path):
    event_dir = tmp_path / "events"
    meta_dir = tmp_path / "metadata"
    event_dir.mkdir()
    meta_dir.mkdir()

    events = [
        {"gameId": 7, "gameEventId": 100, "gameEvents": {"period": 1}},
        {"gameId": 7, "gameEventId": 100, "gameEvents": {"period": 1}},  # duplicate
        {"gameId": 7, "gameEventId": 101, "gameEvents": {"period": 2}},
        {"gameId": 7, "gameEventId": 102, "gameEvents": {"period": 3}},
    ]
    (event_dir / "7.json").write_text(json.dumps(events))
    meta = [
        {
            "id": 7,
            "awayTeam": {"name": "A"},
            "awayTeamKit": {"primaryColor": "#000"},
            "homeTeam": {"name": "H"},
            "homeTeamKit": {"primaryColor": "#fff"},
            "homeTeamStartLeft": True,
            "homeTeamStartLeftExtraTime": False,
            "startPeriod2": None,
        }
    ]
    (meta_dir / "7.json").write_text(json.dumps(meta))

    df = pl.DataFrame(
        {
            "gameId": [7, 7, 7, 7],
            "gameEventId": [102, 100, 101, 100],
            "x": [1.0, 2.0, 3.0, 4.0],
        }
    )
    out = add_period_columns(df, str(event_dir), str(meta_dir))

    assert out.height == df.height
    assert out["x"].to_list() == [1.0, 2.0, 3.0, 4.0]  # row order preserved
    assert out["period"].to_list() == [3, 1, 2, 1]
    assert out["homeTeamStartLeftExtraTime"].to_list() == [False] * 4


def test_add_period_columns_fails_on_unknown_event(tmp_path):
    event_dir = tmp_path / "events"
    meta_dir = tmp_path / "metadata"
    event_dir.mkdir()
    meta_dir.mkdir()
    (event_dir / "7.json").write_text(
        json.dumps([{"gameId": 7, "gameEventId": 1, "gameEvents": {"period": 1}}])
    )
    (meta_dir / "7.json").write_text(
        json.dumps(
            [
                {
                    "id": 7,
                    "awayTeam": {"name": "A"},
                    "awayTeamKit": {"primaryColor": "#000"},
                    "homeTeam": {"name": "H"},
                    "homeTeamKit": {"primaryColor": "#fff"},
                    "homeTeamStartLeft": True,
                    "startPeriod2": 1.0,
                }
            ]
        )
    )
    df = pl.DataFrame({"gameId": [7], "gameEventId": [999]})
    with pytest.raises(ValueError, match="no period"):
        add_period_columns(df, str(event_dir), str(meta_dir))
