import math

import polars as pl
import pytest

from soccerai.data.possessions import segment_possessions, timeline_targets


def events(teams, types, times, periods=None, game=1):
    n = len(teams)
    return pl.DataFrame(
        {
            "index": list(range(n)),
            "gameId": [game] * n,
            "gameEventId": list(range(100, 100 + n)),
            "possessionEventId": [float(200 + i) for i in range(n)],
            "period": periods or [1] * n,
            "teamName": teams,
            "possessionEventType": types,
            "startTime": [float(t) for t in times],
        }
    )


def ball(ev: pl.DataFrame, xs) -> pl.DataFrame:
    return ev.select("gameId", "gameEventId", "possessionEventId").with_columns(
        pl.lit(None, dtype=pl.String).alias("team"),
        pl.Series("x", [float(x) for x in xs]),
        pl.lit(34.0).alias("y"),
    )


META = pl.DataFrame(
    {
        "gameId": [1],
        "homeTeamName": ["A"],
        "homeTeamStartLeft": [True],  # A attacks right in period 1
        "homeTeamStartLeftExtraTime": [None],
    }
)


def test_possessions_are_runs_of_one_team_broken_by_teamless_events():
    ev = events(
        ["A", "A", "B", None, "B", "B", "A"],
        ["PA"] * 7,
        range(7),
        periods=[1, 1, 1, 1, 1, 2, 2],
    )
    assert segment_possessions(ev) == [[0, 1], [2], [4], [5], [6]]


def test_timeline_targets_follow_the_team_across_possessions():
    # A passes, loses the ball, wins it back in the box and shoots
    ev = events(
        ["A", "A", "B", "A", "A"],
        ["PA", "PA", "PA", "PA", "SH"],
        [0, 2, 4, 9, 10],
    )
    t = timeline_targets(ev, ball(ev, [50, 70, 60, 95, 98]), META).sort("index")

    assert t["time_to_shot"].to_list()[:2] == [10.0, 8.0]
    assert math.isinf(t["time_to_shot"][2])  # B never shoots
    assert t["time_to_shot"][3] == 1.0
    assert math.isinf(t["time_to_shot"][4])  # strictly after the event
    # box reached by A at t = 9 (x = 95 >= 105 - 16.5); the event itself counts
    assert t["time_to_box"].to_list()[:2] == [9.0, 7.0]
    assert t["time_to_box"][3] == 0.0
    # B attacks left: x = 60 is not its penalty area
    assert math.isinf(t["time_to_box"][2])
    assert t["time_to_period_end"].to_list() == [10.0, 8.0, 6.0, 1.0, 0.0]


def test_attack_direction_switches_with_the_period():
    ev = events(["A", "A"], ["PA", "PA"], [0, 1], periods=[2, 2])
    t = timeline_targets(ev, ball(ev, [5, 98]), META).sort("index")
    # in period 2 A attacks left: only x = 5 is in its penalty area
    assert t["time_to_box"].to_list() == [0.0, pytest.approx(math.inf)]
