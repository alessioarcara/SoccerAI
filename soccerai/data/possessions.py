"""
Possessions of a whole match and per-event targets read from the match
timeline, for the early-warning task: every event of every possession is an
example, labelled by what its team does in the following seconds (instead of
by how a hand-picked chain ends).
"""

import polars as pl

from soccerai.data.annotations import FRAME_KEYS
from soccerai.data.config import X_GOAL_RIGHT, Y_GOAL

# penalty area: 16.5 m deep, 40.32 m wide (pitch coordinates in metres)
BOX_DEPTH = 16.5
BOX_HALF_WIDTH = 20.16
VALID_PERIODS = (1, 2, 3, 4)


def segment_possessions(event_df: pl.DataFrame) -> list[list[int]]:
    """
    Event indices of every possession: maximal runs of consecutive events of
    the same team in the same game period. Events without a team (stoppages,
    contested balls) or outside the regular and extra-time periods end the
    current run.
    """
    possessions: list[list[int]] = []
    run: list[int] = []
    current_key = None
    for row in event_df.sort("index").iter_rows(named=True):
        valid = row["teamName"] is not None and row["period"] in VALID_PERIODS
        key = (row["gameId"], row["period"], row["teamName"]) if valid else None
        if key != current_key and run:
            possessions.append(run)
            run = []
        current_key = key
        if valid:
            run.append(row["index"])
    if run:
        possessions.append(run)
    return possessions


def _attacks_right() -> pl.Expr:
    """Whether the team of the event attacks towards x = pitch length."""
    start_left_extra_time = pl.col("homeTeamStartLeftExtraTime").fill_null(
        pl.col("homeTeamStartLeft")
    )
    home_start_left = (
        pl.when(pl.col("period").is_in([3, 4]))
        .then(start_left_extra_time)
        .otherwise(pl.col("homeTeamStartLeft"))
        .cast(pl.Boolean)
    )
    home_attacks_right = home_start_left == pl.col("period").is_in([1, 3])
    is_home = pl.col("teamName") == pl.col("homeTeamName")
    return is_home == home_attacks_right


def timeline_targets(
    event_df: pl.DataFrame, players_df: pl.DataFrame, metadata_df: pl.DataFrame
) -> pl.DataFrame:
    """
    For every event with a team, seconds (video time, `startTime`) until:

    - `time_to_shot`: the next shot of the same team, strictly after the
      event, in the same game period (+inf if none);
    - `time_to_box`: the next event of the same team, the event itself
      included, with the ball in the penalty area that team attacks (+inf);
    - `time_to_period_end`: the last event of the period, so that targets
      whose horizon runs past it can be masked.

    The timeline ignores possession boundaries: a ball lost and won back
    before a shot still leads to that shot.
    """
    ball = players_df.filter(pl.col("team").is_null()).select(
        *FRAME_KEYS, pl.col("x").alias("ball_x"), pl.col("y").alias("ball_y")
    )
    meta = metadata_df.select(
        pl.col("gameId").cast(pl.Int64),
        "homeTeamName",
        "homeTeamStartLeft",
        "homeTeamStartLeftExtraTime",
    )
    events = (
        event_df.filter(
            pl.col("teamName").is_not_null() & pl.col("period").is_in(VALID_PERIODS)
        )
        .select(
            "index",
            *FRAME_KEYS,
            "period",
            "teamName",
            "possessionEventType",
            "startTime",
        )
        .with_columns(pl.col("gameId").cast(pl.Int64))
        .join(
            ball.with_columns(pl.col("gameId").cast(pl.Int64)),
            on=FRAME_KEYS,
            how="left",
        )
        .join(meta, on="gameId", how="left")
    )
    goal_x = pl.when(_attacks_right()).then(X_GOAL_RIGHT).otherwise(0.0)
    events = events.with_columns(
        (
            ((pl.col("ball_x") - goal_x).abs() <= BOX_DEPTH)
            & ((pl.col("ball_y") - Y_GOAL).abs() <= BOX_HALF_WIDTH)
        )
        .fill_null(False)
        .alias("in_box")
    )

    timeline = ["gameId", "period", "teamName"]
    shots = events.filter(pl.col("possessionEventType") == "SH").select(
        *timeline, pl.col("index").alias("_idx"), pl.col("startTime").alias("_t")
    )
    boxes = events.filter(pl.col("in_box")).select(
        *timeline, pl.col("index").alias("_idx"), pl.col("startTime").alias("_t")
    )

    def time_to(targets: pl.DataFrame, name: str, include_self: bool) -> pl.DataFrame:
        after = (
            pl.col("_idx") >= pl.col("index")
            if include_self
            else pl.col("_idx") > pl.col("index")
        )
        return (
            events.select("index", *timeline, "startTime")
            .join(targets, on=timeline)
            .filter(after)
            .group_by("index")
            .agg(
                (pl.col("_t") - pl.col("startTime"))
                .clip(lower_bound=0)
                .min()
                .alias(name)
            )
        )

    period_end = events.group_by("gameId", "period").agg(
        pl.col("startTime").max().alias("_period_end")
    )
    return (
        events.select("index", "gameId", "period", "startTime")
        .join(
            time_to(shots, "time_to_shot", include_self=False), on="index", how="left"
        )
        .join(time_to(boxes, "time_to_box", include_self=True), on="index", how="left")
        .join(period_end, on=["gameId", "period"], how="left")
        .select(
            "index",
            pl.col("time_to_shot").fill_null(float("inf")),
            pl.col("time_to_box").fill_null(float("inf")),
            (pl.col("_period_end") - pl.col("startTime")).alias("time_to_period_end"),
        )
    )
