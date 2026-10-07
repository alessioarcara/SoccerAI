import polars as pl
import pytest
from factories import make_raw_df
from test_transformers import make_data_cfg, make_dataset_stub

from soccerai.data.config import X_GOAL_LEFT, X_GOAL_RIGHT


def _prepare(chains):
    ds = make_dataset_stub(make_data_cfg())
    return ds._prepare_dataframe(make_raw_df(chains))


@pytest.mark.parametrize(
    "period,start_left,start_left_et,home_goal",
    [
        (1, True, None, X_GOAL_RIGHT),
        (2, True, None, X_GOAL_LEFT),
        (1, False, None, X_GOAL_LEFT),
        (2, False, None, X_GOAL_RIGHT),
        (3, True, False, X_GOAL_LEFT),  # extra time: home starts on the right
        (4, True, False, X_GOAL_RIGHT),
        (3, True, None, X_GOAL_RIGHT),  # missing extra-time flag -> regular flag
    ],
)
def test_goal_side_follows_game_period(period, start_left, start_left_et, home_goal):
    df = _prepare(
        [
            dict(
                game_id=1,
                chain_id=0,
                label=1,
                n_frames=1,
                period=period,
                home_start_left=start_left,
                home_start_left_et=start_left_et,
                possession="home",
            )
        ]
    )
    home_rows = df.filter(pl.col("is_possession_team") == 1)
    away_rows = df.filter(pl.col("is_possession_team") == 0)
    assert home_rows.height == 11 and away_rows.height == 11
    assert set(home_rows["x_goal"].to_list()) == {home_goal}
    assert set(away_rows["x_goal"].to_list()) == {
        X_GOAL_RIGHT + X_GOAL_LEFT - home_goal
    }
    assert set(df["y_goal"].to_list()) == {34.0}


def test_prepare_dataframe_requires_period_column():
    ds = make_dataset_stub(make_data_cfg())
    raw = make_raw_df([dict(game_id=1, chain_id=0, label=1, n_frames=1)]).drop("period")
    with pytest.raises(ValueError, match="period"):
        ds._prepare_dataframe(raw)


def test_prepare_dataframe_drops_ball_rows_and_shot_frames():
    df = _prepare(
        [
            dict(
                game_id=1,
                chain_id=0,
                label=1,
                n_frames=3,
                event_types=["PA", "CR", "SH"],
            ),
        ]
    )
    # the shot frame is removed, the ball row is folded into per-player columns
    assert df["possessionEventType"].unique().to_list() == ["CR", "PA"] or set(
        df["possessionEventType"].unique().to_list()
    ) == {"PA", "CR"}
    assert df.height == 2 * 22
    assert "x_ball" in df.columns and "team" not in df.columns
    assert df.filter(pl.col("is_ball_carrier") == 1).height == 2
