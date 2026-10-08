import numpy as np
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
            {
                "game_id": 1,
                "chain_id": 0,
                "label": 1,
                "n_frames": 1,
                "period": period,
                "home_start_left": start_left,
                "home_start_left_et": start_left_et,
                "possession": "home",
            }
        ]
    )
    # every node refers to the goal attacked by the possession (home) team
    assert df.height == 22
    assert set(df["x_goal"].to_list()) == {home_goal}
    assert set(df["y_goal"].to_list()) == {34.0}


def test_prepare_dataframe_requires_period_column():
    ds = make_dataset_stub(make_data_cfg())
    raw = make_raw_df([{"game_id": 1, "chain_id": 0, "label": 1, "n_frames": 1}]).drop(
        "period"
    )
    with pytest.raises(ValueError, match="period"):
        ds._prepare_dataframe(raw)


def test_prepare_dataframe_drops_ball_rows_and_shot_frames():
    df = _prepare(
        [
            {
                "game_id": 1,
                "chain_id": 0,
                "label": 1,
                "n_frames": 3,
                "event_types": ["PA", "CR", "SH"],
            },
        ]
    )
    # the shot frame is removed, the ball row is folded into per-player columns
    assert set(df["possessionEventType"].unique().to_list()) == {"PA", "CR"}
    assert df.height == 2 * 22
    assert "x_ball" in df.columns and "team" not in df.columns
    assert df.filter(pl.col("is_ball_carrier") == 1).height == 2


@pytest.mark.parametrize("frame", [0, 1, 2])
def test_missing_carrier_discards_whole_chain(frame):
    raw = make_raw_df(
        [
            {"game_id": 1, "chain_id": 0, "label": 1, "n_frames": 3},
            {"game_id": 1, "chain_id": 1, "label": 0, "n_frames": 1},
        ]
    ).with_columns(
        pl.when(pl.col("index") == frame)
        .then(pl.lit("Unknown actor"))
        .otherwise(pl.col("playerName"))
        .alias("playerName")
    )
    df = make_dataset_stub(make_data_cfg())._prepare_dataframe(raw)
    assert df["chain_id"].unique().to_list() == [1]


def test_missing_ball_position_discards_chain_only_when_ball_features_used():
    raw = make_raw_df(
        [
            {"game_id": 1, "chain_id": 0, "label": 1, "n_frames": 2},
            {"game_id": 1, "chain_id": 1, "label": 0, "n_frames": 1},
        ]
    ).with_columns(
        pl.when((pl.col("index") == 1) & pl.col("team").is_null())
        .then(None)
        .otherwise(pl.col("x"))
        .alias("x")
    )
    with_ball = make_dataset_stub(make_data_cfg())._prepare_dataframe(raw)
    without_ball = make_dataset_stub(
        make_data_cfg(include_ball_features=False)
    )._prepare_dataframe(raw)
    assert with_ball["chain_id"].unique().to_list() == [1]
    assert set(without_ball["chain_id"].to_list()) == {0, 1}


def test_chain_cannot_cross_period_boundary():
    raw = make_raw_df(
        [
            {"game_id": 1, "chain_id": 0, "label": 0, "n_frames": 2},
            {"game_id": 1, "chain_id": 1, "label": 0, "n_frames": 1},
        ]
    ).with_columns(
        pl.when(pl.col("index") == 1)
        .then(2)
        .otherwise(pl.col("period"))
        .alias("period")
    )
    df = make_dataset_stub(make_data_cfg())._prepare_dataframe(raw)
    assert df["chain_id"].unique().to_list() == [1]


def test_unknown_possession_frame_cannot_silently_shorten_chain():
    raw = make_raw_df(
        [
            {
                "game_id": 1,
                "chain_id": 0,
                "label": 0,
                "n_frames": 3,
                "event_types": ["PA", None, "PA"],
            },
            {"game_id": 1, "chain_id": 1, "label": 0, "n_frames": 1},
        ]
    )
    df = make_dataset_stub(make_data_cfg())._prepare_dataframe(raw)
    assert df["chain_id"].unique().to_list() == [1]


def test_age_buckets_handle_missing_ages():
    raw = make_raw_df([{"game_id": 1, "chain_id": 0, "label": 1, "n_frames": 1}])
    # player with no age anywhere -> "unknown"; player 0 (null age) takes it from Age Info
    raw = raw.with_columns(
        pl.when((pl.col("jerseyNum") == "4") & (pl.col("team") == "home"))
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(pl.col("Age Info"))
        .alias("Age Info"),
        pl.when((pl.col("jerseyNum") == "4") & (pl.col("team") == "home"))
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("age"))
        .alias("age"),
    )
    ds = make_dataset_stub(make_data_cfg())
    df = ds._prepare_dataframe(raw)
    home = df.filter(pl.col("is_possession_team") == 1).sort("jerseyNum")
    ages = dict(zip(home["jerseyNum"].to_list(), home["age"].to_list()))
    assert ages["4"] == "unknown"
    assert ages["1"] == "20-28"  # Age Info 25 - 2.5 years
    assert ages["11"] == "20-28"  # age 30 - 2.5 = 27.5
    assert "35+" not in ages.values()


def test_overlapping_positive_chains_are_disambiguated():
    base = make_raw_df(
        [
            {
                "game_id": 1,
                "chain_id": 0,
                "label": 1,
                "n_frames": 3,
                "event_types": ["PA", "PA", "SH"],
            },
            {
                "game_id": 1,
                "chain_id": 1,
                "label": 1,
                "n_frames": 2,
                "event_types": ["PA", "SH"],
            },
            {
                "game_id": 1,
                "chain_id": 2,
                "label": 0,
                "n_frames": 3,
                "event_types": ["PA", "SH", "PA"],
            },
        ]
    )
    # chain 1 (second shot of the possession) also contains the frames of chain 0
    overlap = base.filter(pl.col("chain_id") == 0).with_columns(
        pl.lit(1, dtype=pl.Int64).alias("chain_id")
    )
    raw = pl.concat([base, overlap])

    ds = make_dataset_stub(make_data_cfg())
    df = ds._prepare_dataframe(raw)

    frames = (
        df.group_by(["chain_id", "gameEventId"]).len().sort(["chain_id", "gameEventId"])
    )
    assert frames["len"].unique().to_list() == [22]  # no duplicated frames
    chain_frames = {
        cid: sorted(grp["gameEventId"].to_list())
        for cid, grp in frames.group_by("chain_id", maintain_order=True)
    }
    chain_frames = {
        int(k[0]) if isinstance(k, tuple) else int(k): v
        for k, v in chain_frames.items()
    }
    assert chain_frames[0] == [0, 1]  # shared frames stay with the first shot
    assert chain_frames[1] == [3]  # chain 1 keeps only the frames after the first shot
    assert 2 not in chain_frames  # negative chain containing a shot is dropped


def test_processed_file_names_depend_on_config():
    from soccerai.data.converters import BipartiteGraphConverter

    a = make_dataset_stub(make_data_cfg())
    b = make_dataset_stub(make_data_cfg(include_ball_features=False))
    for ds in (a, b):
        ds.converter = BipartiteGraphConverter()
    assert a.processed_file_names != b.processed_file_names
    assert a.processed_file_names == make_dataset_stub(
        make_data_cfg()
    ).__class__.processed_file_names.fget(a)
    assert len(a.processed_file_names) == 3 and a.processed_file_names[2].endswith(
        ".json"
    )


def test_attack_direction_normalisation_mirrors_frames_attacking_left():
    spec = {
        "game_id": 1,
        "chain_id": 0,
        "label": 1,
        "n_frames": 1,
        "period": 2,
        "possession": "home",
    }
    raw = make_raw_df([spec])
    plain = make_dataset_stub(make_data_cfg())._prepare_dataframe(raw).sort("jerseyNum")
    normed = (
        make_dataset_stub(make_data_cfg(normalize_attack_direction=True))
        ._prepare_dataframe(raw)
        .sort("jerseyNum")
    )

    # home starts left and plays period 2 -> attacks left -> frame mirrored
    assert set(plain["x_goal"].to_list()) == {X_GOAL_LEFT}
    assert set(normed["x_goal"].to_list()) == {X_GOAL_RIGHT}
    for col in ["x", "x_ball"]:
        np.testing.assert_allclose(
            normed[col].to_numpy(), 105.0 - plain[col].to_numpy()
        )
    for col in ["cos", "vx", "cos_ball", "vx_ball"]:
        np.testing.assert_allclose(normed[col].to_numpy(), -plain[col].to_numpy())
    for col in ["y", "sin", "vy", "y_ball", "sin_ball", "vy_ball"]:
        np.testing.assert_allclose(normed[col].to_numpy(), plain[col].to_numpy())


def test_attack_direction_normalisation_keeps_frames_attacking_right():
    spec = {
        "game_id": 1,
        "chain_id": 0,
        "label": 0,
        "n_frames": 1,
        "period": 2,
        "possession": "away",
    }
    raw = make_raw_df([spec])
    plain = make_dataset_stub(make_data_cfg())._prepare_dataframe(raw)
    normed = make_dataset_stub(
        make_data_cfg(normalize_attack_direction=True)
    )._prepare_dataframe(raw)
    assert set(normed["x_goal"].to_list()) == {X_GOAL_RIGHT}
    np.testing.assert_allclose(normed["x"].to_numpy(), plain["x"].to_numpy())
    np.testing.assert_allclose(normed["cos"].to_numpy(), plain["cos"].to_numpy())


def _feature_names(cfg):
    raw = make_raw_df([{"game_id": 1, "chain_id": 0, "label": 1, "n_frames": 2}])
    ds = make_dataset_stub(cfg)
    df = ds._prepare_dataframe(raw)
    out = ds._create_preprocessor(df).fit_transform(df)
    return df, out.columns


def test_roster_and_clock_features_are_optional():
    from soccerai.data.config import SHOOTING_STATS

    _, with_all = _feature_names(make_data_cfg())
    df, without = _feature_names(
        make_data_cfg(use_roster_features=False, use_match_clock=False)
    )

    for col in ["Weight", "Market Value", "goals", "age_20-28", "frameTime"]:
        assert col in with_all and col not in without
    assert not any(c in without for c in SHOOTING_STATS)
    assert "playerRole_GK" in without and "ball_dist" in without and "dz" in without
    assert set(df["height_cm"].to_list()) == {180.0}
    assert "event_index" in without  # identifier kept for ordering


def test_event_index_orders_frames_of_a_chain():
    raw = make_raw_df([{"game_id": 1, "chain_id": 0, "label": 1, "n_frames": 3}])
    ds = make_dataset_stub(make_data_cfg())
    df = ds._prepare_dataframe(raw)
    assert "index" not in df.columns
    per_frame = (
        df.group_by("gameEventId")
        .agg(pl.col("event_index").first())
        .sort("gameEventId")
    )
    assert per_frame["event_index"].to_list() == [0, 1, 2]


@pytest.mark.parametrize("split_mode", ["chronological", "random"])
def test_split_games_is_disjoint_and_covers_every_game(split_mode):
    ds = make_dataset_stub(make_data_cfg(split_mode=split_mode))
    df = pl.DataFrame({"gameId": [g for g in range(1, 9) for _ in range(3)]})
    train, val = ds._split_games(df, list(range(1, 9)))
    train_games = set(train["gameId"].to_list())
    val_games = set(val["gameId"].to_list())
    assert not train_games & val_games
    assert train_games | val_games == set(range(1, 9))
    assert len(val_games) == 2  # val_ratio = 0.25


def test_random_split_ratio_ignores_dropped_games():
    # 8 games known, only 4 left after dropping: val takes 25% of the 4
    ds = make_dataset_stub(make_data_cfg(split_mode="random"))
    df = pl.DataFrame({"gameId": [1, 2, 3, 4]})
    train, val = ds._split_games(df, list(range(1, 9)))
    assert val.height == 1 and train.height == 3
