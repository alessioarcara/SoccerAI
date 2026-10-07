import numpy as np
import polars as pl
import pytest
from factories import make_raw_df

from soccerai.data.dataset import WorldCup2022Dataset
from soccerai.data.transformers import (
    BallLocationTransformer,
    GoalLocationTransformer,
    NonPossessionShootingStatsMask,
    PlayerLocationTransformer,
)
from soccerai.training.trainer_config import DataConfig

DIAG = float(np.hypot(105.0, 68.0))


def make_data_cfg(**overrides) -> DataConfig:
    defaults = dict(
        val_ratio=0.25,
        include_goal_features=True,
        include_ball_features=True,
        use_macro_roles=False,
        use_augmentations=False,
        use_regression_imputing=False,
        use_pca_on_roster_cols=False,
        mask_non_possession_shooting_stats=False,
        connection_mode="bipartite",
        normalize_attack_direction=False,
    )
    defaults.update(overrides)
    return DataConfig(**defaults)


def make_dataset_stub(cfg: DataConfig) -> WorldCup2022Dataset:
    """A dataset object without triggering `process()` / file loading."""
    ds = WorldCup2022Dataset.__new__(WorldCup2022Dataset)
    ds.cfg = cfg
    ds.random_state = 0
    return ds


def test_ball_transformer_reads_columns_by_name():
    # columns deliberately in a scrambled order
    df = pl.DataFrame(
        {
            "height_cm": [180.0, 170.0],
            "vy_ball": [1.0, -1.0],
            "x": [10.0, 50.0],
            "z_ball": [0.5, 2.0],
            "cos": [1.0, 0.0],
            "vx": [2.0, 0.0],
            "y_ball": [34.0, 10.0],
            "sin": [0.0, 1.0],
            "x_ball": [13.0, 50.0],
            "vx_ball": [5.0, 0.0],
            "cos_ball": [0.0, 0.0],
            "sin_ball": [1.0, 1.0],
            "vy": [0.0, 3.0],
            "y": [30.0, 10.0],
        }
    )
    out = BallLocationTransformer().fit(df).transform(df)
    assert out.shape == (2, 5)
    ball_dist, dz, sim, dvx, dvy = out.T

    np.testing.assert_allclose(ball_dist, [5.0 / DIAG, 0.0], atol=1e-5)
    expected_dz = 2.0 / (1.0 + np.exp(-(np.array([0.5, 2.0]) - [1.8, 1.7]))) - 1.0
    np.testing.assert_allclose(dz, expected_dz, atol=1e-9)
    np.testing.assert_allclose(sim, [0.0, 1.0])  # cos_ball*cos + sin_ball*sin
    np.testing.assert_allclose(dvx, [3.0, 0.0])
    np.testing.assert_allclose(dvy, [1.0, -4.0])
    assert list(BallLocationTransformer().get_feature_names_out()) == [
        "ball_dist",
        "dz",
        "ball_direction_sim",
        "dvx",
        "dvy",
    ]


def test_missing_input_column_raises():
    df = pl.DataFrame({"x": [1.0], "y": [2.0]})
    with pytest.raises(ValueError, match="missing input columns"):
        GoalLocationTransformer().fit(df).transform(df)


def test_player_location_scales_and_clips():
    df = pl.DataFrame(
        {
            "vx": [1.0, -1.0],
            "x": [52.5, 120.0],
            "sin": [0.0, 1.0],
            "y": [34.0, -3.0],
            "cos": [1.0, 0.0],
            "vy": [0.0, 2.0],
        }
    )
    out = PlayerLocationTransformer().fit(df).transform(df)
    np.testing.assert_allclose(out[:, 0], [0.5, 1.0])
    np.testing.assert_allclose(out[:, 1], [0.5, 0.0])
    np.testing.assert_allclose(
        out[:, 2:], [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, -1.0, 2.0]]
    )


def test_goal_location_features():
    df = pl.DataFrame({"x_goal": [105.0], "y": [34.0], "x": [95.0], "y_goal": [34.0]})
    dist, cos, sin = GoalLocationTransformer().fit(df).transform(df)[0]
    assert dist == pytest.approx(10.0 / DIAG, rel=1e-4)
    assert cos == pytest.approx(1.0, abs=1e-5)
    assert sin == pytest.approx(0.0, abs=1e-5)


def test_non_possession_mask_keeps_names_and_zeroes_rows():
    df = pl.DataFrame(
        {"goals": [3.0, 5.0], "is_possession_team_1": [1.0, 0.0], "shots": [7.0, 9.0]}
    )
    t = NonPossessionShootingStatsMask().fit(df)
    out = t.transform(df)
    assert list(t.get_feature_names_out()) == ["goals", "shots", "is_possession_team_1"]
    np.testing.assert_allclose(out, [[3.0, 7.0, 1.0], [0.0, 0.0, 0.0]])


def test_preprocessor_produces_consistent_ball_and_goal_features():
    raw = make_raw_df(
        [
            dict(game_id=1, chain_id=0, label=1, n_frames=2),
            dict(game_id=1, chain_id=1, label=0, n_frames=2, possession="away"),
        ]
    )
    ds = make_dataset_stub(make_data_cfg())
    df = ds._prepare_dataframe(raw)
    prep = ds._create_preprocessor(df)
    out = prep.fit_transform(df)

    assert isinstance(out, pl.DataFrame)
    names = out.columns
    for required in [
        "ball_dist",
        "dz",
        "ball_direction_sim",
        "dvx",
        "dvy",
        "goal_dist",
        "x",
        "y",
        "vx",
        "vy",
        "is_possession_team_1",
        "is_ball_carrier_1",
        "chain_id",
        "label",
    ]:
        assert required in names, required
    assert "x_goal" not in names and "y_goal" not in names and "height_cm" not in names

    # ball_dist must equal the real planar distance between player and ball
    expected = np.hypot(df["x_ball"] - df["x"], df["y_ball"] - df["y"]) / DIAG
    np.testing.assert_allclose(
        out["ball_dist"].to_numpy(), expected.to_numpy(), atol=1e-5
    )

    dz = out["dz"].to_numpy()
    assert dz.min() > -1.0 and dz.max() < 1.0 and dz.std() > 0.0
    sim = out["ball_direction_sim"].to_numpy()
    assert sim.min() >= -1.0 - 1e-6 and sim.max() <= 1.0 + 1e-6
    assert out["x"].min() >= 0.0 and out["x"].max() <= 1.0
