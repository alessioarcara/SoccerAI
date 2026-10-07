import polars as pl
from factories import make_raw_df
from test_transformers import make_data_cfg, make_dataset_stub


def _games_df():
    specs = []
    chain_id = 0
    for game in range(1, 9):
        labels = [1] if game == 7 else [1, 0]  # game 7 has no negative chain
        for label in labels:
            specs.append(dict(game_id=game, chain_id=chain_id, label=label, n_frames=2))
            chain_id += 1
    return make_raw_df(specs)


def _games(df):
    return sorted(df.select("gameId").unique()["gameId"].to_list())


def test_chronological_split_uses_the_last_games_and_drops_positive_only_games():
    ds = make_dataset_stub(make_data_cfg(val_ratio=0.25))
    df = ds._prepare_dataframe(_games_df())
    all_games = _games(df)
    df = ds._drop_games_without_negatives(df)
    train, val = ds._split_games(df, all_games)
    assert _games(train) == [1, 2, 3, 4, 5, 6]
    assert _games(val) == [8]  # game 7 dropped: only positives


def test_random_split_is_seeded_and_disjoint():
    ds = make_dataset_stub(make_data_cfg(val_ratio=0.25, split_mode="random"))
    df = ds._prepare_dataframe(_games_df())
    train, val = ds._split_games(df, _games(df))
    train2, val2 = ds._split_games(df, _games(df))
    assert _games(val) == _games(val2) and len(_games(val)) == 2
    assert not set(_games(train)) & set(_games(val))
    assert set(_games(train)) | set(_games(val)) == set(range(1, 9))


def test_positive_chains_must_end_near_the_goal():
    raw = make_raw_df(
        [
            dict(game_id=1, chain_id=0, label=1, n_frames=2),  # ends near the goal
            dict(game_id=1, chain_id=1, label=1, n_frames=2),  # ends far away
            dict(game_id=1, chain_id=2, label=0, n_frames=2),  # negatives untouched
        ]
    )
    is_carrier = pl.col("playerName") == pl.col("playerName_right")
    last_frame = pl.col("gameEventId") == pl.col("gameEventId").max().over("chain_id")
    raw = raw.with_columns(
        pl.when(is_carrier & last_frame & (pl.col("chain_id") == 0))
        .then(95.0)
        .when(is_carrier & last_frame & (pl.col("chain_id") == 1))
        .then(40.0)
        .otherwise(pl.col("x"))
        .alias("x")
    )
    # home starts left, period 1 -> attacks the goal at x = 105
    ds = make_dataset_stub(make_data_cfg(goal_window_for_positives=25.0))
    df = ds._prepare_dataframe(raw)
    assert sorted(df["chain_id"].unique().to_list()) == [0, 2]

    ds_all = make_dataset_stub(make_data_cfg(goal_window_for_positives=None))
    assert sorted(ds_all._prepare_dataframe(raw)["chain_id"].unique().to_list()) == [
        0,
        1,
        2,
    ]
