import polars as pl


def add_goal_features(df: pl.DataFrame) -> pl.DataFrame:
    df = add_goal_positions(df)
    pass


def add_goal_positions(df: pl.DataFrame) -> pl.DataFrame:
    is_home_team = df["team"] == "home"
    is_second_half = df["frameTime"] > df["startPeriod2"]
    is_goal_right = (
        (
            is_home_team & df["homeTeamStartLeft"] & is_second_half.not_()
        )  # home team attacking right in 1st half
        | (
            is_home_team & df["homeTeamStartLeft"].not_() & is_second_half
        )  # home team attacking right in 2nd half
        | (
            is_home_team.not_() & df["homeTeamStartLeft"] & is_second_half
        )  # away team attacking right in 2nd half
        | (
            is_home_team.not_() & df["homeTeamStartLeft"].not_() & is_second_half.not_()
        )  # away team attacking right in 1st half
    )

    return df.with_columns(
        pl.when(is_goal_right).then(105.0).otherwise(0).alias("x_goal"),
        pl.lit(34.0).alias("y_goal"),
    )
