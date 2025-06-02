import json
from pathlib import Path
from typing import Callable, List, Optional

import polars as pl
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch_geometric.data import InMemoryDataset

from soccerai.data.config import X_GOAL_LEFT, X_GOAL_RIGHT, Y_GOAL
from soccerai.data.converters import GraphConverter
from soccerai.data.transformers import (
    GoalLocationTransformer,
    PlayerPositionTransformer,
)
from soccerai.data.utils import reorder_dataframe_cols
from soccerai.training.trainer_config import TrainerConfig


class WorldCup2022Dataset(InMemoryDataset):
    FEATURE_NAMES_FILE = "feature_names.json"

    def __init__(
        self,
        root: str,
        converter: GraphConverter,
        split: str,
        cfg: TrainerConfig,
        transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        self.converter = converter
        self.split = split
        self.cfg = cfg
        super().__init__(root=root, transform=transform, force_reload=force_reload)

        data_path_idx = 0 if self.split == "train" else 1
        self.load(self.processed_paths[data_path_idx])

        fp = Path(self.processed_dir) / self.FEATURE_NAMES_FILE
        self.feature_names = (
            json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else None
        )

    @property
    def raw_file_names(self) -> List[str]:
        return ["dataset.parquet"]

    @property
    def processed_file_names(self) -> List[str]:
        return ["train_data.pt", "val_data.pt"]

    @property
    def num_global_features(self) -> int:
        try:
            return self[0].u.shape[-1]
        except (IndexError, AttributeError):
            return 0

    def _split_dataframe_by_event_keys(
        self,
        df: pl.DataFrame,
        key_cols: list[str] = ["gameEventId", "possessionEventId"],
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        event_keys_df = df.select(key_cols).unique().sort(key_cols)
        n_events = event_keys_df.height
        cut = int((1.0 - self.cfg.val_ratio) * n_events)
        train_keys = event_keys_df.slice(0, cut)
        val_keys = event_keys_df.slice(cut, n_events)

        train_df = df.join(train_keys, on=key_cols, how="semi")
        val_df = df.join(val_keys, on=key_cols, how="semi")
        return train_df, val_df

    def _prepare_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        cols_to_drop = [
            "gameEventType",
            "index",
            "gameId",
            "startTime",
            "endTime",
            "index_right",
            "gameId_right",
            "jerseyNum",
            "visibility",
            "z",
            "videoUrl",
            "homeTeamName",
            "awayTeamName",
            "Age Info",
            "Full Name",
            "Height",
            "birth_date",
            "teamName",
            "playerId",
        ]
        df = df.drop(cols_to_drop)

        df = (
            df.filter(pl.col("team").is_not_null())
            .filter(pl.col("possessionEventType") != "SH")
            .filter(pl.col("playerName").is_not_null())
        )

        df = df.with_columns(
            [
                # (mm:ss) → s
                (
                    (
                        pl.col("frameTime").str.split(":").list.get(0).cast(pl.UInt16)
                        * 60
                        + pl.col("frameTime").str.split(":").list.get(1).cast(pl.UInt16)
                    ).alias("frameTime")
                ),
                (
                    pl.when(pl.col("playerName") == pl.col("playerName_right"))
                    .then(1)
                    .otherwise(0)
                ).alias("is_ball_carrier"),
                (pl.col("Weight").str.replace("kg", "").cast(pl.Float64)),
                (pl.col("height_cm").cast(pl.Float64)),
                (
                    pl.when(pl.col("age") < 20)
                    .then(pl.lit("Under 20"))
                    .when(pl.col("age") < 29)
                    .then(pl.lit("20-28"))
                    .when(pl.col("age") < 35)
                    .then(pl.lit("29-35"))
                    .otherwise(pl.lit("35+"))
                    .alias("age")
                ),
            ]
        ).drop(["playerName", "playerName_right"])

        possession_team = df.filter(pl.col("is_ball_carrier") == 1)["team"][0]

        df = df.with_columns(
            [
                (
                    pl.when(pl.col("team") == possession_team)
                    .then(1)
                    .otherwise(0)
                    .alias("is_possession_team")
                ),
            ]
        )

        if self.cfg.use_goal_features:
            is_home_team = df["team"] == "home"
            is_second_half = df["frameTime"] > df["startPeriod2"]

            is_goal_right = (
                (is_home_team & df["homeTeamStartLeft"] & ~is_second_half)
                | (is_home_team & ~df["homeTeamStartLeft"] & is_second_half)
                | (~is_home_team & ~df["homeTeamStartLeft"] & ~is_second_half)
                | (~is_home_team & df["homeTeamStartLeft"] & is_second_half)
            )

            df = df.with_columns(
                [
                    pl.when(is_goal_right)
                    .then(X_GOAL_RIGHT)
                    .otherwise(X_GOAL_LEFT)
                    .alias("x_goal"),
                    pl.lit(Y_GOAL).alias("y_goal"),
                ]
            )

        df = df.drop(["team", "homeTeamStartLeft", "startPeriod2"])

        return df

    def _create_preprocessor(self, df: pl.DataFrame) -> ColumnTransformer:
        cat_cols = [
            "possessionEventType",
            "playerRole",
            "is_possession_team",
            "is_ball_carrier",
            "age",
        ]
        pos_cols = ["x", "y"]
        exclude_cols = set(
            cat_cols + pos_cols + ["gameEventId", "possessionEventId", "label"]
        )
        if self.cfg.use_goal_features:
            goal_cols = ["x_goal", "y_goal"]
            exclude_cols.update(goal_cols)
        num_cols = [c for c in df.columns if c not in exclude_cols]

        num_pipe = Pipeline(
            [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
        )
        cat_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore", drop="if_binary", sparse_output=False
                    ),
                ),
            ]
        )

        transformers = [
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
            ("player_pos", PlayerPositionTransformer(), pos_cols),
        ]

        if self.cfg.use_goal_features:
            transformers.append(
                ("goal_loc", GoalLocationTransformer(), pos_cols + goal_cols)
            )

        prep = ColumnTransformer(
            transformers,
            remainder="passthrough",
            verbose_feature_names_out=False,  # No prefixes
        )

        prep.set_output(transform="polars")
        return prep

    def process(self):
        df = self._prepare_dataframe(pl.read_parquet(self.raw_paths[0]))

        train_df, val_df = self._split_dataframe_by_event_keys(df)

        logger.info(
            "DataFrame split → train: {} rows, val: {} rows",
            train_df.height,
            val_df.height,
        )

        self.preprocessor = self._create_preprocessor(train_df)

        first = ["x", "y", "is_possession_team_1", "is_ball_carrier_1"]
        train_transformed = reorder_dataframe_cols(
            self.preprocessor.fit_transform(train_df), first
        )
        val_transformed = reorder_dataframe_cols(
            self.preprocessor.transform(val_df), first
        )

        train_data_list, feature_names = self.converter.convert_dataframe_to_data_list(
            train_transformed
        )
        val_data_list, _ = self.converter.convert_dataframe_to_data_list(
            val_transformed
        )

        self.save(train_data_list, self.processed_paths[0])
        self.save(val_data_list, self.processed_paths[1])

        fp = Path(self.processed_dir) / self.FEATURE_NAMES_FILE
        fp.write_text(json.dumps(feature_names, ensure_ascii=False, indent=4))
