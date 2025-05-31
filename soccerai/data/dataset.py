from typing import Callable, Optional

import polars as pl
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from torch_geometric.data import InMemoryDataset

from soccerai.data.converters import GraphConverter
from soccerai.data.converters.utils import get_goal_positions
from soccerai.data.utils import reorder_dataframe_cols
from soccerai.training.trainer_config import TrainerConfig


class WorldCup2022Dataset(InMemoryDataset):
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

    @property
    def raw_file_names(self):
        return ["dataset.parquet"]

    @property
    def processed_file_names(self):
        return ["train_data.pt", "val_data.pt"]

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

    def _clean_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
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
                ).alias("ballPossession"),
                (pl.col("Weight").str.replace("kg", "").cast(pl.UInt8)),
                (pl.col("height_cm").cast(pl.UInt8)),
            ]
        ).drop(["playerName", "playerName_right"])
        if self.cfg.use_goal_features:
            df = self._add_goal_features(df)
        return df

    def _create_preprocessor(self, df: pl.DataFrame) -> ColumnTransformer:
        cat_cols = ["possessionEventType", "team", "playerRole", "ballPossession"]
        coord_cols = ["x", "y"]
        exclude_cols = set(
            cat_cols + coord_cols + ["gameEventId", "possessionEventId", "label"]
        )
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
                        handle_unknown="ignore", sparse_output=False, drop="if_binary"
                    ),
                ),
            ]
        )
        coord_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("minmax_scaler", MinMaxScaler(feature_range=(-1, 1))),
            ]
        )

        prep = ColumnTransformer(
            [
                ("num", num_pipe, num_cols),
                ("cat", cat_pipe, cat_cols),
                ("coord", coord_pipe, coord_cols),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,  # No prefixes
        )

        prep.set_output(transform="polars")
        return prep

    def _add_goal_features(self, df: pl.DataFrame) -> pl.DataFrame:
        df = get_goal_positions(df)
        df = df.with_columns(
            (
                (
                    (pl.col("x") - pl.col("x_goal")) ** 2
                    + (pl.col("y") - pl.col("y_goal")) ** 2
                )
                ** 0.5
            ).alias("goal_distance")
        )
        df = df.with_columns(
            pl.arctan2(pl.col("y_goal") - pl.col("y"), pl.col("x_goal") - pl.col("x"))
            .degrees()
            .alias("goal_angle")
        )

        return df.drop("x_goal", "y_goal")

    def process(self):
        df_raw = pl.read_parquet(self.raw_paths[0])
        df_clean = self._clean_dataframe(df_raw)

        train_df, val_df = self._split_dataframe_by_event_keys(df_clean)

        logger.info(
            "DataFrame split → train: {} rows, val: {} rows",
            train_df.height,
            val_df.height,
        )

        self.preprocessor = self._create_preprocessor(train_df)

        first = ["x", "y", "team_home", "ballPossession_1"]
        train_transformed = reorder_dataframe_cols(
            self.preprocessor.fit_transform(train_df), first
        )
        val_transformed = reorder_dataframe_cols(
            self.preprocessor.transform(val_df), first
        )

        train_data_list = self.converter.convert_dataframe_to_data_list(
            train_transformed
        )
        val_data_list = self.converter.convert_dataframe_to_data_list(val_transformed)

        self.save(train_data_list, self.processed_paths[0])
        self.save(val_data_list, self.processed_paths[1])
