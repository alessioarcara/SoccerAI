import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Optional

import polars as pl
import torch
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from torch_geometric.data import Data, InMemoryDataset

from soccerai.data.config import X_GOAL_LEFT, X_GOAL_RIGHT, Y_GOAL
from soccerai.data.converters import GraphConverter
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
    def raw_file_names(self):
        return ["dataset.parquet"]

    @property
    def processed_file_names(self):
        return ["train_data.pt", "val_data.pt"]

    def _split_by_game_id(
        self, df: pl.DataFrame, val_ratio: float
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Split the DataFrame into training and validation sets based on gameId."""

        game_ids_df = df.select(["gameId"]).unique().sort("gameId")
        n_games = game_ids_df.height

        n_train_games = int((1.0 - val_ratio) * n_games)

        train_game_ids = game_ids_df.slice(0, n_train_games)
        val_game_ids = game_ids_df.slice(n_train_games, n_games)

        train_df = df.join(train_game_ids, on="gameId", how="semi")
        val_df = df.join(val_game_ids, on="gameId", how="semi")

        # -------------------------------------------------------
        print(">>> After split by gameId:")
        print(f"Number of rows in train_df: {train_df.height}")
        print(
            f"Number of games in train_df: {train_df.select('gameId').unique().height}"
        )
        print(f"Number of rows in val_df:   {val_df.height}")
        print(f"Number of games in val_df: {val_df.select('gameId').unique().height}")

        return train_df, val_df

    def _prepare_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        cols_to_drop = [
            "gameEventType",
            "index",
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
            cat_cols
            + coord_cols
            + [
                "gameEventId",
                "possessionEventId",
                "label",
                "gameId",
                "frameTime",
                "sequence_id",
            ]
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
                is_home_team.not_()
                & df["homeTeamStartLeft"].not_()
                & is_second_half.not_()
            )  # away team attacking right in 1st half
        )
        df = df.with_columns(
            pl.when(is_goal_right)
            .then(X_GOAL_RIGHT)
            .otherwise(X_GOAL_LEFT)
            .alias("x_goal"),
            pl.lit(Y_GOAL).alias("y_goal"),
        )

        df = df.with_columns(
            (
                (
                    (pl.col("x") - pl.col("x_goal")) ** 2
                    + (pl.col("y") - pl.col("y_goal")) ** 2
                )
                ** 0.5
            ).alias("goal_distance"),
            pl.arctan2(pl.col("y_goal") - pl.col("y"), pl.col("x_goal") - pl.col("x"))
            .degrees()
            .alias("goal_angle"),
        )
        return df.drop("x_goal", "y_goal", "homeTeamStartLeft", "startPeriod2")

    def _group_by_sequence_id(self, data_list: List[Data]) -> List[List[Data]]:
        """ """
        # build a dictionary: key = sequence_id, value = list of Data
        buckets: dict[int, List[Data]] = defaultdict(list)
        for data in data_list:
            seq_id = int(data.sequence_id.item())
            buckets[seq_id].append(data)

        # sort each list by ascending frameTime and append it to sequences
        sequences: List[List[Data]] = []
        for seq_id, items in buckets.items():
            items.sort(key=lambda d: float(d.frameTime.item()))
            sequences.append(items)

        return sequences

    def process(self):
        df = self._prepare_dataframe(pl.read_parquet(self.raw_paths[0]))

        train_df, val_df = self._split_by_game_id(df, val_ratio=0.2)

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

        # -------------------------------------------------------
        print(">>> After pre-processing:")
        print(
            f"  train_transformed: {train_transformed.height} rows, "
            f"{len(train_transformed.columns)} columns"
        )
        print("  Columns of train_transformed:", train_transformed.columns)
        print(train_transformed.head(3))

        print(
            f"  val_transformed:   {val_transformed.height} rows, "
            f"{len(val_transformed.columns)} columns"
        )
        print("  Columns of val_transformed:", val_transformed.columns)
        print(val_transformed.head(3))
        # --------------------------------------------------------------------

        train_data_list, feature_names = self.converter.convert_dataframe_to_data_list(
            train_transformed
        )

        # -------------------------------------------------------
        print(">>> After convert_dataframe_to_data_list(train):")
        print(
            f"  Number of Data objects generated (train_data_list): {len(train_data_list)}"
        )
        for i, data in enumerate(train_data_list[:2]):
            print(f"--- Data train #{i} ---")
            print("  x.shape:", data.x.shape)  # (NUM_PLAYERS, n_features)
            print("  edge_index.shape:", data.edge_index.shape)  # (2, n_edges)
            print("  y (label):", data.y.item())  # 0 or 1
            print("  sequence_id:", data.sequence_id.item())  # integer
            print("  frameTime:", data.frameTime.item())

        val_data_list, _ = self.converter.convert_dataframe_to_data_list(
            val_transformed
        )

        train_sequences = self._group_by_sequence_id(train_data_list)
        val_sequences = self._group_by_sequence_id(val_data_list)

        print(">>> After group_by_sequence_id:")
        print(f"  Number of distinct sequence_id in train: {len(train_sequences)}")
        for i in range(min(3, len(train_sequences))):
            seq = train_sequences[i]
            print(
                f"   - Train sequence [{i}] – sequence_id = {seq[0].sequence_id.item()}, "
                f"n_frames = {len(seq)}, "
                f"frameTimes = {[d.frameTime.item() for d in seq[:5]]} ..."
            )

        print(f"  Number of distinct sequence_id in val:   {len(val_sequences)}")
        for i in range(min(3, len(val_sequences))):
            seq = val_sequences[i]
            print(
                f"   - Val sequence [{i}] – sequence_id = {seq[0].sequence_id.item()}, "
                f"n_frames = {len(seq)}, "
                f"frameTimes = {[d.frameTime.item() for d in seq[:5]]} ..."
            )

        train_out = os.path.join(self.root, "train_data.pt")
        val_out = os.path.join(self.root, "val_data.pt")

        print(
            f"  Saving train_sequences ({len(train_sequences)} sequences) to {train_out}"
        )
        torch.save(train_sequences, train_out)

        print(
            f"  Saving val_sequences   ({len(val_sequences)}   sequences) to {val_out}"
        )
        torch.save(val_sequences, val_out)

        fp = Path(self.processed_dir) / self.FEATURE_NAMES_FILE
        fp.write_text(json.dumps(feature_names, ensure_ascii=False, indent=4))
