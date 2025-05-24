from typing import List

import polars as pl
import torch
from torch_geometric.data import Data
from typing_extensions import assert_never

from soccerai.data.converters import ConnectionMode, GraphConverter
from soccerai.data.converters.utils import add_goal_features


class ShotPredictionGraphConverter(GraphConverter):
    def _create_edge_index(self) -> torch.Tensor:
        if self.mode == ConnectionMode.FULLY_CONNECTED:
            src = []
            dst = []
            for i in range(22):
                for j in range(22):
                    if i != j:
                        src.append(i)
                        dst.append(j)
            return torch.tensor([src, dst], dtype=torch.long)
        assert_never(self.mode)

    def _preprocess_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        columns_to_drop = [
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
        df = df.drop(columns_to_drop)

        df = df.filter(pl.col("team").is_not_null())
        df = df.filter(pl.col("possessionEventType") != "SH")
        df = df.filter(pl.col("playerName").is_not_null())

        df = df.with_columns(
            [
                (
                    pl.col("frameTime").str.split(":").list.get(0).cast(pl.UInt16) * 60
                    + pl.col("frameTime").str.split(":").list.get(1).cast(pl.UInt16)
                ).alias("frameTime"),
                (
                    pl.when(pl.col("playerName") == pl.col("playerName_right"))
                    .then(1)
                    .otherwise(0)
                ).alias("ballPossession"),
                (pl.col("Weight").str.replace("kg", "").cast(pl.UInt8)),
                (pl.col("height_cm").cast(pl.UInt8)),
            ]
        )

        df = add_goal_features(df)

        df = df.drop(["playerName", "playerName_right"])
        categorical_cols = ["possessionEventType", "team", "playerRole"]
        exclude_from_norm = set(
            categorical_cols
            + ["gameEventId", "possessionEventId", "label", "ballPossession"]
        )

        cols_to_norm = [c for c in df.columns if c not in exclude_from_norm]

        df = df.with_columns(
            [
                (
                    (pl.col(c).fill_null(strategy="mean") - pl.col(c).mean())
                    / pl.col(c).std()
                ).alias(c)
                for c in cols_to_norm
            ]
        )

        df = df.to_dummies(categorical_cols)

        df = df.select(pl.col("x"), pl.col("y"), pl.all().exclude(["x", "y"]))

        return df

    def convert_dataframe_to_data_list(self, df: pl.DataFrame) -> List[Data]:
        processed_df = self._preprocess_dataframe(df)

        data_list = []

        for _, event_df in processed_df.group_by(["gameEventId", "possessionEventId"]):
            if event_df.height != 22:
                continue

            event_df_x = event_df.drop("gameEventId", "possessionEventId", "label")
            edge_idx = self._create_edge_index()

            x = torch.tensor(event_df_x.to_numpy(), dtype=torch.float32)
            y = torch.tensor(event_df["label"][0], dtype=torch.float32).view(1, 1)

            data_list.append(
                Data(
                    x=x,
                    edge_index=edge_idx,
                    y=y,
                )
            )
        return data_list
