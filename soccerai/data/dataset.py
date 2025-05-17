from typing import List, Optional

import polars as pl
import torch
from torch_geometric.data import Data, InMemoryDataset


class WorldCup2022Dataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        mode: str,
        transform: Optional[callable] = None,
    ):
        self.mode = mode
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return "dataset.parquet"

    @property
    def processed_file_names(self):
        return "data.pt"

    def process(self):
        df = pl.read_parquet(self.raw_paths[0])
        df = self._preprocess_dataframe(df)
        data_list = self._convert_dataframe_to_data_list(df)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def create_edge_index(self) -> torch.Tensor:
        if self.mode == "fully_connected":
            src = []
            dst = []
            for i in range(22):
                src += [i] * 21
                for j in range(22):
                    if i != j:
                        dst += [j]
            return torch.tensor([src, dst], dtype=torch.long)

        raise NotImplementedError(f"Connection mode {self.mode.value} not implemented")

    def _preprocess_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.drop(
            [
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
        )

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

        df = df.drop(["playerName", "playerName_right"])

        df = df.to_dummies(["possessionEventType", "team", "playerRole"])

        df = df.fill_null(strategy="mean")

        return df

    def _convert_dataframe_to_data_list(self, df: pl.DataFrame) -> List[Data]:
        data_list = []

        for _, event_df in df.group_by(["gameEventId", "possessionEventId"]):
            if event_df.height != 22:
                continue
            event_df_x = event_df.drop("gameEventId", "possessionEventId", "label")
            edge_idx = self.create_edge_index()

            x = []
            for i in range(22):
                player_row = event_df_x.row(i, named=True)
                x.append(list(player_row.values()))

            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor(event_df["label"][0], dtype=torch.long)

            data_list.append(
                Data(
                    x=x,
                    edge_index=edge_idx,
                    y=y,
                )
            )

        return data_list
