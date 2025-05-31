from typing import List, Tuple

import polars as pl
import torch
from torch_geometric.data import Data
from typing_extensions import assert_never

from soccerai.data.converters import ConnectionMode, GraphConverter


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

    def convert_dataframe_to_data_list(
        self, df: pl.DataFrame
    ) -> Tuple[List[Data], List[str]]:
        data_list: list[Data] = []

        for _, event_df in df.group_by(["gameEventId", "possessionEventId"]):
            if event_df.height != 22:
                continue

            x_df = event_df.drop("gameEventId", "possessionEventId", "label")
            edge_idx = self._create_edge_index()

            x = torch.tensor(x_df.to_numpy(), dtype=torch.float32)
            y = torch.tensor(event_df["label"][0], dtype=torch.float32).view(1, 1)

            data_list.append(Data(x=x, edge_index=edge_idx, y=y))

        return data_list, x_df.columns
