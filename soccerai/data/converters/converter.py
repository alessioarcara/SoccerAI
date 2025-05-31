from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data


class GraphConverter(ABC):
    @abstractmethod
    def _create_edge_index(self) -> torch.Tensor:
        pass

    @abstractmethod
    def convert_dataframe_to_data_list(self):
        pass


class FullyConnectedGraphConverter(GraphConverter):
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

    def _create_edge_index(self) -> torch.Tensor:
        src = []
        dst = []
        for i in range(22):
            for j in range(22):
                if i != j:
                    src.append(i)
                    dst.append(j)
        return torch.tensor([src, dst], dtype=torch.long)


class BipartiteGraphConverter(GraphConverter):
    def convert_dataframe_to_data_list(
        self, df: pl.DataFrame
    ) -> Tuple[List[Data], List[str]]:
        data_list: list[Data] = []

        for _, event_df in df.group_by(["gameEventId", "possessionEventId"]):
            if event_df.height != 22:
                continue

            x_df = event_df.drop("gameEventId", "possessionEventId", "label")
            edge_idx = self._create_edge_index(x_df["team_home"])
            edge_attr = self._create_edge_attr(edge_index=edge_idx, x_df=x_df)

            x = torch.tensor(x_df.to_numpy(), dtype=torch.float32)
            y = torch.tensor(event_df["label"][0], dtype=torch.float32).view(1, 1)

            data_list.append(Data(x=x, edge_index=edge_idx, y=y, edge_attr=edge_attr))

        return data_list, x_df.columns

    def _create_edge_index(self, team: pl.Series) -> torch.Tensor:
        src = []
        dst = []
        for i in range(22):
            for j in range(22):
                if i != j and team[i] != team[j]:
                    src.append(i)
                    dst.append(j)

        return torch.tensor([src, dst], dtype=torch.long)

    def _create_edge_attr(self, edge_idx, x_df: pl.DataFrame) -> torch.Tensor:
        edge_attr = []
        for edge in edge_idx.transpose():
            src_player = x_df.row(edge[0], named=True)
            dst_player = x_df.row(edge_idx[1], named=True)
            players_distance = np.linalg.norm(
                [src_player["x"] - dst_player["x"], src_player["y"] - dst_player["y"]]
            )
            edge_attr.append(np.exp(-players_distance * src_player["goal_distance"]))
        return edge_attr


def create_graph_converter(connection_mode: str) -> GraphConverter:
    match connection_mode:
        case "fully_connected":
            return FullyConnectedGraphConverter()
        case _:
            raise ValueError("Invalid connection mode")
