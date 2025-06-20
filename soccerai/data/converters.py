from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data
from torch_geometric.typing import (
    OptTensor,
)


class GraphConverter(ABC):
    NUM_PLAYERS = 22
    GLOBAL_FEATURE_PREFIXES = ["possessionEventType", "frameTime", "duration"]

    @abstractmethod
    def _create_edges(
        self, x_df: pl.DataFrame
    ) -> Tuple[torch.Tensor, OptTensor, OptTensor]:
        pass

    def convert_dataframe_to_data_list(
        self, df: pl.DataFrame
    ) -> Tuple[List[Data], List[str]]:
        data_list: List[Data] = []

        for _, event_df in df.group_by(["gameEventId", "possessionEventId"]):
            if event_df.height != self.NUM_PLAYERS:
                continue

            global_feature_cols = [
                c
                for c in event_df.columns
                if any(c.startswith(pref) for pref in self.GLOBAL_FEATURE_PREFIXES)
            ]

            jersey_series = event_df["jerseyNum"].cast(pl.Int64)

            node_df = event_df.drop(
                *[
                    "gameEventId",
                    "possessionEventId",
                    "label",
                    "chain_id",
                    "gameId",
                    "jerseyNum",
                ],
                *global_feature_cols,
            )

            global_df = event_df.select(global_feature_cols).head(1)

            chain_id = int(event_df["chain_id"][0])
            frame_time = float(event_df["frameTime"][0])
            label = float(event_df["label"][0])

            edge_idx, edge_weight, edge_attr = self._create_edges(node_df)

            x = torch.tensor(node_df.to_numpy(), dtype=torch.float32)
            u = torch.tensor(global_df.to_numpy(), dtype=torch.float32)
            y = torch.tensor(label, dtype=torch.float32).view(1, 1)
            jersey_numbers = torch.tensor(jersey_series.to_numpy(), dtype=torch.long)

            data_list.append(
                Data(
                    x=x,
                    edge_index=edge_idx,
                    edge_weight=edge_weight,
                    edge_attr=edge_attr,
                    u=u,
                    y=y,
                    chain_id=chain_id,
                    frame_time=frame_time,
                    jersey_numbers=jersey_numbers,
                )
            )

        return data_list, node_df.columns


class FullyConnectedGraphConverter(GraphConverter):
    def _create_edges(
        self, x_df: pl.DataFrame
    ) -> Tuple[torch.Tensor, OptTensor, OptTensor]:
        src, dst = [], []

        for i in range(self.NUM_PLAYERS):
            for j in range(self.NUM_PLAYERS):
                if i != j:
                    src.append(i)
                    dst.append(j)

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_weight = edge_attr = None
        return edge_index, edge_weight, edge_attr


class BipartiteGraphConverter(GraphConverter):
    """
    Builds a bipartite graph where each player is connected to every opponent,
    using edge weights that reflect their proximity.

    When extended to a two-hop view, this construction captures:
      • How many opposing players are nearby (first-hop).
      • How many of a player's own teammates are near those opponents (second-hop).

    In this way, each player's embedding can reflect both direct proximity to
    opponents and the local defensive/offensive support structure.
    """

    def _create_edges(
        self, x_df: pl.DataFrame
    ) -> Tuple[torch.Tensor, OptTensor, OptTensor]:
        positions = x_df.select(["x", "y"]).to_numpy()
        teams = x_df["is_possession_team_1"].to_numpy()

        src, dst, weights = [], [], []

        for i in range(self.NUM_PLAYERS):
            a_values = []
            valid_j = []
            for j in range(self.NUM_PLAYERS):
                if i != j and teams[i] != teams[j]:
                    players_distance = np.linalg.norm(positions[i] - positions[j])
                    a_values.append(np.exp(-players_distance))
                    valid_j.append(j)
            if a_values:
                denominator = np.sum(a_values)
                for j, a in zip(valid_j, a_values):
                    src.append(i)
                    dst.append(j)
                    weights.append(a / denominator)

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_weight = edge_attr = torch.tensor(weights, dtype=torch.float32)
        return edge_index, edge_weight, edge_attr


def create_graph_converter(connection_mode: str) -> GraphConverter:
    match connection_mode:
        case "fully_connected":
            return FullyConnectedGraphConverter()
        case "bipartite":
            return BipartiteGraphConverter()
        case _:
            raise ValueError("Invalid connection mode")
