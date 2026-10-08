from abc import ABC, abstractmethod

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
    ) -> tuple[torch.Tensor, OptTensor, OptTensor]:
        pass

    def convert_dataframe_to_data_list(
        self, df: pl.DataFrame
    ) -> tuple[list[Data], list[str]]:
        data_list: list[Data] = []

        for _, event_df in df.group_by(
            ["gameEventId", "possessionEventId"], maintain_order=True
        ):
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
                    "event_index",
                    "label",
                    "chain_id",
                    "gameId",
                    "jerseyNum",
                ],
                *global_feature_cols,
            )

            global_df = event_df.select(global_feature_cols).head(1)

            chain_id = int(event_df["chain_id"][0])
            event_index = int(event_df["event_index"][0])
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
                    event_index=event_index,
                    jersey_numbers=jersey_numbers,
                )
            )

        return data_list, node_df.columns


class FullyConnectedGraphConverter(GraphConverter):
    def _create_edges(
        self, x_df: pl.DataFrame
    ) -> tuple[torch.Tensor, OptTensor, OptTensor]:
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

    Weights are `exp(-d / length_scale)` with `d` the distance in metres, so
    that an opponent at `length_scale` metres weighs 1/e of a marking one;
    they are left unnormalised (GCN applies its own symmetric normalisation,
    the other layers consume them as edge attributes).
    """

    def __init__(
        self,
        length_scale: float = 10.0,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ):
        self.length_scale = length_scale
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

    def _create_edges(
        self, x_df: pl.DataFrame
    ) -> tuple[torch.Tensor, OptTensor, OptTensor]:
        # node coordinates are already scaled to [0, 1]: back to metres, so
        # that the proximity weight has a physical length scale
        positions = x_df.select(["x", "y"]).to_numpy() * np.array(
            [self.pitch_length, self.pitch_width]
        )
        teams = x_df["is_possession_team_1"].to_numpy()

        distances = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :], axis=-1
        )
        opponents = teams[:, None] != teams[None, :]
        src, dst = np.nonzero(opponents)
        weights = np.exp(-distances[src, dst] / self.length_scale)

        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        edge_weight = edge_attr = torch.tensor(weights, dtype=torch.float32)
        return edge_index, edge_weight, edge_attr


def create_graph_converter(
    connection_mode: str, edge_length_scale: float = 10.0
) -> GraphConverter:
    match connection_mode:
        case "fully_connected":
            return FullyConnectedGraphConverter()
        case "bipartite":
            return BipartiteGraphConverter(length_scale=edge_length_scale)
        case _:
            raise ValueError("Invalid connection mode")
