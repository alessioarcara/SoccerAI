from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import polars as pl
import torch
from loguru import logger
from torch_geometric.data import Data
from torch_geometric.typing import (
    OptTensor,
)

# per-frame seconds to the next shot / penalty-area touch of the possession
# team and to the end of the period, carried to the graphs as attributes
TIMELINE_COLUMNS = ["time_to_shot", "time_to_box", "time_to_period_end"]


class GraphConverter(ABC):
    NUM_PLAYERS = 22
    GLOBAL_FEATURE_PREFIXES: ClassVar[list[str]] = [
        "possessionEventType",
        "frameTime",
        "duration",
    ]
    ID_COLUMNS: ClassVar[list[str]] = [
        "gameEventId",
        "possessionEventId",
        "event_index",
        "label",
        *TIMELINE_COLUMNS,
        "chain_id",
        "gameId",
        "jerseyNum",
        "node_id",
    ]

    @abstractmethod
    def _create_edges(
        self, x_df: pl.DataFrame
    ) -> tuple[torch.Tensor, OptTensor, OptTensor]:
        pass

    def convert_dataframe_to_data_list(
        self, df: pl.DataFrame
    ) -> tuple[list[Data], list[str]]:
        data_list: list[Data] = []
        skipped_frames = 0
        skipped_chains: set[int] = set()
        global_feature_cols = [
            c
            for c in df.columns
            if any(c.startswith(pref) for pref in self.GLOBAL_FEATURE_PREFIXES)
        ]
        feature_names = [
            c for c in df.columns if c not in self.ID_COLUMNS + global_feature_cols
        ]
        chain_nodes: dict[int, tuple] = {}
        chain_labels: dict[int, float] = {}
        chain_games: dict[int, int] = {}

        for _, event_df in df.group_by(
            ["gameId", "chain_id", "gameEventId", "possessionEventId"],
            maintain_order=True,
        ):
            chain_id = int(event_df["chain_id"][0])
            game_id = int(event_df["gameId"][0])
            if chain_games.setdefault(chain_id, game_id) != game_id:
                skipped_chains.add(chain_id)
                continue
            if event_df.height != self.NUM_PLAYERS:
                skipped_frames += 1
                skipped_chains.add(chain_id)
                continue

            if "node_id" in event_df.columns:
                event_df = event_df.sort("node_id")
                node_ids = tuple(event_df["node_id"].to_list())
                if (
                    len(set(node_ids)) != self.NUM_PLAYERS
                    or None in node_ids
                    or chain_nodes.setdefault(chain_id, node_ids) != node_ids
                ):
                    skipped_chains.add(chain_id)
                    continue
            if (
                "is_possession_team_1" in event_df.columns
                and event_df["is_possession_team_1"].sum() != 11
            ):
                skipped_chains.add(chain_id)
                continue

            jersey_series = event_df["jerseyNum"].cast(pl.Int64)

            node_df = event_df.select(feature_names)

            global_df = event_df.select(global_feature_cols).head(1)

            event_index = int(event_df["event_index"][0])
            label = float(event_df["label"][0])
            if (
                event_df["label"].n_unique() != 1
                or label not in (0.0, 1.0)
                or chain_labels.setdefault(chain_id, label) != label
                or not np.isfinite(node_df.to_numpy()).all()
                or not np.isfinite(global_df.to_numpy()).all()
            ):
                skipped_chains.add(chain_id)
                continue

            edge_idx, edge_weight, edge_attr = self._create_edges(node_df)

            x = torch.tensor(node_df.to_numpy(), dtype=torch.float32)
            u = torch.tensor(global_df.to_numpy(), dtype=torch.float32)
            y = torch.tensor(label, dtype=torch.float32).view(1, 1)
            jersey_numbers = torch.tensor(jersey_series.to_numpy(), dtype=torch.long)
            timeline = {
                c: float(event_df[c][0])
                for c in TIMELINE_COLUMNS
                if c in event_df.columns
            }

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
                    **timeline,
                )
            )

        if skipped_chains:
            logger.warning(
                "Discarded {} chain(s) with invalid frames ({} frame(s) without exactly {} players)",
                len(skipped_chains),
                skipped_frames,
                self.NUM_PLAYERS,
            )
            data_list = [
                data for data in data_list if data.chain_id not in skipped_chains
            ]

        return data_list, feature_names


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
        if not all(
            np.isfinite(v) and v > 0 for v in (length_scale, pitch_length, pitch_width)
        ):
            raise ValueError(
                "length_scale and pitch dimensions must be positive and finite"
            )
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
