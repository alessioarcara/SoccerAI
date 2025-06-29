from typing import Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric_temporal.nn as pygt_nn
from torch_geometric.typing import Adj, OptTensor

from soccerai.models.typings import ReadoutType, RNNType
from soccerai.training.trainer_config import NeckConfig

READOUT_AGGREGATIONS: Dict[ReadoutType, Type[pyg_nn.Aggregation]] = {
    "sum": pyg_nn.SumAggregation,
    "mean": pyg_nn.MeanAggregation,
    "max": pyg_nn.MaxAggregation,
}

GRNN_CELLS: Dict[RNNType, Type[nn.Module]] = {
    "gru": pygt_nn.recurrent.GConvGRU,
    "lstm": pygt_nn.recurrent.GConvLSTM,
}

RNN_CELLS: Dict[RNNType, Type[nn.Module]] = {"gru": nn.GRUCell, "lstm": nn.LSTMCell}


class GraphGlobalFusion(nn.Module):
    """
    Fuse graph-level and global feature vectors into one concatenated vector:
    1. Readout over nodes -> graph embedding
    2. Linear projection + ReLU -> global embedding
    3. Concatenate [graph || global]
    """

    def __init__(self, glob_din: int, cfg: NeckConfig):
        super().__init__()
        self.readout = READOUT_AGGREGATIONS[cfg.readout]()
        self.global_proj = pyg_nn.Linear(glob_din, cfg.glob_dout)

    def forward(
        self, z: torch.Tensor, u: torch.Tensor, batch: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        z_list = z if isinstance(z, list) else [z]

        graph_embs = [
            self.readout(x=z, index=batch, dim_size=batch_size) for z in z_list
        ]
        graph_emb = torch.cat(graph_embs, dim=1)
        glob_emb = F.relu(self.global_proj(u), inplace=True)
        return torch.cat([graph_emb, glob_emb], dim=-1)


class TemporalFusion(nn.Module):
    """
    Apply temporal and fusion operations on graph and global features.

    Modes:
    - "node":
        1) Apply temporal over node embeddings.
        2) Fuse graph and global features.
    - "graph":
        1) Fuse graph and global features.
        2) Apply temporal over the fused vectors.
    """

    def __init__(
        self,
        backbone_dout: int,
        node_dim: int,
        glob_din: int,
        cfg: NeckConfig,
    ):
        super().__init__()
        self.mode = cfg.mode
        self.fusion = GraphGlobalFusion(glob_din, cfg)

        self.raw_features_proj: nn.Module = nn.Identity()
        if self.mode == "node":
            if cfg.raw_features_proj:
                self.raw_features_proj = nn.Sequential(
                    pyg_nn.Linear(node_dim, cfg.proj_dout), nn.ReLU()
                )
                grnn_din = backbone_dout + cfg.proj_dout

            else:
                grnn_din = backbone_dout + node_dim
            self.grnn = GRNN_CELLS[cfg.rnn_type](
                in_channels=grnn_din, out_channels=backbone_dout, K=1
            )

        elif self.mode == "graph":
            self.rnn = RNN_CELLS[cfg.rnn_type](
                input_size=cfg.rnn_din, hidden_size=cfg.rnn_dout
            )

        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def forward(
        self,
        z: torch.Tensor,
        u: torch.Tensor,
        x: torch.Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
        prev_h: OptTensor = None,
        prev_c: OptTensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_proj = self.raw_features_proj(x)

        fused = self.fusion(z, u, batch, batch_size)

        if self.mode == "node":
            grnn_input = torch.cat([z, x_proj], dim=-1)

            if isinstance(self.grnn, pygt_nn.GConvLSTM):
                h, c = self.grnn(
                    grnn_input,
                    edge_index,
                    edge_weight,
                    prev_h,
                    prev_c,
                )

            elif isinstance(self.grnn, pygt_nn.GConvGRU):
                h = self.grnn(
                    grnn_input,
                    edge_index,
                    edge_weight,
                    prev_h,
                )
                c = None

            return fused, h, c

        else:  # graph
            if isinstance(self.rnn, nn.LSTMCell):
                state: Optional[Tuple[OptTensor, OptTensor]] = (prev_h, prev_c)
                if prev_h is None or prev_c is None:
                    state = None

                h, c = self.rnn(fused, state)

            elif isinstance(self.rnn, nn.GRUCell):
                h = self.rnn(fused, prev_h)
                c = None

            return h, h, c
