from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric_temporal.nn as pygt_nn
from torch_geometric.typing import Adj, OptTensor

from soccerai.models.typings import ReadoutType, TemporalMode
from soccerai.training.trainer_config import NeckConfig

READOUT_AGGREGATIONS: Dict[ReadoutType, pyg_nn.Aggregation] = {
    "sum": pyg_nn.SumAggregation,
    "mean": pyg_nn.MeanAggregation,
    "max": pyg_nn.MaxAggregation,
}


class GraphGlobalFusion(nn.Module):
    """
    Fuse graph-level and global feature vectors into one concatenated vector:
    1. Readout over nodes -> graph embedding
    2. Linear projection + ReLU -> global embedding
    3. Concatenate [graph || global]
    """

    def __init__(self, backbone_dout: int, glob_din: int, cfg: NeckConfig):
        super().__init__()
        self.readout = READOUT_AGGREGATIONS[cfg.readout]()
        self.global_proj = pyg_nn.Linear(glob_din, backbone_dout)

    def forward(
        self, z: torch.Tensor, u: torch.Tensor, batch: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        graph_emb = self.readout(x=z, index=batch, dim_size=batch_size)
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
        node_dim: int,
        backbone_dout: int,
        glob_din: int,
        cfg: NeckConfig,
        mode: TemporalMode = "node",
    ):
        super().__init__()
        self.mode = mode

        self.grnn = pygt_nn.recurrent.GConvGRU(
            in_channels=backbone_dout + node_dim,
            out_channels=cfg.dhid,
            K=1,
        )
        self.fusion = GraphGlobalFusion(backbone_dout, glob_din, cfg)
        self.rnn = nn.GRUCell(input_size=backbone_dout * 2, hidden_size=cfg.dhid)

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "node":
            h = self.grnn(
                torch.concat([z, x], dim=-1),
                edge_index,
                edge_weight,
                prev_h,
            )
            fused = self.fusion(z, u, batch, batch_size)
            return fused, h

        else:  # graph
            fused = self.fusion(z, u, batch, batch_size)
            h = self.rnn(fused, prev_h)
            return h, h
