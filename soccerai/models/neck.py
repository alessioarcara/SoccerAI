from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric_temporal.nn as pygt_nn
from torch_geometric.typing import Adj, OptTensor

from soccerai.models.typings import ReadoutType
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

    def __init__(self, glob_din: int, glob_dout: int, cfg: NeckConfig):
        super().__init__()
        self.readout = READOUT_AGGREGATIONS[cfg.readout]()
        self.global_proj = pyg_nn.Linear(glob_din, glob_dout)

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
        self.fusion = GraphGlobalFusion(glob_din, cfg.glob_dout, cfg)

        if self.mode == "node":
            self.grnn = pygt_nn.recurrent.GConvGRU(
                in_channels=backbone_dout + node_dim,
                out_channels=backbone_dout,
                K=1,
            )
        elif self.mode == "graph":
            self.rnn = nn.GRUCell(
                input_size=cfg.rnn_din,
                hidden_size=cfg.rnn_dout,
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fused = self.fusion(z, u, batch, batch_size)

        if self.mode == "node":
            h = self.grnn(
                torch.cat([z, x], dim=-1),
                edge_index,
                edge_weight,
                prev_h,
            )
            return fused, h

        if self.mode == "graph":
            h = self.rnn(fused, prev_h)
            return h, h

        raise RuntimeError(f"Unhandled mode: {self.mode}")
