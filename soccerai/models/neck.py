from typing import Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric_temporal.nn as pygt_nn
from torch_geometric.typing import Adj, OptTensor

from soccerai.models.typings import NeckType, ReadoutType
from soccerai.training.trainer_config import NeckConfig

READOUT_AGGREGATIONS: Dict[ReadoutType, Type[pyg_nn.Aggregation]] = {
    "sum": pyg_nn.SumAggregation,
    "mean": pyg_nn.MeanAggregation,
}

RECURRENT_MODELS: Dict[NeckType, Type[nn.Module]] = {
    "gru": pygt_nn.recurrent.GConvGRU,
    "lstm": pygt_nn.recurrent.GConvLSTM,
}


class GraphAndGlobalFusion(nn.Module):
    """
    Fuse graph-level and global feature vectors into a single concatenated output of size (2 x graph_din).

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
        fused_emb = torch.cat([graph_emb, glob_emb], dim=-1)
        return fused_emb


class TemporalGraphAndGlobalFusion(nn.Module):
    def __init__(self, node_dim, backbone_dout: int, glob_din: int, cfg: NeckConfig):
        super().__init__()
        self.grnn = RECURRENT_MODELS[cfg.type](backbone_dout + node_dim, cfg.dhid, 1)
        self.fuse = GraphAndGlobalFusion(backbone_dout, glob_din, cfg)

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.grnn, pygt_nn.recurrent.GConvLSTM):
            h, c = self.grnn(
                torch.concat([z, x], dim=-1), edge_index, edge_weight, prev_h, prev_c
            )
        else:
            h = self.grnn(torch.concat([z, x], dim=-1), edge_index, edge_weight, prev_h)
            c = prev_c
        fused_emb = self.fuse(z, u, batch, batch_size)
        return fused_emb, h, c
