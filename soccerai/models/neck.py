from typing import Dict, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric_temporal.nn as pygt_nn
from torch_geometric.typing import Adj, OptTensor

from soccerai.training.trainer_config import NeckConfig

READOUT: Dict[str, Type[nn.Module]] = {
    "sum": pyg_nn.SumAggregation,
    "mean": pyg_nn.MeanAggregation,
}


class Neck(nn.Module):
    def __init__(self, embed_din: int, glob_din: int, cfg: NeckConfig):
        super().__init__()
        self.readout = READOUT[cfg.readout]()
        self.global_proj = pyg_nn.Linear(glob_din, embed_din)

    def forward(
        self,
        u: torch.Tensor,
        z: torch.Tensor,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
    ):
        graph_emb = self.readout(z, index=batch, dim_size=batch_size)
        global_emb = F.relu(self.global_proj(u))
        fused_emb = torch.cat([graph_emb, global_emb], dim=-1)
        return fused_emb


class TNeck(nn.Module):
    def __init__(self, node_din, backbone_dout: int, cfg: NeckConfig, neck: Neck):
        super().__init__()
        self.grnn = pygt_nn.recurrent.GConvGRU(
            backbone_dout + node_din, cfg.dhid, cfg.k
        )
        self.neck = neck

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        z: torch.Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
        prev_h: OptTensor = None,
    ):
        h = self.grnn(torch.concat([z, x], dim=-1), edge_index, edge_weight, prev_h)

        return self.neck.forward(u, z, batch, batch_size), h
