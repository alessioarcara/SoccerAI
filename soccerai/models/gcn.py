from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.typing import Adj, OptTensor

from soccerai.training.trainer_config import TrainerConfig


class GCN(torch.nn.Module):
    def __init__(self, cfg: TrainerConfig, din: int, dglob: int, dout: int):
        super(GCN, self).__init__()
        self.cfg = cfg
        dmid = cfg.dim

        self.conv1 = pyg_nn.GCNConv(din, dmid)
        self.conv2 = pyg_nn.GCNConv(dmid, dmid)
        self.mean_pool = pyg_nn.MeanAggregation()

        head_din = dmid + dglob if cfg.use_global_features else dmid
        self.head = nn.Sequential(
            nn.Linear(head_din, dmid),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(dmid, dout),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        u: torch.Tensor,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
    ):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)

        graph_emb = self.mean_pool(x, batch, dim_size=batch_size)

        if self.cfg.use_global_features:
            h = torch.cat([graph_emb, u], dim=-1)
        else:
            h = graph_emb

        return self.head(h)
