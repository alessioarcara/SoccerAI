from typing import Optional

import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.typing import Adj, OptTensor


class GCN(torch.nn.Module):
    def __init__(self, din: int, dmid: int, dout: int):
        super(GCN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(din, dmid)
        self.conv2 = pyg_nn.GCNConv(dmid, dmid)
        self.mean_pool = pyg_nn.MeanAggregation()
        self.lin1 = pyg_nn.Linear(dmid, dmid)
        self.lin2 = pyg_nn.Linear(dmid, dout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
    ):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)

        x = self.mean_pool(x, batch, dim_size=batch_size)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin2(x)
