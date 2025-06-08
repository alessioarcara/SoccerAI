import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.typing import Adj, OptTensor


class GCNBackbone(nn.Module):
    def __init__(self, din: int, dmid: int):
        super(GCNBackbone, self).__init__()
        self.conv1 = pyg_nn.GCNConv(din, dmid)
        self.norm1 = pyg_nn.LayerNorm(dmid)
        self.conv2 = pyg_nn.GCNConv(dmid, dmid)
        self.norm2 = pyg_nn.LayerNorm(dmid)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
    ):
        x = F.relu(self.norm1(self.conv1(x, edge_index, edge_weight)))
        return F.relu(self.norm2(self.conv2(x, edge_index, edge_weight)))
