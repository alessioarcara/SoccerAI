import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.typing import Adj, OptTensor


class GCNBackbone(nn.Module):
    def __init__(self, din: int, dmid: int):
        super(GCNBackbone, self).__init__()
        self.conv1 = pyg_nn.GCNConv(din, dmid)
        # self.norm1 = pyg_nn.LayerNorm(dmid)
        self.conv2 = pyg_nn.GCNConv(dmid, dmid)
        # self.norm2 = pyg_nn.LayerNorm(dmid)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
    ):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        return F.relu(self.conv2(x, edge_index, edge_weight))


# class GIN(nn.Module):
#     def __init__(self, din: int, dmid: int, dout: int):
#         super(GIN, self).__init__()

#         mlp1 = nn.Sequential(
#             nn.Linear(din, dmid),
#             nn.BatchNorm1d(dmid),
#             nn.ReLU(),
#             nn.Linear(dmid, dmid),
#             nn.ReLU(),
#         )
#         mlp2 = nn.Sequential(
#             nn.Linear(dmid, dmid),
#             nn.BatchNorm1d(dmid),
#             nn.ReLU(),
#             nn.Linear(dmid, dmid),
#             nn.ReLU(),
#         )
#         mlp3 = nn.Sequential(
#             nn.Linear(dmid, dmid),
#             nn.BatchNorm1d(dmid),
#             nn.ReLU(),
#             nn.Linear(dmid, dmid),
#             nn.ReLU(),
#         )

#         self.conv1 = pyg_nn.GINConv(mlp1)
#         self.conv2 = pyg_nn.GINConv(mlp2)
#         self.conv3 = pyg_nn.GINConv(mlp3)

#         self.sum_pool = SumAggregation()

#         self.lin1 = nn.Linear(dmid * 3, dmid * 3)
#         self.lin2 = nn.Linear(dmid * 3, dout)

#     def forward(
#         self,
#         x: torch.Tensor,
#         edge_index: Adj,
#         edge_weight: OptTensor = None,
#         edge_attr: OptTensor = None,
#         batch: OptTensor = None,
#         batch_size: Optional[int] = None,
#     ):
#         h1 = self.conv1(x, edge_index)
#         h2 = self.conv2(h1, edge_index)
#         h3 = self.conv3(h2, edge_index)

#         g1 = self.sum_pool(h1, batch, dim_size=batch_size)
#         g2 = self.sum_pool(h2, batch, dim_size=batch_size)
#         g3 = self.sum_pool(h3, batch, dim_size=batch_size)

#         h = torch.cat([g1, g2, g3], dim=1)
#         h = F.relu(self.lin1(h))
#         h = F.dropout(h, p=0.5, training=self.training)
#         return self.lin2(h)
