import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.typing import Adj, OptTensor


class GCNBackbone(nn.Module):
    def __init__(self, din: int, dmid: int):
        super(GCNBackbone, self).__init__()
        self.conv1 = pyg_nn.GCNConv(din, dmid)
        self.conv2 = pyg_nn.GCNConv(dmid, dmid)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
    ):
        x = F.relu(self.conv1(x, edge_index, edge_weight), inplace=True)
        return F.relu(self.conv2(x, edge_index, edge_weight), inplace=True)


def build_mlp(din: int, dmid: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(din, dmid), nn.BatchNorm1d(dmid), nn.ReLU(), nn.Linear(dmid, dmid)
    )


# edge dim?
class GINEBackbone(nn.Module):
    def __init__(self, din: int, dmid: int, n_layers: int = 5):
        super().__init__()

        self.convs = nn.ModuleList([pyg_nn.GINEConv(nn=build_mlp(din, dmid))])

        for _ in range(n_layers - 1):
            self.convs.append(pyg_nn.GINEConv(nn=build_mlp(dmid, dmid)))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
    ):
        pass

    #     outputs = []

    #     for i, conv in enumerate(self.convs[:-1]):
    #         x = conv(x, edge_index, edge_attr)
    #         outputs.append(x)

    #     return x
    # return torch.cat(outputs, dim=-1)

    # return self.conv(x, edge_index, edge_attr)

    # def forward(
    #     self,
    #     x: torch.Tensor,
    #     edge_index: Adj,
    #     edge_weight: OptTensor = None,
    #     edge_attr: OptTensor = None,
    #     batch: OptTensor = None,
    #     batch_size: Optional[int] = None,
    # ):
    #     g1 = self.sum_pool(h1, batch, dim_size=batch_size)
    #     g2 = self.sum_pool(h2, batch, dim_size=batch_size)
    #     g3 = self.sum_pool(h3, batch, dim_size=batch_size)

    #     h = torch.cat([g1, g2, g3], dim=1)
    #     h = F.relu(self.lin1(h))
    #     h = F.dropout(h, p=0.5, training=self.training)
    #     return self.lin2(h)
