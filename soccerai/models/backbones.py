import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.typing import Adj, OptTensor


class GCNBackbone(nn.Module):
    def __init__(self, din: int, dmid: int, dout: int):
        super(GCNBackbone, self).__init__()
        self.conv1 = pyg_nn.GCNConv(din, dmid)
        self.conv2 = pyg_nn.GCNConv(dmid, dout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        prev_h: OptTensor = None,
    ):
        f = F.relu(self.conv1(x, edge_index, edge_weight))

        if prev_h is None:
            prev_h = torch.zeros_like(f, device=f.device)

        return F.relu(self.conv2(f + prev_h, edge_index, edge_weight))


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


class GraphSAGEBackbone(nn.Module):
    def __init__(
        self,
        din: int,
        dmid: int,
        dout: int,
        num_layers: int = 2,
        aggr_type: str = "mean",
        l2_norm: bool = True,
        dropout: float = 0.0,
    ):
        super(GraphSAGEBackbone, self).__init__()
        aggr_type = aggr_type.lower()
        if aggr_type not in {"mean", "pool", "lstm"}:
            raise ValueError("aggr_type must be 'mean', 'pool', or 'lstm'.")
        self.aggr_type = aggr_type
        self.dropout = dropout

        self.convs = nn.ModuleList()

        for i in range(num_layers):
            in_dim = din if i == 0 else dmid
            out_dim = dout if i == num_layers - 1 else dmid

            if aggr_type == "mean":
                conv = pyg_nn.SAGEConv(in_dim, out_dim, aggr="mean", normalize=l2_norm)

            elif aggr_type == "pool":
                conv = pyg_nn.SAGEConv(
                    in_dim, out_dim, aggr="max", project=True, normalize=l2_norm
                )

            else:  # lstm
                conv = pyg_nn.SAGEConv(in_dim, out_dim, aggr="lstm", normalize=l2_norm)

            self.convs.append(conv)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: OptTensor = None,
        prev_h: OptTensor = None,
    ):
        x = F.relu(self.convs[0](x, edge_index), inplace=True)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        if prev_h is None:
            prev_h = torch.zeros_like(x)

        for conv in self.convs[1:]:
            x = F.relu(conv(x + prev_h, edge_index), inplace=True)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x
