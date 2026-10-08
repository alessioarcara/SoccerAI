from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import (
    to_dense_adj,
    to_dense_batch,
)

from soccerai.models.necks import TemporalFusion
from soccerai.models.typings import ReadoutType


class DenseSageGNN(torch.nn.Module):
    """
    DenseSageGNN: a GNN operating on dense adjacency matrices.

    Based on the PyTorch Geometric DiffPool example
    (https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_diff_pool.py),
    """

    def __init__(
        self, din: int, dhid: int, dout: int, normalize: bool = True, lin: bool = True
    ):
        super().__init__()

        self.conv1 = pyg_nn.DenseSAGEConv(din, dhid, normalize)
        self.conv2 = pyg_nn.DenseSAGEConv(dhid, dhid, normalize)
        self.conv3 = pyg_nn.DenseSAGEConv(dhid, dout, normalize)

        self.bn1 = pyg_nn.BatchNorm(dhid)
        self.bn2 = pyg_nn.BatchNorm(dhid)
        self.bn3 = pyg_nn.BatchNorm(dout)

        self.lin = None
        if lin:
            self.lin = pyg_nn.Linear(2 * dhid + dout, dout)

    def bn(self, i: int, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)  # (B*N, Node_dim)
        x = getattr(self, f"bn{i}")(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor, mask: OptTensor = None
    ) -> torch.Tensor:
        x0 = x
        x1 = F.relu(self.bn(1, self.conv1(x0, adj, mask)), inplace=True)
        x2 = F.relu(self.bn(2, self.conv2(x1, adj, mask)), inplace=True)
        x3 = F.relu(self.bn(3, self.conv3(x2, adj, mask)), inplace=True)

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = self.lin(x).relu()

        return x


EPS = 1e-15


def diffpool_aux_losses(
    adj: torch.Tensor, s: torch.Tensor, mask: OptTensor = None
) -> torch.Tensor:
    """
    Per-graph DiffPool regularisers: link-prediction loss plus assignment
    entropy, shape (B,). Same terms as `dense_diff_pool`, which only returns
    their batch mean, so that the trainer can drop padded frames.
    """
    s = torch.softmax(s, dim=-1)
    if mask is not None:
        s = s * mask.unsqueeze(-1).to(s.dtype)
    n = adj.size(-1)
    link = (adj - s @ s.transpose(1, 2)).flatten(1).norm(p=2, dim=1) / (n * n)
    ent = (-s * torch.log(s + EPS)).sum(dim=-1)  # (B, N)
    if mask is None:
        ent = ent.mean(dim=-1)
    else:
        m = mask.to(ent.dtype)
        ent = (ent * m).sum(dim=-1) / m.sum(dim=-1).clamp(min=1.0)
    return link + ent


class DiffPoolBackbone(nn.Module):
    """
    Two DiffPool levels over a dense SAGE stack, then a readout of the
    clusters: returns one embedding per graph, shape (B, out_dim).

    The link-prediction and assignment-entropy regularisers of every graph
    (shape (B,)) are left in `aux_loss` after each forward pass.
    """

    def __init__(
        self,
        din: int,
        dhid: int,
        pooling_ratio: float = 0.25,
        dhid_multiplier: int = 1,
        readout: ReadoutType = "mean",
        num_nodes: int = 22,
    ):
        super().__init__()
        dhid1, dhid2, dhid3 = (
            max(1, int(dhid * (dhid_multiplier**i))) for i in range(3)
        )
        self.readout = readout

        n_clusters = ceil(pooling_ratio * num_nodes)
        self.gnn1_pool = DenseSageGNN(din, dhid1, n_clusters)
        self.gnn1_embed = DenseSageGNN(din, dhid1, dhid1, lin=False)

        n_clusters = ceil(pooling_ratio * n_clusters)
        self.gnn2_pool = DenseSageGNN(3 * dhid1, dhid2, n_clusters)
        self.gnn2_embed = DenseSageGNN(3 * dhid1, dhid2, dhid2, lin=False)

        self.gnn3_embed = DenseSageGNN(3 * dhid2, dhid3, dhid3, lin=False)
        # the dense SAGE stack concatenates its three layers
        self.out_dim = 3 * dhid3
        self.aux_loss: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch=batch, edge_attr=edge_attr)

        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        # one value per graph (B,); the trainer averages them over the real
        # (non-padded) graphs and adds them to the classification loss
        aux_loss = diffpool_aux_losses(adj, s, mask)
        x, adj, _, _ = pyg_nn.dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        aux_loss = aux_loss + diffpool_aux_losses(adj, s)
        x, adj, _, _ = pyg_nn.dense_diff_pool(x, adj, s)
        self.aux_loss = aux_loss

        x = self.gnn3_embed(x, adj)

        if self.readout == "mean":
            return x.mean(dim=1)
        if self.readout == "sum":
            return x.sum(dim=1)
        return x.max(dim=1).values


class HierarchicalGNN(nn.Module):
    """
    Temporal model over a backbone that pools the graph itself (DiffPool):
    same interface and neck/head as `TemporalGNN`.
    """

    def __init__(
        self, backbone: DiffPoolBackbone, neck: TemporalFusion, head: nn.Module
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    @property
    def aux_loss(self) -> torch.Tensor | None:
        return self.backbone.aux_loss

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        u: torch.Tensor,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: int | None = None,
        prev_h: OptTensor = None,
        prev_c: OptTensor = None,
    ):
        graph_emb = self.backbone(
            x, edge_index, edge_weight, edge_attr, batch, batch_size
        )
        fused, h, c = self.neck.forward_pooled(graph_emb, u, prev_h, prev_c)
        return self.head(fused), h, c
