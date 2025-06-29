from math import ceil
from typing import Optional

import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import (
    to_dense_adj,
    to_dense_batch,
)


class DenseGNN(torch.nn.Module):
    """
    GNN which works on the full dense adjacency matrix
    taken from
    https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_diff_pool.py

    but modified for
    GraphConv because it is a weighted one
    """

    def __init__(
        self, din: int, dhid: int, dout: int, normalize: bool = False, lin: bool = True
    ):
        super().__init__()

        self.conv1 = pyg_nn.DenseSAGEConv(din, dhid, normalize=True)
        self.bn1 = torch.nn.BatchNorm1d(dhid)
        self.conv2 = pyg_nn.DenseSAGEConv(dhid, dhid, normalize=True)
        self.bn2 = torch.nn.BatchNorm1d(dhid)
        self.conv3 = pyg_nn.DenseSAGEConv(dhid, dout, normalize=True)
        self.bn3 = torch.nn.BatchNorm1d(dout)

        self.lin = None
        if lin is True:
            self.lin = torch.nn.Linear(2 * dhid + dout, dout)

    def bn(self, i: int, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, f"bn{i}")(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor, mask: OptTensor = None
    ) -> torch.Tensor:
        x0 = x
        x1 = self.bn(1, self.conv1(x0, adj, mask).relu())
        x2 = self.bn(2, self.conv2(x1, adj, mask).relu())
        x3 = self.bn(3, self.conv3(x2, adj, mask).relu())

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = self.lin(x).relu()

        return x


class HierarchicalGNN(nn.Module):
    def __init__(self, din: int, head: nn.Module):
        super().__init__()

        num_nodes = ceil(0.25 * 22)
        self.gnn1_pool = DenseGNN(din, 64, num_nodes)
        self.gnn1_embed = DenseGNN(din, 64, 64, lin=False)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = DenseGNN(3 * 64, 64, num_nodes)
        self.gnn2_embed = DenseGNN(3 * 64, 64, 64, lin=False)

        self.gnn3_embed = DenseGNN(3 * 64, 64, 64, lin=False)

        self.rnn = nn.LSTMCell(3 * 64, 3 * 64)
        self.head = head

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        u: torch.Tensor,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
        prev_h: OptTensor = None,
        prev_c: OptTensor = None,
    ):
        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch=batch, edge_attr=edge_attr)

        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = pyg_nn.dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = pyg_nn.dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        state = (prev_h, prev_c)
        if prev_h is None or prev_c is None:
            state = None
        h, c = self.rnn(x, state)
        return self.head(h), h, c
