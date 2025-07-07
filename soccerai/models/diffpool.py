from math import ceil
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import (
    to_dense_adj,
    to_dense_batch,
)

from soccerai.models.necks import RNN_CELLS
from soccerai.training.trainer_config import DiffPoolConfig, ModelConfig


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


class HierarchicalGNN(nn.Module):
    def __init__(self, din: int, glob_din: int, cfg: ModelConfig, head: nn.Module):
        super().__init__()

        assert isinstance(cfg.backbone, DiffPoolConfig)
        pooling_ratio = cfg.backbone.pooling_ratio
        base_dhid = cfg.backbone.dhid
        factor = cfg.backbone.dhid_multiplier
        self.readout = cfg.neck.readout

        # Backbone
        dhid_levels = [max(1, int(base_dhid * (factor**i))) for i in range(3)]
        dhid1, dhid2, dhid3 = dhid_levels

        num_nodes = ceil(pooling_ratio * 22)
        self.gnn1_pool = DenseSageGNN(din, dhid1, num_nodes)
        self.gnn1_embed = DenseSageGNN(din, dhid1, dhid1, lin=False)

        num_nodes = ceil(pooling_ratio * num_nodes)
        self.gnn2_pool = DenseSageGNN(3 * dhid1, dhid2, num_nodes)
        self.gnn2_embed = DenseSageGNN(3 * dhid1, dhid2, dhid2, lin=False)

        self.gnn3_embed = DenseSageGNN(3 * dhid2, dhid3, dhid3, lin=False)

        # Neck
        self.global_proj = pyg_nn.Linear(glob_din, cfg.neck.glob_dout)

        self.rnn = RNN_CELLS[cfg.neck.rnn_type](
            input_size=cfg.neck.rnn_din, hidden_size=cfg.neck.rnn_dout
        )

        # Head
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

        x, adj, _, _ = pyg_nn.dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, _, _ = pyg_nn.dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        if self.readout == "mean":
            graph_emb = x.mean(dim=1)
        elif self.readout == "sum":
            graph_emb = x.sum(dim=1)
        else:  # "max"
            graph_emb, _ = x.max(dim=1)

        glob_emb = F.relu(self.global_proj(u), inplace=True)
        fused = torch.cat([graph_emb, glob_emb], dim=-1)

        if isinstance(self.rnn, nn.LSTMCell):
            state: Optional[Tuple[OptTensor, OptTensor]] = (
                prev_h,
                prev_c,
            )
            h, c = self.rnn(fused, state if prev_h is not None else None)
        else:  # GRUCell
            h = self.rnn(fused, prev_h)
            c = None

        logits = self.head(h)
        return logits, h, c
