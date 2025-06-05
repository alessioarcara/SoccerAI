import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric_temporal.nn.recurrent import TGCN2


class RecurrentGCN(torch.nn.Module):
    def __init__(self, din, dmid, dout, batch_size):
        super(RecurrentGCN, self).__init__()
        self.recurrent = TGCN2(
            din, dmid, batch_size
        )  # The network assumes a static graph structure shared across the entire batch, edge_index must have shape [2, E].
        self.linear = torch.nn.Linear(dmid, dout)
        self.pool = pyg_nn.MeanAggregation()

    def forward(self, x, edge_index, prev_hidden_state):
        if edge_index.shape[0] != 2:
            edge_index = edge_index[0]
        h = self.recurrent(x, edge_index, None, prev_hidden_state)
        y = self.pool(h)
        y = F.relu(y)
        y = self.linear(y)
        return y, h
