import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric_temporal.nn.recurrent import TGCN2


class RecurrentGCN(torch.nn.Module):
    def __init__(self, din, dmid, dout, batch_size):
        super(RecurrentGCN, self).__init__()
        self.recurrent = TGCN2(
            din, dmid, batch_size, add_self_loops=False
        )  # The network assumes a static graph structure shared across the entire batch, edge_index must have shape [2, E].
        # self.pool = pyg_nn.MeanAggregation()
        self.pool = pyg_nn.GlobalAttention(nn.Sequential(nn.Linear(dmid, 1)))
        self.lin1 = pyg_nn.Linear(dmid, dmid)
        self.lin2 = pyg_nn.Linear(dmid, dout)

    def forward(self, x, edge_index, edge_weights, prev_hidden_state):
        # if edge_index.shape[0] != 2:
        #    edge_index = edge_index[0]
        edge_index = edge_index.reshape(2, -1)
        edge_weights = edge_weights.reshape(-1)
        # x = x.reshape(-1, x.shape[-1])
        h = self.recurrent(x, edge_index, edge_weights, prev_hidden_state)
        y = self.pool(h)
        y = F.relu(self.lin1(y))
        y = self.lin2(y)
        return y, h
