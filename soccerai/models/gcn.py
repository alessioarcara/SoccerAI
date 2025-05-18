import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


class GCN(torch.nn.Module):
    def __init__(self, din: int, dmid: int, dout: int):
        super(GCN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(din, dmid)
        self.conv2 = pyg_nn.GCNConv(dmid, dmid)
        self.lin = pyg_nn.Linear(dmid, dout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = pyg_nn.global_add_pool(x, batch)

        x = self.lin(x)

        return x
