import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


class GIN(nn.Module):
    def __init__(self, din: int, dmid: int, dout: int):
        super(GIN, self).__init__()
        mlp1 = nn.Sequential(
            nn.Linear(din, dmid),
            nn.BatchNorm1d(dmid),
            nn.ReLU(),
            nn.Linear(dmid, dmid),
            nn.ReLU(),
        )
        mlp2 = nn.Sequential(
            nn.Linear(dmid, dmid),
            nn.BatchNorm1d(dmid),
            nn.ReLU(),
            nn.Linear(dmid, dmid),
            nn.ReLU(),
        )
        mlp3 = nn.Sequential(
            nn.Linear(dmid, dmid),
            nn.BatchNorm1d(dmid),
            nn.ReLU(),
            nn.Linear(dmid, dmid),
            nn.ReLU(),
        )

        self.conv1 = pyg_nn.GINConv(mlp1)
        self.conv2 = pyg_nn.GINConv(mlp2)
        self.conv3 = pyg_nn.GINConv(mlp3)

        self.lin1 = nn.Linear(dmid * 3, dmid * 3)
        self.lin2 = nn.Linear(dmid * 3, dout)

    def forward(self, data):
        x, edge_index, batch, batch_size = (
            data.x,
            data.edge_index,
            data.batch,
            data.batch_size,
        )

        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        g1 = pyg_nn.global_add_pool(h1, batch, size=batch_size)
        g2 = pyg_nn.global_add_pool(h2, batch, size=batch_size)
        g3 = pyg_nn.global_add_pool(h3, batch, size=batch_size)

        h = torch.cat([g1, g2, g3], dim=1)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return h
