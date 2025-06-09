import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


class GraphClassificationHead(nn.Module):
    def __init__(self, din: int, dout: int, p_drop: float = 0.5):
        super(GraphClassificationHead, self).__init__()
        self.lin1 = pyg_nn.Linear(din, din)
        self.drop = nn.Dropout(p=p_drop)
        self.lin2 = pyg_nn.Linear(din, dout)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.lin1(x), inplace=True)
        return self.lin2(self.drop(x))
