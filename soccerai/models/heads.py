import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


class GraphClassificationHead(nn.Module):
    def __init__(self, din: int, dout: int, p_drop: float):
        super(GraphClassificationHead, self).__init__()
        self.lin1 = pyg_nn.Linear(din, din // 2)
        self.drop = nn.Dropout(p=p_drop)
        self.lin2 = pyg_nn.Linear(din // 2, din // 4)
        self.lin3 = pyg_nn.Linear(din // 4, dout)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.lin1(x), inplace=True)
        x = F.relu(self.lin2(self.drop(x)), inplace=True)
        return self.lin3(self.drop(x))
