import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from soccerai.training.trainer_config import HeadConfig


class GraphClassificationHead(nn.Module):
    def __init__(self, din: int, cfg: HeadConfig):
        super(GraphClassificationHead, self).__init__()
        self.lin1 = pyg_nn.Linear(din, din)
        self.drop = nn.Dropout(p=cfg.drop)
        self.lin2 = pyg_nn.Linear(din, cfg.dout)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.lin1(x), inplace=True)
        return self.lin2(self.drop(x))
