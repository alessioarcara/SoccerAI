from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.typing import OptTensor


class GraphClassificationHead(nn.Module):
    def __init__(self, din: int, dout: int, p_drop: float = 0.5):
        super(GraphClassificationHead, self).__init__()
        self.p_drop = p_drop
        self.mean_pool = pyg_nn.MeanAggregation()
        self.lin1 = pyg_nn.Linear(din, din)
        self.lin2 = pyg_nn.Linear(din, dout)

    def forward(
        self, x: torch.Tensor, batch: OptTensor = None, batch_size: Optional[int] = None
    ):
        x = self.mean_pool(x, batch, dim_size=batch_size)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.p_drop, training=self.training)
        return self.lin2(x)
