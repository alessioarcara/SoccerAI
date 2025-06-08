from typing import Tuple

import torch
import torch.nn as nn
from torch_geometric.typing import Adj, OptTensor

from soccerai.models.backbones import GCNBackbone
from soccerai.models.heads import GraphClassificationHead


class RGCN(torch.nn.Module):
    def __init__(self, din: int, dmid: int, dout: int, dhid: int):
        super(RGCN, self).__init__()
        self.spatial = GCNBackbone(din, dmid)
        self.temporal = nn.GRUCell(dmid, dhid, bias=False)
        self.head = GraphClassificationHead(dhid, dout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        prev_h: OptTensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.spatial(x, edge_index, edge_weight, edge_attr)
        h = self.temporal(x, prev_h)
        return self.head(h), h
