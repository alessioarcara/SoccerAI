from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.typing import Adj, OptTensor

from soccerai.data.dataset import WorldCup2022Dataset
from soccerai.models.backbones import GCNBackbone
from soccerai.models.heads import GraphClassificationHead
from soccerai.training.trainer_config import Config


def create_model(cfg: Config, train_ds: WorldCup2022Dataset) -> nn.Module:
    match cfg.model.model_name:
        case "gcn":
            return GCN(train_ds.num_node_features, cfg.model.dmid)
        case "rgcn":
            return RGCN(train_ds.num_node_features, cfg.model.dmid, cfg.model.dhid)
        case _:
            raise ValueError("Invalid model name")


class GCN(torch.nn.Module):
    def __init__(self, din: int, dmid: int, dout: int = 1):
        super(GCN, self).__init__()
        self.backbone = GCNBackbone(din, dmid)
        self.head = GraphClassificationHead(dmid, dout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
    ):
        x = self.backbone(x, edge_index, edge_weight, edge_attr)
        return self.head(x, batch, batch_size)


class RGCN(torch.nn.Module):
    def __init__(self, din: int, dmid: int, dhid: int, dout: int = 1):
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
