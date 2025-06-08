from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric_temporal.nn as pygt_nn
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
        case "gcrnn":
            return GCRNN(train_ds.num_node_features)
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


class GCRNN(nn.Module):
    # TODO: aggiungere iperparametro per scegliere tipologia di skip connection
    def __init__(self, din: int, dout: int = 1):
        super(GCRNN, self).__init__()
        self.gcn1 = pyg_nn.GCNConv(din, 256)
        self.gcn2 = pyg_nn.GCNConv(256, 128)
        self.gcrn = pygt_nn.recurrent.GConvGRU(128 + din, 256, 1)
        self.head = GraphClassificationHead(128, dout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        prev_h: OptTensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_index = edge_index.to(torch.long)
        f = F.relu(self.gcn1(x, edge_index, edge_weight))

        if prev_h is None:
            prev_h = torch.zeros_like(f, device=f.device)

        z = F.relu(self.gcn2(f + prev_h, edge_index, edge_weight))
        h = self.gcrn(torch.concat([z, x], dim=-1), edge_index, edge_weight, prev_h)
        return self.head(z), h
