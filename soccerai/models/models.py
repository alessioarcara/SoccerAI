from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.typing import Adj, OptTensor

from soccerai.data.dataset import WorldCup2022Dataset
from soccerai.models.backbones import BackboneRegistry
from soccerai.models.heads import GraphClassificationHead
from soccerai.models.necks import GraphGlobalFusion, TemporalFusion
from soccerai.training.trainer_config import Config


class GNN(nn.Module):
    def __init__(self, backbone: nn.Module, neck: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        u: torch.Tensor,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
    ):
        z = self.backbone(x, edge_index, edge_weight, edge_attr, batch, batch_size)
        fused_emb = self.neck(z, u, batch, batch_size)
        return self.head(fused_emb)


class TemporalGNN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        head: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        u: torch.Tensor,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
        prev_h: OptTensor = None,
        prev_c: OptTensor = None,
    ):
        z = self.backbone(
            x,
            edge_index,
            edge_weight,
            edge_attr,
            batch,
            batch_size,
            prev_h,
        )
        fused_emb, h, c = self.neck(
            z, u, x, edge_index, edge_weight, batch, batch_size, prev_h, prev_c
        )
        return self.head(fused_emb), h, c


def build_model(cfg: Config, train_ds: WorldCup2022Dataset) -> nn.Module:
    backbone = BackboneRegistry.create(
        cfg.model.backbone.type, train_ds.num_node_features, cfg.model.backbone
    )
    head = GraphClassificationHead(cfg.model.head)

    if cfg.model.use_temporal:
        return TemporalGNN(
            backbone,
            TemporalFusion(
                cfg.model.backbone.dout,
                train_ds.num_node_features,
                train_ds.num_global_features,
                cfg.model.neck,
            ),
            head,
        )
    else:
        return GNN(
            backbone,
            GraphGlobalFusion(train_ds.num_global_features, cfg.model.neck),
            head,
        )
