from typing import Optional

import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.typing import Adj, OptTensor

from soccerai.data.dataset import WorldCup2022Dataset
from soccerai.models.backbone import BackboneRegistry
from soccerai.models.backbones import GraphSAGEBackbone
from soccerai.models.head import GraphClassificationHead
from soccerai.models.neck import Neck, TNeck
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
        fused_emb = self.neck(u, z, batch, batch_size)
        return self.head(fused_emb)


class GraphSAGE(nn.Module):
    """
    Implements the GraphSAGE architecture as in the original
    paper *Inductive Representation Learning on Large Graphs* (Hamilton et al.,
    2017) — arXiv:1706.02216.
    """

    def __init__(
        self,
        node_feature_din: int,
        glob_feature_din: int,
        dmid: int,
        dropout_layer: float,
        dropout_head: float,
        num_layers: int,
        aggr_type: str,
        l2_norm: bool = True,
        dout: int = 1,
    ) -> None:
        super().__init__()
        self.backbone = GraphSAGEBackbone(
            din=node_feature_din,
            dmid=dmid,
            dout=dmid,
            num_layers=num_layers,
            aggr_type=aggr_type,
            l2_norm=l2_norm,
            dropout=dropout_layer,
        )
        self.mean_pool = pyg_nn.MeanAggregation()
        self.global_proj = nn.Linear(glob_feature_din, dmid)
        self.head = GraphClassificationHead(dmid * 2, dout, p_drop=dropout_head)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        u: torch.Tensor,
        batch: OptTensor = None,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        node_emb = self.backbone(x, edge_index)
        graph_emb = self.mean_pool(node_emb, batch)

        global_emb = F.relu(self.global_proj(u), inplace=True)
        fused = torch.cat([graph_emb, global_emb], dim=-1)

        return self.head(fused)


class TGNN(nn.Module):
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
        prev_h: OptTensor = None,
    ):
        z = self.backbone(x, edge_index, edge_weight, edge_attr, batch, batch_size)
        fused_emb, h = self.neck(
            x, u, z, edge_index, edge_weight, batch, batch_size, prev_h
        )

        return self.head(fused_emb), h


def create_model(cfg: Config, train_ds: WorldCup2022Dataset) -> nn.Module:
    backbone = BackboneRegistry.create(
        cfg.model.backbone.name, train_ds.num_node_features, cfg.model.backbone
    )
    neck = Neck(cfg.model.backbone.dout, train_ds.num_global_features, cfg.model.neck)
    head = GraphClassificationHead(cfg.model.backbone.dout * 2, cfg.model.head)

    match cfg.model.name:
        case "gnn":
            return GNN(backbone, neck, head)
        case "tgnn":
            return TGNN(
                backbone,
                TNeck(
                    train_ds.num_node_features,
                    cfg.model.backbone.dout,
                    cfg.model.neck,
                    neck,
                ),
                head,
            )
