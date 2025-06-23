from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric_temporal.nn as pygt_nn
from torch_geometric.typing import Adj, OptTensor

from soccerai.data.dataset import WorldCup2022Dataset
from soccerai.models.backbones import GCNBackbone, GraphSAGEBackbone
from soccerai.models.heads import GraphClassificationHead
from soccerai.training.trainer_config import Config


def create_model(cfg: Config, train_ds: WorldCup2022Dataset) -> nn.Module:
    match cfg.model.model_name:
        case "gcn":
            return GCN(
                train_ds.num_node_features, train_ds.num_global_features, cfg.model.dmid
            )
        case "gcrnn":
            return GCRNN(
                train_ds.num_node_features,
                train_ds.num_global_features,
                cfg.model.backbone,
                cfg.model.num_layers,
                cfg.model.dropout_head,
            )

        case "graphsage":
            return GraphSAGE(
                node_feature_din=train_ds.num_node_features,
                glob_feature_din=train_ds.num_global_features,
                dmid=cfg.model.dmid,
                num_layers=cfg.model.num_layers,
                dropout_layer=cfg.model.dropout_layer,
                dropout_head=cfg.model.dropout_head,
                aggr_type=cfg.model.aggr_type,
            )
        case _:
            raise ValueError("Invalid model name")


class GCN(torch.nn.Module):
    def __init__(
        self, node_feature_din: int, glob_feature_din: int, dmid: int, dout: int = 1
    ):
        super(GCN, self).__init__()
        self.backbone = GCNBackbone(node_feature_din, dmid, dmid)
        self.global_proj = nn.Linear(glob_feature_din, dmid)
        self.mean_pool = pyg_nn.MeanAggregation()
        self.head = GraphClassificationHead(dmid * 2, dout)

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
        node_emb = self.backbone(x, edge_index, edge_weight, edge_attr)
        graph_emb = self.mean_pool(node_emb, batch, dim_size=batch_size)
        global_emb = F.relu(self.global_proj(u), inplace=True)

        fused_emb = torch.cat([graph_emb, global_emb], dim=-1)
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


class GCRNN(nn.Module):
    def __init__(
        self,
        node_feature_din: int,
        glob_feature_din: int,
        backbone: str,
        n_layers: int,
        p_drop: float,
        dout: int = 1,
    ):
        super(GCRNN, self).__init__()
        self.backbone: nn.Module
        match backbone:
            case "gcn":
                self.backbone = GCNBackbone(node_feature_din, 256, 128)
            case "graphsage":
                self.backbone = GraphSAGEBackbone(node_feature_din, 256, 128)

        self.global_proj = nn.Linear(glob_feature_din, 128)

        self.gcrn = pygt_nn.recurrent.GConvGRU(128 + node_feature_din, 256, 1)

        self.mean_pool = pyg_nn.MeanAggregation()
        self.head = GraphClassificationHead(256, dout, p_drop)

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.backbone(x, edge_index, edge_weight, prev_h=prev_h)

        h = self.gcrn(torch.concat([z, x], dim=-1), edge_index, edge_weight, prev_h)

        graph_emb = self.mean_pool(z, index=batch, dim_size=batch_size)
        global_emb = F.relu(self.global_proj(u))

        fused_emb = torch.cat([graph_emb, global_emb], dim=-1)
        out = self.head(fused_emb)

        return out, h
