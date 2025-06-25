from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric_temporal.nn as pygt_nn
from torch_geometric.typing import Adj, OptTensor

from soccerai.data.dataset import WorldCup2022Dataset
from soccerai.models.backbones import GATv2Backbone, GCNBackbone
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
                node_feature_din=train_ds.num_node_features,
                glob_feature_din=train_ds.num_global_features,
                backbone=cfg.model.backbone,
                n_layers=cfg.model.num_layers,
                p_drop=cfg.model.dropout_head,
                use_edge_attr=cfg.model.use_edge_attr,
            )
        case "gatv2":
            return GATv2(
                node_feature_din=train_ds.num_node_features,
                glob_feature_din=train_ds.num_global_features,
                dmid=cfg.model.dmid,
                use_edge_attr=cfg.model.use_edge_attr,
                dropout_layer=cfg.model.dropout_layer,
                dropout_head=cfg.model.dropout_head,
                num_layers=cfg.model.num_layers,
            )
        case _:
            raise ValueError("Invalid model name")


class GCN(torch.nn.Module):
    def __init__(
        self, node_feature_din: int, glob_feature_din: int, dmid: int, dout: int = 1
    ):
        super(GCN, self).__init__()
        self.backbone = GCNBackbone(node_feature_din, dmid, dout)
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


class GCRNN(nn.Module):
    def __init__(
        self,
        node_feature_din: int,
        glob_feature_din: int,
        backbone: str,
        n_layers: int,
        p_drop: float,
        use_edge_attr: bool,
        dout: int = 1,
    ):
        super(GCRNN, self).__init__()
        self.backbone: nn.Module
        match backbone:
            case "gcn":
                self.backbone = GCNBackbone(node_feature_din, 256, 128)
            case "gatv2":
                self.backbone = GATv2Backbone(
                    node_feature_din, 256, 128, use_edge_attr=use_edge_attr
                )

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


class GATv2(nn.Module):
    def __init__(
        self,
        node_feature_din,
        glob_feature_din,
        dmid,
        use_edge_attr,
        dropout_layer=0.6,
        dropout_head=0.5,
        num_layers=3,
        dout: int = 1,
    ):
        super().__init__()
        self.backbone = GATv2Backbone(
            node_feature_din,
            dmid=dmid,
            dout=dmid,
            use_edge_attr=use_edge_attr,
            num_layers=num_layers,
            dropout=dropout_layer,
        )
        self.mean_pool = pyg_nn.MeanAggregation()
        self.global_proj = nn.Linear(glob_feature_din, dmid)
        self.head = GraphClassificationHead(dmid * 2, dout=1, p_drop=dropout_head)

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
        node_emb = self.backbone(x, edge_index)  # [N, dmid]
        graph_emb = self.mean_pool(node_emb, batch)  # [B, dmid]
        global_emb = F.relu(self.global_proj(u))
        return self.head(torch.cat([graph_emb, global_emb], dim=-1))
