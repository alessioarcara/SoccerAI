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
            return GCN(
                train_ds.num_node_features, train_ds.num_global_features, cfg.model.dmid
            )
        case "gcrnn":
            return GCRNN(train_ds.num_node_features, train_ds.num_global_features)
        case _:
            raise ValueError("Invalid model name")


class GCN(torch.nn.Module):
    def __init__(
        self, node_feature_din: int, glob_feature_din: int, dmid: int, dout: int = 1
    ):
        super(GCN, self).__init__()
        self.backbone = GCNBackbone(node_feature_din, dmid)
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
    def __init__(self, node_feature_din: int, glob_feature_din: int, dout: int = 1):
        super(GCRNN, self).__init__()
        self.gcn1 = pyg_nn.GCNConv(node_feature_din, 256)
        self.gcn2 = pyg_nn.GCNConv(256, 128)

        self.global_proj = nn.Linear(glob_feature_din, 128)

        self.gcrn = pygt_nn.recurrent.GConvGRU(128 + node_feature_din, 256, 1)

        self.mean_pool = pyg_nn.MeanAggregation()
        self.head = GraphClassificationHead(256, dout)

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
        f = F.relu(self.gcn1(x, edge_index, edge_weight))

        if prev_h is None:
            prev_h = torch.zeros_like(f, device=f.device)

        z = F.relu(self.gcn2(f + prev_h, edge_index, edge_weight))

        h = self.gcrn(torch.concat([z, x], dim=-1), edge_index, edge_weight, prev_h)

        graph_emb = self.mean_pool(z, index=batch, dim_size=batch_size)
        global_emb = F.relu(self.global_proj(u))

        fused_emb = torch.cat([graph_emb, global_emb], dim=-1)
        out = self.head(fused_emb)

        return out, h
