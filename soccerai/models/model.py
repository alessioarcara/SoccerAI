from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric_temporal.nn as pygt_nn
from torch_geometric.typing import Adj, OptTensor

from soccerai.data.dataset import WorldCup2022Dataset
from soccerai.models.backbones import GCNBackbone, GCNIIBackbone
from soccerai.models.heads import GraphClassificationHead
from soccerai.models.utils import get_norm, get_readout
from soccerai.training.trainer_config import Config


def create_model(cfg: Config, train_ds: WorldCup2022Dataset) -> nn.Module:
    match cfg.model.model_name:
        case "gcn":
            return GCN(
                train_ds.num_node_features,
                train_ds.num_global_features,
                cfg.model.dmid,
                cfg.model.head_drop,
                cfg.model.conv_drop,
                get_norm(cfg.model.norm),
                get_readout(cfg.model.readout),
            )
        case "gcn2":
            return GCNII(
                train_ds.num_node_features,
                train_ds.num_global_features,
                cfg.model.dmid,
                cfg.model.head_drop,
                cfg.model.conv_drop,
                cfg.model.n_layers,
                get_norm(cfg.model.norm),
                get_readout(cfg.model.readout),
                cfg.model.skip_stride,
            )
        case "gcrnn":
            return GCRNN(
                train_ds.num_node_features,
                train_ds.num_global_features,
                cfg.model.backbone,
                cfg.model.n_layers,
                cfg.model.head_drop,
                cfg.model.conv_drop,
                get_norm(cfg.model.norm),
                get_readout(cfg.model.readout),
                cfg.model.skip_stride,
            )
        case _:
            raise ValueError("Invalid model name")


class GCN(torch.nn.Module):
    def __init__(
        self,
        node_feature_din: int,
        glob_feature_din: int,
        dmid: int,
        head_drop: float,
        conv_drop: float,
        norm: Optional[nn.Module],
        readout: pyg_nn.Aggregation,
        dout: int = 1,
    ):
        super(GCN, self).__init__()
        self.backbone = GCNBackbone(node_feature_din, dmid, dmid, conv_drop, norm)
        self.global_proj = nn.Linear(glob_feature_din, dmid)
        self.readout = readout
        self.head = GraphClassificationHead(dmid * 2, dout, head_drop)

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
        graph_emb = self.readout(node_emb, batch, dim_size=batch_size)
        global_emb = F.relu(self.global_proj(u), inplace=True)

        fused_emb = torch.cat([graph_emb, global_emb], dim=-1)
        return self.head(fused_emb)


class GCNII(nn.Module):
    def __init__(
        self,
        node_feature_din: int,
        glob_feature_din: int,
        dmid: int,
        head_drop: float,
        conv_drop: float,
        n_layers: int,
        norm: Optional[nn.Module],
        readout: pyg_nn.Aggregation,
        skip_stride: int,
        dout: int = 1,
    ):
        super(GCNII, self).__init__()
        self.backbone = GCNIIBackbone(
            node_feature_din, dmid, dmid, n_layers, norm, skip_stride, conv_drop
        )
        self.global_proj = nn.Linear(glob_feature_din, dmid)
        self.readout = readout
        self.head = GraphClassificationHead(dmid * 2, dout, head_drop)

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
        graph_emb = self.readout(node_emb, batch, dim_size=batch_size)
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
        head_drop: float,
        conv_drop: float,
        norm: Optional[nn.Module],
        readout: pyg_nn.Aggregation,
        skip_stride: int,
        dout: int = 1,
    ):
        super(GCRNN, self).__init__()
        self.backbone: nn.Module
        match backbone:
            case "gcn":
                self.backbone = GCNBackbone(node_feature_din, 256, 128, conv_drop, norm)
            case "gcn2":
                self.backbone = GCNIIBackbone(
                    node_feature_din,
                    256,
                    128,
                    n_layers,
                    norm,
                    skip_stride,
                    conv_drop,
                )
        self.global_proj = nn.Linear(glob_feature_din, 128)

        self.gcrn = pygt_nn.recurrent.GConvGRU(128 + node_feature_din, 256, 1)
        self.readout = readout
        self.head = GraphClassificationHead(256, dout, head_drop)

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
        z = self.backbone(
            x,
            edge_index,
            edge_weight,
            prev_h=prev_h,
            batch=batch,
            batch_size=batch_size,
        )

        h = self.gcrn(torch.concat([z, x], dim=-1), edge_index, edge_weight, prev_h)

        graph_emb = self.readout(z, index=batch, dim_size=batch_size)
        global_emb = F.relu(self.global_proj(u))

        fused_emb = torch.cat([graph_emb, global_emb], dim=-1)
        out = self.head(fused_emb)

        return out, h
