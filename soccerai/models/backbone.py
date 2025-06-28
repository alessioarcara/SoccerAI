from typing import Callable, Dict, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import dropout_edge

from soccerai.models.typings import NormalizationType, ResidualSumMode
from soccerai.training.trainer_config import (
    GATv2Config,
    GCN2Config,
    GCNConfig,
    GINEConfig,
    GraphSAGEConfig,
)


class BackboneRegistry:
    _registry: Dict[str, Type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
        def decorator(backbone: Type[nn.Module]) -> Type[nn.Module]:
            cls._registry[name] = backbone
            return backbone

        return decorator

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> nn.Module:
        if name not in cls._registry:
            raise ValueError(f"Backbone '{name}' not registered")
        return cls._registry[name](*args, **kwargs)


class Identity(nn.Identity):
    def forward(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return input


class BatchNorm(pyg_nn.BatchNorm):
    def forward(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return super().forward(input)


NORMALIZATIONS: Dict[NormalizationType, Type[nn.Module]] = {
    "none": Identity,
    "batch": BatchNorm,
    "layer": pyg_nn.LayerNorm,
    "instance": pyg_nn.InstanceNorm,
    "graph": pyg_nn.GraphNorm,
}


def sum_residual(
    h: torch.Tensor,
    residual: Optional[torch.Tensor],
    mode: ResidualSumMode,
    layer_idx: int,
    n_layers: int,
) -> torch.Tensor:
    """
    Apply residual tensor according to sum strategy.

    - 'every': add residual at all layers except the first (layer_idx > 0)
    - 'last': add residual only before the final layer (layer_idx == n_layers - 1)
    """
    if residual is None or mode == "none":
        return h

    if mode == "every" and layer_idx > 0:
        return h + residual

    if mode == "last" and layer_idx == n_layers - 1:
        return h + residual

    return h


@BackboneRegistry.register("gcn")
class GCNBackbone(nn.Module):
    def __init__(self, din: int, cfg: GCNConfig):
        super().__init__()
        self.drop = nn.Dropout(cfg.drop)
        self.conv1 = pyg_nn.GCNConv(din, cfg.dout)
        self.norm1 = NORMALIZATIONS[cfg.norm](cfg.dout)
        self.conv2 = pyg_nn.GCNConv(cfg.dout, cfg.dout)
        self.norm2 = NORMALIZATIONS[cfg.norm](cfg.dout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
        residual: OptTensor = None,
    ):
        h = self.drop(
            F.relu(
                self.norm1(
                    self.conv1(x, edge_index, edge_weight),
                    batch=batch,
                    batch_size=batch_size,
                ),
                inplace=True,
            )
        )

        if residual is not None:
            h = h + residual

        h = F.relu(
            self.norm2(
                self.conv2(h, edge_index, edge_weight),
                batch=batch,
                batch_size=batch_size,
            ),
            inplace=True,
        )
        return h


@BackboneRegistry.register("gcn2")
class GCNIIBackbone(nn.Module):
    def __init__(self, din: int, cfg: GCN2Config):
        super().__init__()
        self.residual_sum_mode = cfg.residual_sum_mode
        self.drop = nn.Dropout(cfg.drop)
        self.node_proj = pyg_nn.Linear(din, cfg.dout)

        self.convs = nn.ModuleList(
            [
                pyg_nn.GCN2Conv(
                    cfg.dout, alpha=0.5, theta=1.0, layer=i + 1, shared_weights=False
                )
                for i in range(cfg.n_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [NORMALIZATIONS[cfg.norm](cfg.dout) for _ in range(cfg.n_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
        residual: OptTensor = None,
    ):
        h = h0 = self.node_proj(x)
        n_layers = len(self.convs)

        for layer_idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = sum_residual(
                h,
                residual,
                self.residual_sum_mode,
                layer_idx=layer_idx,
                n_layers=n_layers,
            )
            h = self.drop(
                F.relu(
                    norm(
                        conv(h, h0, edge_index, edge_weight),
                        batch=batch,
                        batch_size=batch_size,
                    ),
                    inplace=True,
                )
            )

        return h


@BackboneRegistry.register("graphsage")
class GraphSAGEBackbone(nn.Module):
    def __init__(
        self,
        din: int,
        cfg: GraphSAGEConfig,
    ):
        super().__init__()
        self.residual_sum_mode = cfg.residual_sum_mode
        self.drop = nn.Dropout(cfg.drop)
        project = cfg.aggr_type == "max"

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(cfg.n_layers):
            self.convs.append(
                pyg_nn.SAGEConv(
                    in_channels=din,
                    out_channels=cfg.dout,
                    aggr=cfg.aggr_type,
                    project=project,
                    normalize=cfg.l2_norm,
                )
            )
            self.norms.append(NORMALIZATIONS[cfg.norm](cfg.dout))
            din = cfg.dout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
        residual: OptTensor = None,
    ):
        h = x
        n_layers = len(self.convs)

        for layer_idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = sum_residual(
                h,
                residual,
                self.residual_sum_mode,
                layer_idx=layer_idx,
                n_layers=n_layers,
            )
            h = self.drop(
                F.relu(
                    norm(conv(h, edge_index), batch=batch, batch_size=batch_size),
                )
            )

        return h


def build_mlp(din: int, dmid: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(din, dmid), nn.ReLU(), nn.Linear(dmid, dmid))


@BackboneRegistry.register("gine")
class GINEBackbone(nn.Module):
    def __init__(self, din: int, cfg: GINEConfig):
        super().__init__()
        self.drop = nn.Dropout(cfg.drop)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(cfg.n_layers):
            self.convs.append(
                pyg_nn.GINEConv(
                    nn=build_mlp(din, cfg.dout),
                    edge_dim=1,
                    train_eps=True,
                )
            )
            self.norms.append(NORMALIZATIONS[cfg.norm](cfg.dout))
            din = cfg.dout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
        residual: OptTensor = None,
    ):
        outs = []
        h = x

        if edge_attr is not None:
            edge_attr = edge_attr.unsqueeze(1)

        for conv, norm in zip(self.convs, self.norms):
            h = self.drop(
                F.relu(
                    norm(
                        conv(h, edge_index, edge_attr=edge_attr),
                        batch=batch,
                        batch_size=batch_size,
                    ),
                    inplace=True,
                )
            )
            outs.append(h)

        return outs


@BackboneRegistry.register("gatv2")
class GATv2Backbone(nn.Module):
    def __init__(self, din: int, cfg: GATv2Config):
        super().__init__()

        self.use_edge_attr = cfg.use_edge_attr
        self.edge_dropout = cfg.edge_dropout
        self.drop = nn.Dropout(cfg.drop)
        self.residual_sum_mode = cfg.residual_sum_mode

        dims = [din] + [cfg.dout] * cfg.n_layers

        # edge_attr: we feed our precomputed distance‐decay weights ([E,1])
        # into each GATv2Conv as attention biases via edge_dim=1
        self.convs = nn.ModuleList(
            [
                pyg_nn.GATv2Conv(
                    in_channels=in_dim,
                    out_channels=(
                        out_dim if i == cfg.n_layers - 1 else out_dim // cfg.num_heads
                    ),
                    heads=cfg.num_heads,
                    concat=(i != cfg.n_layers - 1),
                    dropout=cfg.drop,
                    edge_dim=(1 if cfg.use_edge_attr else None),
                )
                for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:]))
            ]
        )

        self.norms = nn.ModuleList(
            [NORMALIZATIONS[cfg.norm](out_dim) for out_dim in dims[1:-1]]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
        residual: OptTensor = None,
    ) -> torch.Tensor:
        h = self.drop(x)
        n_layers = len(self.convs)

        for layer_idx, conv in enumerate(self.convs):
            if self.training and self.edge_dropout > 0.0:
                edge_index, edge_mask = dropout_edge(
                    edge_index,
                    p=self.edge_dropout,
                    force_undirected=False,
                    training=True,
                )
                if edge_weight is not None:
                    edge_weight = edge_weight[edge_mask]

            h = sum_residual(
                h,
                residual,
                self.residual_sum_mode,
                layer_idx=layer_idx,
                n_layers=n_layers,
            )

            h = conv(
                h,
                edge_index,
                edge_attr=edge_weight if self.use_edge_attr else None,
            )

            if layer_idx < n_layers - 1:
                h = self.drop(
                    F.elu(
                        self.norms[layer_idx](h, batch=batch, batch_size=batch_size),
                        inplace=True,
                    )
                )

        return h
