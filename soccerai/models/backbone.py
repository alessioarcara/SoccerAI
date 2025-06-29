from typing import Callable, Dict, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import dropout_edge

from soccerai.models.typings import NormalizationType
from soccerai.models.utils import build_layers, sum_residual
from soccerai.training.trainer_config import (
    GATv2Config,
    GCN2Config,
    GCNConfig,
    GINEConfig,
    GraphGPSConfig,
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


def build_mlp(din: int, dmid: int, dout: Optional[int] = None) -> nn.Sequential:
    if dout is None:
        dout = dmid
    return nn.Sequential(nn.Linear(din, dmid), nn.ReLU(), nn.Linear(dmid, dout))


class GNNPlusLayer(nn.Module):
    def __init__(
        self,
        conv_layer: nn.Module,
        din: int,
        dout: int,
        p_drop: float,
        norm: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.proj = pyg_nn.Linear(din, dout) if din != dout else Identity()

        self.norm = norm if norm is not None else Identity()

        self.drop = nn.Dropout(p_drop)

        self.conv_layer = conv_layer

        self.mlp = build_mlp(dout, dout * 2, dout)

    def forward(
        self,
        x: torch.Tensor,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
        **conv_kwargs,
    ) -> torch.Tensor:
        x_in = self.proj(x)

        x = F.relu(
            self.drop(
                self.norm(
                    self.conv_layer(x, **conv_kwargs),
                    batch,
                    batch_size,
                )
            )
        )

        return x + self.mlp(x_in + x)


@BackboneRegistry.register("gcn")
class GCNBackbone(nn.Module):
    def __init__(self, din: int, cfg: GCNConfig):
        super().__init__()
        self.residual_sum_mode = cfg.residual_sum_mode
        self.drop = nn.Dropout(cfg.drop)

        def conv_fn(d, _):
            if cfg.plus:
                return GNNPlusLayer(
                    pyg_nn.GCNConv(d, cfg.dout),
                    d,
                    cfg.dout,
                    cfg.drop,
                    NORMALIZATIONS[cfg.norm](cfg.dout),
                )
            return pyg_nn.GCNConv(d, cfg.dout)

        def norm_fn(_):
            return NORMALIZATIONS[cfg.norm](cfg.dout)

        self.convs, self.norms = build_layers(
            n_layers=cfg.n_layers,
            din=din,
            dout=cfg.dout,
            conv_factory=conv_fn,
            norm_factory=norm_fn,
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
        h = x

        for layer_idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = sum_residual(
                h,
                residual,
                self.residual_sum_mode,
                layer_idx=layer_idx,
                n_layers=len(self.convs),
            )
            h = self.drop(
                F.relu(
                    norm(
                        conv(h, edge_index=edge_index, edge_weight=edge_weight),
                        batch=batch,
                        batch_size=batch_size,
                    ),
                    inplace=True,
                )
            )

        return h


@BackboneRegistry.register("gcn2")
class GCNIIBackbone(nn.Module):
    def __init__(self, din: int, cfg: GCN2Config):
        super().__init__()
        self.residual_sum_mode = cfg.residual_sum_mode
        self.drop = nn.Dropout(cfg.drop)
        self.node_proj = pyg_nn.Linear(din, cfg.dout)

        def conv_fn(d, i):
            conv = pyg_nn.GCN2Conv(
                cfg.dout,
                alpha=0.5,
                theta=1.0,
                layer=i + 1,
                shared_weights=False,
            )
            if cfg.plus:
                return GNNPlusLayer(
                    conv,
                    d,
                    cfg.dout,
                    cfg.drop,
                    NORMALIZATIONS[cfg.norm](cfg.dout),
                )
            return conv

        def norm_fn(_):
            return NORMALIZATIONS[cfg.norm](cfg.dout)

        self.convs, self.norms = build_layers(
            n_layers=cfg.n_layers,
            din=cfg.dout,
            dout=cfg.dout,
            conv_factory=conv_fn,
            norm_factory=norm_fn,
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

        for layer_idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = sum_residual(
                h,
                residual,
                self.residual_sum_mode,
                layer_idx=layer_idx,
                n_layers=len(self.convs),
            )
            h = self.drop(
                F.relu(
                    norm(
                        conv(
                            h,
                            x_0=h0,
                            edge_index=edge_index,
                            edge_weight=edge_weight,
                        ),
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

        def conv_fn(d, _):
            return pyg_nn.SAGEConv(
                in_channels=d,
                out_channels=cfg.dout,
                aggr=cfg.aggr_type,
                project=(cfg.aggr_type == "max"),
                normalize=cfg.l2_norm,
            )

        def norm_fn(_):
            return NORMALIZATIONS[cfg.norm](cfg.dout)

        self.convs, self.norms = build_layers(
            n_layers=cfg.n_layers,
            din=din,
            dout=cfg.dout,
            conv_factory=conv_fn,
            norm_factory=norm_fn,
        )

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

        for layer_idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = sum_residual(
                h,
                residual,
                self.residual_sum_mode,
                layer_idx=layer_idx,
                n_layers=len(self.convs),
            )
            h = self.drop(
                F.relu(
                    norm(conv(h, edge_index), batch=batch, batch_size=batch_size),
                )
            )

        return h


@BackboneRegistry.register("gatv2")
class GATv2Backbone(nn.Module):
    def __init__(self, din: int, cfg: GATv2Config):
        super().__init__()
        self.residual_sum_mode = cfg.residual_sum_mode
        self.use_edge_attr = cfg.use_edge_attr
        self.edge_dropout = cfg.edge_dropout
        self.drop = nn.Dropout(cfg.drop)

        def conv_fn(d, i):
            return pyg_nn.GATv2Conv(
                in_channels=d,
                out_channels=(
                    cfg.dout if i == cfg.n_layers - 1 else cfg.dout // cfg.num_heads
                ),
                heads=cfg.num_heads,
                concat=(i < cfg.n_layers - 1),
                dropout=cfg.drop,
                edge_dim=(1 if cfg.use_edge_attr else None),
            )

        def norm_fn(i):
            return (
                NORMALIZATIONS[cfg.norm](cfg.dout)
                if i < cfg.n_layers - 1
                else Identity()
            )

        self.convs, self.norms = build_layers(
            n_layers=cfg.n_layers,
            din=din,
            dout=cfg.dout,
            conv_factory=conv_fn,
            norm_factory=norm_fn,
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
        h = x
        n_layers = len(self.convs)

        for layer_idx, conv in enumerate(self.convs):
            edge_index, edge_mask = dropout_edge(
                edge_index,
                p=self.edge_dropout,
                training=self.training,
            )

            h = sum_residual(
                h,
                residual,
                self.residual_sum_mode,
                layer_idx=layer_idx,
                n_layers=n_layers,
            )

            edge_attr = (
                edge_attr[edge_mask]
                if (self.use_edge_attr and edge_attr is not None)
                else None
            )

            h = conv(h, edge_index, edge_attr=edge_attr)
            # Skip dropout, norm, and activation on last layer:
            # Final GAT layer averages heads (concat=False); further normalization would compress attention differences.
            if layer_idx < n_layers - 1:
                h = self.drop(
                    F.elu(
                        self.norms[layer_idx](h, batch=batch, batch_size=batch_size),
                        inplace=True,
                    )
                )

        return h


@BackboneRegistry.register("gine")
class GINEBackbone(nn.Module):
    def __init__(self, din: int, cfg: GINEConfig):
        super().__init__()
        self.drop = nn.Dropout(cfg.drop)

        def conv_fn(d, _):
            conv = pyg_nn.GINEConv(
                nn=build_mlp(d, cfg.dout),
                edge_dim=1,
                train_eps=cfg.train_eps,
            )

            if cfg.plus:
                return GNNPlusLayer(
                    conv, d, cfg.dout, cfg.drop, NORMALIZATIONS[cfg.norm](cfg.dout)
                )

            return conv

        def norm_fn(_):
            return NORMALIZATIONS[cfg.norm](cfg.dout)

        self.convs, self.norms = build_layers(
            n_layers=cfg.n_layers,
            din=din,
            dout=cfg.dout,
            conv_factory=conv_fn,
            norm_factory=norm_fn,
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
        outs = []
        h = x

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)

        for conv, norm in zip(self.convs, self.norms):
            h = self.drop(
                F.relu(
                    norm(
                        conv(h, edge_index=edge_index, edge_attr=edge_attr),
                        batch=batch,
                        batch_size=batch_size,
                    ),
                    inplace=True,
                )
            )
            outs.append(h)

        return outs


@BackboneRegistry.register("graphgps")
class GraphGPSBackbone(nn.Module):
    def __init__(self, din: int, cfg: GraphGPSConfig):
        super().__init__()
        self.residual_sum_mode = cfg.residual_sum_mode

        self.node_proj = nn.Linear(din, cfg.dout)
        self.edge_proj = nn.Linear(1, cfg.dout)

        def conv_fn(d, _):
            return pyg_nn.GPSConv(
                cfg.dout,
                pyg_nn.GINEConv(build_mlp(cfg.dout, cfg.dout)),
                heads=cfg.heads,
                dropout=cfg.drop,
                attn_kwargs={"dropout": cfg.attn_drop},
            )

        def norm_fn(_):
            return Identity()

        self.convs, _ = build_layers(
            n_layers=cfg.n_layers,
            din=cfg.dout,
            dout=cfg.dout,
            conv_factory=conv_fn,
            norm_factory=norm_fn,
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
        h = self.node_proj(x)

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            edge_attr = self.edge_proj(edge_attr)

        for layer_idx, conv in enumerate(self.convs):
            h = sum_residual(
                h,
                residual,
                self.residual_sum_mode,
                layer_idx=layer_idx,
                n_layers=len(self.convs),
            )
            h = conv(
                h,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch,
            )

        return h
