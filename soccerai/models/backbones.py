from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import dropout_edge

from soccerai.models.layers import BatchNorm, GNNPlusLayer, Identity
from soccerai.models.typings import NormalizationType
from soccerai.models.utils import build_layers, build_mlp
from soccerai.training.trainer_config import (
    GATv2Config,
    GCN2Config,
    GCNConfig,
    GINEConfig,
    GraphGPSConfig,
    GraphSAGEConfig,
)


class BackboneRegistry:
    _registry: dict[str, type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[nn.Module]], type[nn.Module]]:
        def decorator(backbone: type[nn.Module]) -> type[nn.Module]:
            cls._registry[name] = backbone
            return backbone

        return decorator

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> nn.Module:
        if name not in cls._registry:
            raise ValueError(f"Backbone '{name}' not registered")
        return cls._registry[name](*args, **kwargs)


NORMALIZATIONS: dict[NormalizationType, type[nn.Module]] = {
    "none": Identity,
    "batch": BatchNorm,
    "layer": pyg_nn.LayerNorm,
    "instance": pyg_nn.InstanceNorm,
    "graph": pyg_nn.GraphNorm,
}

# names understood by `torch_geometric.nn.GPSConv(norm=...)`
GPS_NORMALIZATIONS: dict[NormalizationType, str | None] = {
    "none": None,
    "batch": "batch_norm",
    "layer": "layer_norm",
    "instance": "instance_norm",
    "graph": "graph_norm",
}


def apply_layer(
    conv: nn.Module,
    norm: nn.Module,
    drop: nn.Module,
    h: torch.Tensor,
    batch: OptTensor,
    batch_size: int | None,
    **conv_kwargs,
) -> torch.Tensor:
    """
    One message-passing block: conv -> norm -> ReLU -> dropout.

    A `GNNPlusLayer` already contains its own normalisation, activation,
    dropout and residual connections, so it is applied as is.
    """
    if isinstance(conv, GNNPlusLayer):
        return conv(h, batch=batch, batch_size=batch_size, **conv_kwargs)

    return drop(
        F.relu(
            norm(conv(h, **conv_kwargs), batch=batch, batch_size=batch_size),
            inplace=True,
        )
    )


@BackboneRegistry.register("gcn")
class GCNBackbone(nn.Module):
    def __init__(self, din: int, cfg: GCNConfig):
        super().__init__()
        self.out_dim = cfg.dout
        self.drop = nn.Dropout(cfg.drop)

        def conv_fn(d, _):
            conv = pyg_nn.GCNConv(d, cfg.dout)
            if cfg.plus:
                return GNNPlusLayer(
                    conv,
                    d,
                    cfg.dout,
                    cfg.drop,
                    NORMALIZATIONS[cfg.norm](cfg.dout),
                )
            else:
                return conv

        def norm_fn(_):
            # a GNN+ layer normalises internally: do not normalise twice
            return Identity() if cfg.plus else NORMALIZATIONS[cfg.norm](cfg.dout)

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
        batch_size: int | None = None,
    ):
        h = x

        for conv, norm in zip(self.convs, self.norms):
            h = apply_layer(
                conv,
                norm,
                self.drop,
                h,
                batch,
                batch_size,
                edge_index=edge_index,
                edge_weight=edge_weight,
            )

        return h


@BackboneRegistry.register("gcn2")
class GCNIIBackbone(nn.Module):
    def __init__(self, din: int, cfg: GCN2Config):
        super().__init__()
        self.out_dim = cfg.dout
        self.drop = nn.Dropout(cfg.drop)
        self.node_proj = pyg_nn.Linear(din, cfg.dout)

        def conv_fn(d, i):
            return pyg_nn.GCN2Conv(
                cfg.dout,
                alpha=0.5,
                theta=1.0,
                layer=i + 1,
                shared_weights=False,
            )

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
        batch_size: int | None = None,
    ):
        h = h0 = self.node_proj(x)

        for conv, norm in zip(self.convs, self.norms):
            h = apply_layer(
                conv,
                norm,
                self.drop,
                h,
                batch,
                batch_size,
                x_0=h0,
                edge_index=edge_index,
                edge_weight=edge_weight,
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
        self.out_dim = cfg.dout
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
        batch_size: int | None = None,
    ):
        h = x

        for conv, norm in zip(self.convs, self.norms):
            h = apply_layer(
                conv, norm, self.drop, h, batch, batch_size, edge_index=edge_index
            )

        return h


@BackboneRegistry.register("gatv2")
class GATv2Backbone(nn.Module):
    def __init__(self, din: int, cfg: GATv2Config):
        super().__init__()
        self.out_dim = cfg.dout
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
        batch_size: int | None = None,
    ) -> torch.Tensor:
        h = x
        n_layers = len(self.convs)

        # drop edges once per forward pass (dropping again at every layer
        # would compound the rate: p=0.5 keeps 25% of the edges at layer 2)
        edge_index, edge_mask = dropout_edge(
            edge_index, p=self.edge_dropout, training=self.training
        )
        edge_attr = (
            edge_attr[edge_mask]
            if (self.use_edge_attr and edge_attr is not None)
            else None
        )

        for layer_idx, conv in enumerate(self.convs):
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
            else:
                return conv

        def norm_fn(_):
            # a GNN+ layer normalises internally: do not normalise twice
            return Identity() if cfg.plus else NORMALIZATIONS[cfg.norm](cfg.dout)

        self.convs, self.norms = build_layers(
            n_layers=cfg.n_layers,
            din=din,
            dout=cfg.dout,
            conv_factory=conv_fn,
            norm_factory=norm_fn,
        )
        # jumping-knowledge style output: the embeddings of every layer
        self.out_dim = cfg.n_layers * cfg.dout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: int | None = None,
    ):
        outs = []
        h = x

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)

        for conv, norm in zip(self.convs, self.norms):
            h = apply_layer(
                conv,
                norm,
                self.drop,
                h,
                batch,
                batch_size,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )
            outs.append(h)

        return outs


@BackboneRegistry.register("graphgps")
class GraphGPSBackbone(nn.Module):
    def __init__(self, din: int, cfg: GraphGPSConfig):
        super().__init__()
        self.out_dim = cfg.dout

        self.node_proj = nn.Linear(din, cfg.dout)
        self.edge_proj = nn.Linear(1, cfg.dout)

        def conv_fn(d, _):
            return pyg_nn.GPSConv(
                cfg.dout,
                pyg_nn.GINEConv(build_mlp(cfg.dout, cfg.dout)),
                heads=cfg.heads,
                dropout=cfg.drop,
                norm=GPS_NORMALIZATIONS[cfg.norm],
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
        batch_size: int | None = None,
    ):
        h = self.node_proj(x)

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            edge_attr = self.edge_proj(edge_attr)

        for conv in self.convs:
            h = conv(
                h,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch,
            )

        return h
