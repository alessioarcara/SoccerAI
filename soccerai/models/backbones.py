from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import dropout_edge

from soccerai.models.layers import BatchNorm, GNNPlusLayer, Identity
from soccerai.models.typings import AggregationType, NormalizationType
from soccerai.models.utils import build_layers, build_mlp

NORMALIZATIONS: dict[NormalizationType, Callable[..., nn.Module]] = {
    "none": Identity,
    "batch": BatchNorm,
    # PyG's LayerNorm normalises over all nodes and channels of a graph
    "layer": pyg_nn.LayerNorm,
    # per-node LayerNorm: unlike "graph"/"instance"/"layer" it does not subtract
    # a per-graph mean, so where the whole play takes place survives
    "node": partial(pyg_nn.LayerNorm, mode="node"),
    "instance": pyg_nn.InstanceNorm,
    "graph": pyg_nn.GraphNorm,
}

# names understood by `torch_geometric.nn.GPSConv(norm=...)`
GPS_NORMALIZATIONS: dict[NormalizationType, str | None] = {
    "none": None,
    "batch": "batch_norm",
    "layer": "layer_norm",
    "node": "layer_norm",
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


class GCNBackbone(nn.Module):
    def __init__(
        self,
        din: int,
        n_layers: int,
        dout: int,
        drop: float = 0.1,
        norm: NormalizationType = "graph",
        plus: bool = False,
    ):
        super().__init__()
        self.out_dim = dout
        self.drop = nn.Dropout(drop)

        def conv_fn(d, _):
            conv = pyg_nn.GCNConv(d, dout)
            if plus:
                return GNNPlusLayer(
                    conv,
                    d,
                    dout,
                    drop,
                    NORMALIZATIONS[norm](dout),
                )
            else:
                return conv

        def norm_fn(_):
            # a GNN+ layer normalises internally: do not normalise twice
            return Identity() if plus else NORMALIZATIONS[norm](dout)

        self.convs, self.norms = build_layers(
            n_layers=n_layers,
            din=din,
            dout=dout,
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


class GCNIIBackbone(nn.Module):
    def __init__(
        self,
        din: int,
        n_layers: int,
        dout: int,
        drop: float = 0.1,
        norm: NormalizationType = "graph",
    ):
        super().__init__()
        self.out_dim = dout
        self.drop = nn.Dropout(drop)
        self.node_proj = pyg_nn.Linear(din, dout)

        def conv_fn(d, i):
            return pyg_nn.GCN2Conv(
                dout,
                alpha=0.5,
                theta=1.0,
                layer=i + 1,
                shared_weights=False,
            )

        def norm_fn(_):
            return NORMALIZATIONS[norm](dout)

        self.convs, self.norms = build_layers(
            n_layers=n_layers,
            din=dout,
            dout=dout,
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


class GraphSAGEBackbone(nn.Module):
    def __init__(
        self,
        din: int,
        n_layers: int,
        dout: int,
        drop: float = 0.1,
        norm: NormalizationType = "graph",
        aggr_type: AggregationType = "max",
        l2_norm: bool = True,
    ):
        super().__init__()
        self.out_dim = dout
        self.drop = nn.Dropout(drop)

        def conv_fn(d, _):
            return pyg_nn.SAGEConv(
                in_channels=d,
                out_channels=dout,
                aggr=aggr_type,
                project=(aggr_type == "max"),
                normalize=l2_norm,
            )

        def norm_fn(_):
            return NORMALIZATIONS[norm](dout)

        self.convs, self.norms = build_layers(
            n_layers=n_layers,
            din=din,
            dout=dout,
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


class GATv2Backbone(nn.Module):
    def __init__(
        self,
        din: int,
        n_layers: int,
        dout: int,
        drop: float = 0.1,
        norm: NormalizationType = "graph",
        num_heads: int = 4,
        use_edge_attr: bool = True,
        edge_dropout: float = 0.0,
    ):
        super().__init__()
        self.out_dim = dout
        self.use_edge_attr = use_edge_attr
        self.edge_dropout = edge_dropout
        self.drop = nn.Dropout(drop)

        def conv_fn(d, i):
            return pyg_nn.GATv2Conv(
                in_channels=d,
                out_channels=(dout if i == n_layers - 1 else dout // num_heads),
                heads=num_heads,
                concat=(i < n_layers - 1),
                dropout=drop,
                edge_dim=(1 if use_edge_attr else None),
            )

        def norm_fn(i):
            return NORMALIZATIONS[norm](dout) if i < n_layers - 1 else Identity()

        self.convs, self.norms = build_layers(
            n_layers=n_layers,
            din=din,
            dout=dout,
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


class GINEBackbone(nn.Module):
    def __init__(
        self,
        din: int,
        n_layers: int,
        dout: int,
        drop: float = 0.1,
        norm: NormalizationType = "graph",
        train_eps: bool = True,
        plus: bool = False,
    ):
        super().__init__()
        self.drop = nn.Dropout(drop)

        def conv_fn(d, _):
            conv = pyg_nn.GINEConv(
                nn=build_mlp(d, dout),
                edge_dim=1,
                train_eps=train_eps,
            )

            if plus:
                return GNNPlusLayer(conv, d, dout, drop, NORMALIZATIONS[norm](dout))
            else:
                return conv

        def norm_fn(_):
            # a GNN+ layer normalises internally: do not normalise twice
            return Identity() if plus else NORMALIZATIONS[norm](dout)

        self.convs, self.norms = build_layers(
            n_layers=n_layers,
            din=din,
            dout=dout,
            conv_factory=conv_fn,
            norm_factory=norm_fn,
        )
        # jumping-knowledge style output: the embeddings of every layer
        self.out_dim = n_layers * dout

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


class GraphGPSBackbone(nn.Module):
    def __init__(
        self,
        din: int,
        n_layers: int,
        dout: int,
        drop: float = 0.1,
        norm: NormalizationType = "layer",
        heads: int = 4,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.out_dim = dout

        self.node_proj = nn.Linear(din, dout)
        self.edge_proj = nn.Linear(1, dout)

        def conv_fn(d, _):
            return pyg_nn.GPSConv(
                dout,
                pyg_nn.GINEConv(build_mlp(dout, dout)),
                heads=heads,
                dropout=drop,
                norm=GPS_NORMALIZATIONS[norm],
                norm_kwargs={"mode": "node"} if norm == "node" else None,
                attn_kwargs={"dropout": attn_drop},
            )

        def norm_fn(_):
            return Identity()

        self.convs, _ = build_layers(
            n_layers=n_layers,
            din=dout,
            dout=dout,
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
