from typing import Callable, Dict, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.typing import Adj, OptTensor

from soccerai.models.typings import NormalizationType
from soccerai.training.trainer_config import BackboneConfig


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


class BatchNorm(nn.BatchNorm1d):
    def forward(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return super().forward(input)


NORMALIZATIONS: Dict[NormalizationType, Type[nn.Module]] = {
    "none": Identity,
    "batch": BatchNorm,
    "layer": pyg_nn.LayerNorm,
    "instance": pyg_nn.InstanceNorm,
    "graph": pyg_nn.GraphNorm,
}


@BackboneRegistry.register("gcn")
class GCNBackbone(nn.Module):
    def __init__(self, din: int, cfg: BackboneConfig):
        super().__init__()
        self.drop = nn.Dropout(cfg.drop)
        self.conv1 = pyg_nn.GCNConv(din, cfg.dhid)
        self.norm1 = NORMALIZATIONS[cfg.norm](cfg.dhid)
        self.conv2 = pyg_nn.GCNConv(cfg.dhid, cfg.dout)
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
    def __init__(self, din: int, cfg: BackboneConfig):
        super().__init__()

        self.drop = nn.Dropout(cfg.drop)
        self.lin1 = pyg_nn.Linear(din, cfg.dhid) if din != cfg.dhid else nn.Identity()

        self.convs = nn.ModuleList(
            [
                pyg_nn.GCN2Conv(
                    cfg.dhid, alpha=0.5, theta=1.0, layer=i + 1, shared_weights=False
                )
                for i in range(cfg.n_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [NORMALIZATIONS[cfg.norm](cfg.dhid) for _ in range(cfg.n_layers)]
        )

        self.lin2 = (
            pyg_nn.Linear(cfg.dhid, cfg.dout) if cfg.dhid != cfg.dout else nn.Identity()
        )

        self.skip_stride = cfg.skip_stride

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
        h = h0 = self.lin1(x)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
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

            if residual is not None and i > 0 and i % self.skip_stride == 0:
                h = h + residual

        return self.lin2(h)


@BackboneRegistry.register("graphsage")
class GraphSAGEBackbone(nn.Module):
    def __init__(
        self,
        din: int,
        cfg: BackboneConfig,
    ):
        super().__init__()
        self.drop = nn.Dropout(cfg.drop)

        project = cfg.aggr_type == "max"

        dims = [din] + [cfg.dhid] * (cfg.n_layers - 1) + [cfg.dout]

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]

            self.convs.append(
                pyg_nn.SAGEConv(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    aggr=cfg.aggr_type,
                    project=project,
                    normalize=cfg.l2_norm,
                )
            )
            self.norms.append(NORMALIZATIONS[cfg.norm](out_dim))

        self.skip_stride = cfg.skip_stride

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
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = self.drop(
                F.relu(
                    norm(conv(h, edge_index), batch=batch, batch_size=batch_size),
                )
            )

            if residual is not None and i > 0 and i % self.skip_stride:
                h = h + residual

        return h


@BackboneRegistry.register("gatv2")
class GATv2Backbone(nn.Module):
    def __init__(self, din: int, cfg: BackboneConfig):
        super().__init__()

        self.use_edge_attr = cfg.use_edge_attr
        self.drop = nn.Dropout(cfg.drop)
        self.skip_stride = cfg.skip_stride

        dims = [din] + [cfg.dhid] * (cfg.n_layers - 1) + [cfg.dout]

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            is_last = i == cfg.n_layers - 1
            out_per_head = out_dim if is_last else out_dim // cfg.num_heads

            self.convs.append(
                pyg_nn.GATv2Conv(
                    in_channels=in_dim,
                    out_channels=out_per_head,
                    heads=cfg.num_heads,
                    concat=not is_last,
                    dropout=cfg.drop,
                    edge_dim=1 if cfg.use_edge_attr else None,
                )
            )

            if not is_last:
                self.norms.append(NORMALIZATIONS[cfg.norm](out_dim))

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
        edge_attr_weight = (
            edge_weight.unsqueeze(-1)
            if self.use_edge_attr and edge_weight is not None
            else None
        )
        h = self.drop(x)

        for i, conv in enumerate(self.convs):
            if residual is not None and i > 0 and i % self.skip_stride == 0:
                h = h + residual

            h = conv(h, edge_index, edge_attr=edge_attr_weight)

            if i < len(self.convs) - 1:
                h = self.drop(
                    F.elu(
                        self.norms[i](h, batch=batch, batch_size=batch_size),
                        inplace=True,
                    )
                )

        return h
