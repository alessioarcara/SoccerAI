from typing import Callable, Dict, List, Optional, Type

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


NORMALIZATIONS: Dict[NormalizationType, Type[nn.Module]] = {
    "none": Identity,
    "layer": pyg_nn.LayerNorm,
    "instance": pyg_nn.InstanceNorm,
    "graph": pyg_nn.GraphNorm,
}


def sum_residuals(
    residuals: Optional[List[OptTensor]], shape_template: torch.Tensor
) -> torch.Tensor:
    if residuals is not None:
        for i in range(len(residuals)):
            if residuals[i] is None:
                residuals[i] = torch.zeros_like(shape_template)

        residual = sum(residuals)
    else:
        residual = torch.zeros_like(shape_template)

    return residual


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
        residuals: Optional[List[OptTensor]] = None,
    ):
        h = F.relu(
            self.norm1(
                self.conv1(x, edge_index, edge_weight),
                batch=batch,
                batch_size=batch_size,
            ),
            inplace=True,
        )

        residual = sum_residuals(residuals, h)

        h = F.relu(
            self.norm2(
                self.conv2(self.drop(h) + residual, edge_index, edge_weight),
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
        residuals: Optional[List[OptTensor]] = None,
    ):
        h = h0 = self.lin1(x)

        residual = sum_residuals(residuals, h)

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
            if i > 0 and i % self.skip_stride == 0:
                h = h + residual

        return self.lin2(h)
