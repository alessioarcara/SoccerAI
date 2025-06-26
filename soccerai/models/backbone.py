from typing import Callable, Dict, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.typing import Adj, OptTensor

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
            raise ValueError(f"Backbone '{name}' not registered.")
        return cls._registry[name](*args, **kwargs)


class Identity(nn.Identity):
    def forward(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return input


NORMALIZATION: Dict[str, Type[nn.Module]] = {
    "layer": pyg_nn.LayerNorm,
    "instance": pyg_nn.InstanceNorm,
    "graph": pyg_nn.GraphNorm,
    "none": Identity,
}


@BackboneRegistry.register("gcn")
class GCNBackbone(nn.Module):
    def __init__(self, din: int, cfg: BackboneConfig):
        super().__init__()
        self.conv1 = pyg_nn.GCNConv(din, cfg.dhid)
        self.drop = nn.Dropout(cfg.drop)
        self.norm1 = NORMALIZATION[cfg.norm](cfg.dhid)
        self.conv2 = pyg_nn.GCNConv(cfg.dhid, cfg.dout)
        self.norm2 = NORMALIZATION[cfg.norm](cfg.dout)

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
        x = F.relu(
            self.norm1(
                self.conv1(x, edge_index, edge_weight),
                batch=batch,
                batch_size=batch_size,
            )
        )

        if residual is None:
            residual = torch.zeros_like(x, device=x.device)

        return F.relu(
            self.norm2(
                self.conv2(self.drop(x) + residual, edge_index, edge_weight),
                batch=batch,
                batch_size=batch_size,
            )
        )


@BackboneRegistry.register("gcn2")
class GCNIIBackbone(nn.Module):
    def __init__(self, din: int, cfg: BackboneConfig):
        super().__init__()

        if din != cfg.dhid:
            self.lin1 = pyg_nn.Linear(din, cfg.dhid)
        else:
            self.lin1 = nn.Identity()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(cfg.n_layers):
            self.convs.append(
                pyg_nn.GCN2Conv(
                    cfg.dhid, alpha=0.5, theta=1.0, layer=i + 1, shared_weights=False
                )
            )
            self.norms.append(NORMALIZATION[cfg.norm](cfg.dhid))
        if cfg.dhid != cfg.dout:
            self.lin2 = pyg_nn.Linear(cfg.dhid, cfg.dout)
        else:
            self.lin2 = nn.Identity()

        self.drop = nn.Dropout(cfg.drop)
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
        x_0 = F.relu(self.lin1(x))
        if residual is None:
            residual = torch.zeros_like(x_0, device=x_0.device)

        x = x_0
        for i in range(len(self.convs)):
            if i % self.skip_stride == 0:
                res = x_0 + residual
            else:
                res = x_0
            x = F.relu(
                self.norms[i](
                    self.convs[i](
                        self.drop(x),
                        res,
                        edge_index,
                        edge_weight,
                    ),
                    batch=batch,
                    batch_size=batch_size,
                )
            )

        return F.relu(self.lin2(self.drop(x)))


class GraphSAGEBackbone(nn.Module):
    def __init__(
        self,
        din: int,
        dmid: int,
        dout: int,
        num_layers: int = 2,
        aggr_type: str = "mean",
        l2_norm: bool = True,
        dropout: float = 0.0,
    ):
        super(GraphSAGEBackbone, self).__init__()
        aggr_type = aggr_type.lower()
        if aggr_type not in {"mean", "pool", "lstm"}:
            raise ValueError("aggr_type must be 'mean', 'pool', or 'lstm'.")
        self.aggr_type = aggr_type
        self.dropout = dropout

        self.convs = nn.ModuleList()

        for i in range(num_layers):
            in_dim = din if i == 0 else dmid
            out_dim = dout if i == num_layers - 1 else dmid

            if aggr_type == "mean":
                conv = pyg_nn.SAGEConv(in_dim, out_dim, aggr="mean", normalize=l2_norm)

            elif aggr_type == "pool":
                conv = pyg_nn.SAGEConv(
                    in_dim, out_dim, aggr="max", project=True, normalize=l2_norm
                )

            else:  # lstm
                conv = pyg_nn.SAGEConv(in_dim, out_dim, aggr="lstm", normalize=l2_norm)

            self.convs.append(conv)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: OptTensor = None,
        prev_h: OptTensor = None,
    ):
        x = F.relu(self.convs[0](x, edge_index), inplace=True)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        if prev_h is None:
            prev_h = torch.zeros_like(x)

        for conv in self.convs[1:]:
            x = F.relu(conv(x + prev_h, edge_index), inplace=True)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x
