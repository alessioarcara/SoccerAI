from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.typing import Adj, OptTensor


class Identity(nn.Identity):
    def forward(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return input


class GCNBackbone(nn.Module):
    def __init__(self, din: int, dmid: int, dout: int):
        super(GCNBackbone, self).__init__()
        self.conv1 = pyg_nn.GCNConv(din, dmid)
        self.conv2 = pyg_nn.GCNConv(dmid, dout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
        prev_h: OptTensor = None,
    ):
        f = F.relu(self.conv1(x, edge_index, edge_weight))

        if prev_h is None:
            prev_h = torch.zeros_like(f, device=f.device)

        return F.relu(self.conv2(f + prev_h, edge_index, edge_weight))


class GCNIIBackbone(nn.Module):
    def __init__(
        self,
        din: int,
        dmid: int,
        dout: int,
        n_layers: int,
        use_norm: bool,
    ):
        super(GCNIIBackbone, self).__init__()
        if din != dmid:
            self.lin1 = pyg_nn.Linear(din, dmid)
        else:
            self.lin1 = nn.Identity()
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(
                pyg_nn.GCN2Conv(
                    dmid, alpha=0.5, theta=1.0, layer=i + 1, shared_weights=False
                )
            )
        if use_norm:
            self.norm = pyg_nn.LayerNorm(dmid)
        else:
            self.norm = Identity()

        if dmid != dout:
            self.lin2 = pyg_nn.Linear(dmid, dout)
        else:
            self.lin2 = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
        prev_h: OptTensor = None,
    ):
        f_0 = F.relu(self.lin1(x))
        if prev_h is None:
            prev_h = torch.zeros_like(f_0, device=f_0.device)
        f = f_0
        for i in range(len(self.convs)):
            f = F.relu(
                self.norm(
                    self.convs[i](f, f_0 + prev_h, edge_index, edge_weight),
                    batch=batch,
                    batch_size=batch_size,
                )
            )

        return F.relu(self.lin2(f))
