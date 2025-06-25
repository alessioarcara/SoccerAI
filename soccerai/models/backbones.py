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
    def __init__(
        self, din: int, dmid: int, dout: int, p_drop: float, norm: Optional[nn.Module]
    ):
        super(GCNBackbone, self).__init__()
        self.conv1 = pyg_nn.GCNConv(din, dmid)
        self.drop = nn.Dropout(p_drop)
        if norm is not None:
            self.norm = norm(dmid)
        else:
            self.norm = Identity()
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
        f = F.relu(
            self.norm(
                self.conv1(x, edge_index, edge_weight),
                batch=batch,
                batch_size=batch_size,
            )
        )

        if prev_h is None:
            prev_h = torch.zeros_like(f, device=f.device)

        return F.relu(self.conv2(self.drop(f) + prev_h, edge_index, edge_weight))


class GCNIIBackbone(nn.Module):
    def __init__(
        self,
        din: int,
        dmid: int,
        dout: int,
        n_layers: int,
        norm: Optional[nn.Module],
        skip_stride: int,
        p_drop: float,
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
        if norm is not None:
            self.norm = norm(dmid)
        else:
            self.norm = Identity()

        if dmid != dout:
            self.lin2 = pyg_nn.Linear(dmid, dout)
        else:
            self.lin2 = nn.Identity()
        self.drop = nn.Dropout(p_drop)
        self.skip_stride = skip_stride

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
            if i % self.skip_stride == 0:
                res = f_0 + prev_h
            else:
                res = f_0
            f = F.relu(
                self.norm(
                    self.convs[i](
                        self.drop(f),
                        res,
                        edge_index,
                        edge_weight,
                    ),
                    batch=batch,
                    batch_size=batch_size,
                )
            )
        return F.relu(self.lin2(self.drop(f)))
