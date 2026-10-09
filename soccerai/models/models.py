import torch
import torch.nn as nn
from torch_geometric.typing import Adj, OptTensor


class GNN(nn.Module):
    def __init__(self, backbone: nn.Module, neck: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        u: torch.Tensor,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: int | None = None,
    ):
        z = self.backbone(x, edge_index, edge_weight, edge_attr, batch, batch_size)
        fused_emb = self.neck(z, u, batch, batch_size, x)
        return self.head(fused_emb)


class TemporalGNN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        head: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        u: torch.Tensor,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: int | None = None,
        prev_h: OptTensor = None,
        prev_c: OptTensor = None,
    ):
        z = self.backbone(x, edge_index, edge_weight, edge_attr, batch, batch_size)
        fused_emb, h, c = self.neck(z, u, x, batch, batch_size, prev_h, prev_c)
        return self.head(fused_emb), h, c
