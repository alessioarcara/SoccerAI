from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.typing import OptTensor

from soccerai.models.utils import build_mlp


class Identity(nn.Identity):
    def forward(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return input


class BatchNorm(pyg_nn.BatchNorm):
    def forward(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return super().forward(input)


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
        self.conv_layer = conv_layer
        self.norm = norm or Identity()
        self.drop = nn.Dropout(p_drop)
        self.mlp = build_mlp(dout, dout * 2, dout)

    def forward(
        self,
        x: torch.Tensor,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
        **conv_kwargs,
    ) -> torch.Tensor:
        x_proj = self.proj(x)

        x = F.relu(
            self.drop(
                self.norm(
                    self.conv_layer(x, **conv_kwargs),
                    batch,
                    batch_size,
                )
            )
        )

        return x + self.mlp(x_proj + x)
