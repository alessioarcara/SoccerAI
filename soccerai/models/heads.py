import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn


class GraphClassificationHead(nn.Module):
    """MLP halving the width at every hidden layer, down to `dout` logits."""

    def __init__(self, din: int, n_layers: int = 2, drop: float = 0.3, dout: int = 1):
        super().__init__()
        layers: list[nn.Module] = []

        for _ in range(n_layers):
            hidden = din // 2
            layers.append(pyg_nn.Linear(din, hidden))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=drop))
            din = hidden

        layers.append(pyg_nn.Linear(din, dout))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.mlp(x)
