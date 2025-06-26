import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

from soccerai.training.trainer_config import HeadConfig


class GraphClassificationHead(nn.Module):
    def __init__(self, din: int, cfg: HeadConfig):
        super().__init__()
        layers = []

        for _ in range(cfg.n_layers):
            dout = din // 2
            layers.append(pyg_nn.Linear(din, dout))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=cfg.drop))
            din = dout

        layers.append(pyg_nn.Linear(din, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.mlp(x)
