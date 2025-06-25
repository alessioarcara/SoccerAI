from typing import Optional

import torch.nn as nn
import torch_geometric.nn as pyg_nn


def get_readout(readout: str) -> pyg_nn.Aggregation:
    match readout:
        case "sum":
            return pyg_nn.SumAggregation()
        case "mean":
            return pyg_nn.MeanAggregation()


def get_norm(norm: str) -> Optional[nn.Module]:
    match norm:
        case "layer":
            return pyg_nn.LayerNorm
        case "instance":
            return pyg_nn.InstanceNorm
        case "graph":
            return pyg_nn.GraphNorm
        case _:
            return None
