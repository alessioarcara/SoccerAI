from typing import Literal

NormalizationType = Literal["none", "batch", "layer", "instance", "graph"]

ReadoutType = Literal["mean", "sum"]

BackboneType = Literal["gcn", "gcn2"]

NeckType = Literal["gru", "lstm"]
