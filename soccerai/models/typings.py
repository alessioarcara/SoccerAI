from typing import Literal

NormalizationType = Literal["none", "batch", "layer", "instance", "graph"]

ReadoutType = Literal["mean", "sum"]

AggregationType = Literal["mean", "pool", "lstm"]
