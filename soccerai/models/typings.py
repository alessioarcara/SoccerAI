from typing import Literal

NormalizationType = Literal["none", "batch", "layer", "instance", "graph"]

ReadoutType = Literal["mean", "sum", "max"]

AggregationType = Literal["mean", "max", "lstm"]

RNNType = Literal["gru", "lstm"]

TemporalMode = Literal["node", "graph"]
