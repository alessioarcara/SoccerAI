from typing import Literal

NormalizationType = Literal["none", "batch", "layer", "instance", "graph"]

ReadoutType = Literal["mean", "sum", "max"]

AggregationType = Literal["mean", "max", "lstm"]

RNNType = Literal["gru", "lstm"]

ResidualSumMode = Literal["none", "every", "last"]

TemporalMode = Literal["node", "graph"]
