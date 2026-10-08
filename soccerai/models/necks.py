import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter

from soccerai.models.typings import ReadoutType, RNNType
from soccerai.training.trainer_config import NeckConfig

READOUT_AGGREGATIONS: dict[ReadoutType, type[pyg_nn.Aggregation]] = {
    "sum": pyg_nn.SumAggregation,
    "mean": pyg_nn.MeanAggregation,
    "max": pyg_nn.MaxAggregation,
}

RNN_CELLS: dict[RNNType, type[nn.Module]] = {"gru": nn.GRUCell, "lstm": nn.LSTMCell}


class GraphGlobalFusion(nn.Module):
    """
    Fuse graph-level and global feature vectors into one concatenated vector:
    1. Readout over nodes -> graph embedding
    2. Linear projection + ReLU -> global embedding
    3. Concatenate [graph || global], or [graph || carrier || global] when
       `carrier_idx` is given: the embedding of the ball carrier, read from
       the node flagged in column `carrier_idx` of the raw node features
       (zeros for a graph without carrier, e.g. a padded frame).
    """

    def __init__(self, glob_din: int, cfg: NeckConfig, carrier_idx: int | None = None):
        super().__init__()
        self.readout = READOUT_AGGREGATIONS[cfg.readout]()
        self.global_proj = pyg_nn.Linear(glob_din, cfg.glob_dout)
        self.carrier_idx = carrier_idx

    def forward(
        self,
        z: torch.Tensor,
        u: torch.Tensor,
        batch: torch.Tensor,
        batch_size: int,
        x: OptTensor = None,
    ) -> torch.Tensor:
        z_cat = torch.cat(z, dim=-1) if isinstance(z, list) else z

        embs = [self.readout(x=z_cat, index=batch, dim_size=batch_size)]
        if self.carrier_idx is not None:
            assert x is not None, "the carrier readout needs the node features"
            is_carrier = x[:, self.carrier_idx] > 0.5
            embs.append(
                scatter(
                    z_cat[is_carrier],
                    batch[is_carrier],
                    dim=0,
                    dim_size=batch_size,
                    reduce="mean",
                )
            )
        embs.append(F.relu(self.global_proj(u), inplace=True))
        return torch.cat(embs, dim=-1)


class RecurrentCell(nn.Module):
    """GRU/LSTM cell with a uniform `(x, h, c) -> (h, c)` interface."""

    def __init__(self, rnn_type: RNNType, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = RNN_CELLS[rnn_type](input_size=input_size, hidden_size=hidden_size)

    def forward(
        self, x: torch.Tensor, prev_h: OptTensor, prev_c: OptTensor
    ) -> tuple[torch.Tensor, OptTensor]:
        if isinstance(self.cell, nn.LSTMCell):
            state = None if prev_h is None or prev_c is None else (prev_h, prev_c)
            h, c = self.cell(x, state)
            return h, c
        return self.cell(x, prev_h), None


class TemporalFusion(nn.Module):
    """
    Apply temporal and fusion operations on graph and global features.

    Modes:
    - "node":
        1) Recurrent cell over every node embedding (shared weights, state per
           node): nodes keep their position across the frames of a chain.
        2) Fuse the readout of the node *states* with the global features.
    - "graph":
        1) Fuse graph and global features.
        2) Recurrent cell over the fused vectors.

    In both modes the vector handed to the head depends on the recurrent
    state, i.e. on the previous frames of the chain.
    """

    def __init__(
        self,
        backbone_dout: int,
        node_dim: int,
        glob_din: int,
        cfg: NeckConfig,
        carrier_idx: int | None = None,
    ):
        super().__init__()
        self.mode = cfg.mode
        self.fusion = GraphGlobalFusion(glob_din, cfg, carrier_idx)

        self.raw_features_proj: nn.Module = nn.Identity()
        if self.mode == "node":
            if cfg.raw_features_proj:
                self.raw_features_proj = nn.Sequential(
                    pyg_nn.Linear(node_dim, cfg.proj_dout), nn.ReLU()
                )
                rnn_din = backbone_dout + cfg.proj_dout
            else:
                rnn_din = backbone_dout + node_dim
            self.norm = nn.LayerNorm(rnn_din)
            self.rnn = RecurrentCell(cfg.rnn_type, rnn_din, cfg.rnn_dout)

        elif self.mode == "graph":
            # the sum readout of 22 nodes and the global projection live on
            # very different scales: normalise the fused vector before the RNN
            rnn_din = cfg.rnn_din
            if carrier_idx is not None:
                rnn_din += backbone_dout  # carrier embedding
            self.norm = nn.LayerNorm(rnn_din)
            self.rnn = RecurrentCell(cfg.rnn_type, rnn_din, cfg.rnn_dout)

        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def forward(
        self,
        z: torch.Tensor,
        u: torch.Tensor,
        x: torch.Tensor,
        batch: OptTensor = None,
        batch_size: int | None = None,
        prev_h: OptTensor = None,
        prev_c: OptTensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, OptTensor]:
        if self.mode == "node":
            z_nodes = torch.cat(z, dim=-1) if isinstance(z, list) else z
            rnn_input = self.norm(torch.cat([z_nodes, self.raw_features_proj(x)], -1))
            h, c = self.rnn(rnn_input, prev_h, prev_c)
            fused = self.fusion(h, u, batch, batch_size, x)
            return fused, h, c

        fused = self.norm(self.fusion(z, u, batch, batch_size, x))
        h, c = self.rnn(fused, prev_h, prev_c)
        return h, h, c
