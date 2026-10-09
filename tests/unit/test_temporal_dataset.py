import numpy as np
import torch
from torch_geometric.data import Data

from soccerai.data.temporal_dataset import TemporalChainsDataset

N_NODES, N_FEAT = 4, 3


class DatasetStub(list):
    """Mimics the parts of WorldCup2022Dataset used to build chains."""

    num_features = N_FEAT
    num_global_features = 2
    feature_names = ["a", "b", "c"]
    transform = None


def make_frame(chain_id: int, order: float, label: float) -> Data:
    return Data(
        x=torch.full((N_NODES, N_FEAT), order),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_weight=torch.ones(2),
        u=torch.tensor([[order, 0.0]]),
        y=torch.tensor([[label]]),
        chain_id=torch.tensor(chain_id),
        event_index=torch.tensor(int(order)),
        jersey_numbers=torch.arange(N_NODES),
    )


def test_chains_are_ordered_and_truncated_to_their_last_frames():
    ds = DatasetStub(
        [
            make_frame(0, 3.0, 1.0),
            make_frame(0, 1.0, 1.0),
            make_frame(1, 5.0, 0.0),
            make_frame(0, 2.0, 1.0),
            make_frame(0, 4.0, 1.0),
        ]
    )
    full = TemporalChainsDataset.from_worldcup_dataset(ds)
    assert [c.snapshot_count for c in full.temporal_chains] == [4, 1]
    np.testing.assert_allclose([u[0, 0] for u in full[0].u], [1.0, 2.0, 3.0, 4.0])

    short = TemporalChainsDataset.from_worldcup_dataset(ds, max_chain_len=2)
    assert [c.snapshot_count for c in short.temporal_chains] == [2, 1]
    np.testing.assert_allclose([u[0, 0] for u in short[0].u], [3.0, 4.0])
    assert ds.transform is None  # dataset transform restored
