import numpy as np
import pytest
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


def make_timed_frame(chain_id, order, label, time_to_shot) -> Data:
    frame = make_frame(chain_id, order, label)
    frame.time_to_shot = torch.tensor(time_to_shot)
    return frame


def timed_dataset() -> DatasetStub:
    # positive chain 0 shoots 1, 5 and 12 s after its frames; chain 1 never
    return DatasetStub(
        [
            make_timed_frame(0, 1.0, 1.0, 12.0),
            make_timed_frame(0, 2.0, 1.0, 5.0),
            make_timed_frame(0, 3.0, 1.0, 1.0),
            make_timed_frame(1, 4.0, 0.0, float("inf")),
            make_timed_frame(1, 5.0, 0.0, float("inf")),
        ]
    )


def test_shot_horizon_turns_on_the_frames_close_to_the_shot():
    chains = TemporalChainsDataset.from_worldcup_dataset(
        timed_dataset(), shot_horizon=8.0
    )
    pos, neg = chains.temporal_chains
    assert [t.item() for t in pos.targets] == [0.0, 1.0, 1.0]
    assert [t.item() for t in neg.targets] == [0.0, 0.0]
    assert [c.item() for c in pos.chain_label] == [1.0, 1.0, 1.0]
    np.testing.assert_allclose([t.item() for t in pos.time_to_shot], [12, 5, 1])
    assert all(np.isinf(t.item()) for t in neg.time_to_shot)
    # chain weight split over the frames: 2/3 positive vs 1/3 + 1 negative
    assert chains.positive_weight() == pytest.approx((1 / 3 + 1) / (2 / 3))


def test_without_horizon_every_frame_has_the_chain_label():
    chains = TemporalChainsDataset.from_worldcup_dataset(timed_dataset())
    pos, _ = chains.temporal_chains
    assert [t.item() for t in pos.targets] == [1.0, 1.0, 1.0]
    assert chains.positive_weight() == 1.0  # one chain per class


def test_shot_horizon_needs_the_time_to_shot():
    ds = DatasetStub([make_frame(0, 1.0, 1.0)])
    with pytest.raises(ValueError, match="time to shot"):
        TemporalChainsDataset.from_worldcup_dataset(ds, shot_horizon=8.0)


def test_collate_pads_chain_labels_and_times():
    chains = TemporalChainsDataset.from_worldcup_dataset(
        timed_dataset(), shot_horizon=8.0
    )
    batch = TemporalChainsDataset.collate(chains.temporal_chains)
    assert batch.chain_label.tolist() == [[1, 0], [1, 0], [1, -1]]
    assert batch.time_to_shot[:, 0].tolist() == [12, 5, 1]
    assert np.isinf(batch.time_to_shot[:2, 1]).all()
    assert np.isnan(batch.time_to_shot[2, 1])


def test_box_decomposition_targets_and_censoring():
    frames = []
    for order, (tts, ttb, end) in enumerate(
        [(12.0, 0.0, 30.0), (5.0, 3.0, 20.0), (1.0, 0.0, 4.0)], start=1
    ):
        frame = make_timed_frame(0, float(order), 1.0, tts)
        frame.time_to_box = torch.tensor(ttb)
        frame.time_to_period_end = torch.tensor(end)
        frames.append(frame)
    chains = TemporalChainsDataset.from_worldcup_dataset(
        DatasetStub(frames), shot_horizon=8.0, box_decomposition=True
    )
    (chain,) = chains.temporal_chains
    # [box, shot] within 8 s; the last frame is 4 s from the end of the period
    assert [t.tolist() for t in chain.targets] == [
        [[1.0, 0.0]],
        [[1.0, 1.0]],
        [[-1.0, -1.0]],
    ]
    # box: 2 positives; shot | box: 1 positive, 1 negative; shot | no box: none
    weights = chains.positive_weight()
    assert weights[0] == 0.0 and weights[1] == pytest.approx(1.0)
