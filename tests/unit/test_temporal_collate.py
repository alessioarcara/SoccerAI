import numpy as np
import torch
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal

from soccerai.data.temporal_dataset import TemporalChainsDataset
from soccerai.training.utils import extract_chain

N_NODES = 6
N_FEAT = 4


def bipartite_edges(n_nodes: int) -> np.ndarray:
    half = n_nodes // 2
    src, dst = [], []
    for i in range(half):
        for j in range(half, n_nodes):
            src += [i, j]
            dst += [j, i]
    return np.array([src, dst], dtype=np.int64)


def make_chain(n_frames: int, label: float, seed: int) -> DynamicGraphTemporalSignal:
    rng = np.random.default_rng(seed)
    ei = bipartite_edges(N_NODES)
    return DynamicGraphTemporalSignal(
        edge_indices=[ei.copy() for _ in range(n_frames)],
        edge_weights=[
            rng.random((ei.shape[1],)).astype(np.float32) for _ in range(n_frames)
        ],
        features=[
            rng.random((N_NODES, N_FEAT)).astype(np.float32) for _ in range(n_frames)
        ],
        targets=[np.full((1, 1), label, dtype=np.float32) for _ in range(n_frames)],
        u=[rng.random((1, 3)).astype(np.float32) for _ in range(n_frames)],
        jersey_numbers=[np.arange(N_NODES, dtype=np.int64) for _ in range(n_frames)],
    )


def test_collate_offsets_node_ids_per_graph():
    chains = [make_chain(3, 1.0, 0), make_chain(1, 0.0, 1), make_chain(2, 1.0, 2)]
    batch = TemporalChainsDataset.collate(chains)

    assert batch.snapshot_count == 3
    for t, snapshot in enumerate(batch):
        assert snapshot.x.shape == (3 * N_NODES, N_FEAT)
        assert snapshot.num_graphs == 3
        assert torch.equal(snapshot.batch, torch.arange(3).repeat_interleave(N_NODES))
        for b, chain in enumerate(chains):
            lo, hi = b * N_NODES, (b + 1) * N_NODES
            edge_mask = (snapshot.edge_index[0] >= lo) & (snapshot.edge_index[0] < hi)
            assert edge_mask.sum() == chain.edge_indices[0].shape[1]
            assert bool(
                (
                    (snapshot.edge_index[1][edge_mask] >= lo)
                    & (snapshot.edge_index[1][edge_mask] < hi)
                ).all()
            )

            valid = t < chain.snapshot_count
            assert bool(snapshot.masks[b]) is valid
            if valid:
                np.testing.assert_allclose(snapshot.x[lo:hi].numpy(), chain.features[t])
                np.testing.assert_allclose(
                    snapshot.edge_attr[edge_mask].numpy(), chain.edge_weights[t]
                )
                assert snapshot.y[b].item() == chain.targets[t].item()
            else:
                assert snapshot.y[b].item() == -1.0
                assert float(snapshot.edge_attr[edge_mask].abs().sum()) == 0.0
                assert float(snapshot.x[lo:hi].abs().sum()) == 0.0


def test_extract_chain_recovers_local_graphs():
    chains = [make_chain(2, 1.0, 0), make_chain(3, 0.0, 1)]
    batch = TemporalChainsDataset.collate(chains)

    for b, chain in enumerate(chains):
        graphs = extract_chain(batch[: chain.snapshot_count], b)
        assert len(graphs) == chain.snapshot_count
        for t, g in enumerate(graphs):
            np.testing.assert_allclose(g.x.numpy(), chain.features[t])
            np.testing.assert_array_equal(g.edge_index.numpy(), chain.edge_indices[t])
            np.testing.assert_allclose(g.edge_attr.numpy(), chain.edge_weights[t])
            assert g.edge_index.max() < N_NODES


def test_collate_single_chain_is_unchanged():
    chain = make_chain(2, 1.0, 5)
    batch = TemporalChainsDataset.collate([chain])
    snapshot = batch[0]
    np.testing.assert_array_equal(snapshot.edge_index.numpy(), chain.edge_indices[0])
    np.testing.assert_allclose(snapshot.x.numpy(), chain.features[0])
