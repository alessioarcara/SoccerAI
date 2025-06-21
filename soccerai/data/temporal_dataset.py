from __future__ import annotations

from collections import defaultdict
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset
from torch_geometric_temporal.signal import (
    DynamicGraphTemporalSignal,
    DynamicGraphTemporalSignalBatch,
)

from soccerai.data.dataset import WorldCup2022Dataset


class TemporalChainsDataset(Dataset):
    """
    Groups frames that belong to the same possession chain into a single
    temporal example
    """

    def __init__(
        self,
        temporal_chains: List[DynamicGraphTemporalSignal],
        num_features: int,
        num_global_features: int,
        feature_names: Sequence[str],
        transform: Optional[Callable] = None,
    ):
        self.temporal_chains = temporal_chains
        self.num_features = num_features
        self.num_global_features = num_global_features
        self.feature_names = feature_names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.temporal_chains)

    def __getitem__(self, idx: int) -> DynamicGraphTemporalSignal:
        temporal_chain = self.temporal_chains[idx]

        if self.transform is not None:
            temporal_chain = self.transform(temporal_chain)

        return temporal_chain

    @staticmethod
    def from_worldcup_dataset(dataset: WorldCup2022Dataset) -> TemporalChainsDataset:
        tmp_transform = dataset.transform
        dataset.transform = None

        buckets = defaultdict(list)
        for data in dataset:
            chain_id = int(data.chain_id.item())
            buckets[chain_id].append(data)

        chains = []
        for chain_id, frames in buckets.items():
            ordered = sorted(frames, key=lambda f: float(f.frame_time.item()))

            edge_indices = [f.edge_index.numpy() for f in ordered]
            node_features = [f.x.numpy() for f in ordered]
            global_features = [f.u.numpy() for f in ordered]
            targets = [f.y.numpy() for f in ordered]
            edge_weights = [f.edge_weight.numpy() for f in ordered]
            jersey_numbers = [f.jersey_numbers.numpy() for f in ordered]

            chains.append(
                DynamicGraphTemporalSignal(
                    edge_indices=edge_indices,
                    edge_weights=edge_weights,
                    features=node_features,
                    targets=targets,
                    u=global_features,
                    jersey_numbers=jersey_numbers,
                )
            )

        return TemporalChainsDataset(
            chains,
            dataset.num_features,
            dataset.num_global_features,
            dataset.feature_names,
            tmp_transform,
        )

    @staticmethod
    def collate(batch: List[DynamicGraphTemporalSignal]):
        T_max = max(c.snapshot_count for c in batch)

        batch_edge_indices = []
        batch_edge_weights = []
        batch_features = []
        batch_targets = []
        batch_u = []
        batch_jersey_numbers = []
        batch_masks = []
        batches = []

        for c in batch:
            T = c.snapshot_count
            pad_frames = T_max - T

            ei, ew, x, y, u, jn = (
                pad_chain(c, pad_frames)
                if pad_frames
                else (
                    c.edge_indices,
                    c.edge_weights,
                    c.features,
                    c.targets,
                    c.u,
                    c.jersey_numbers,
                )
            )

            batch_edge_indices.append(ei)
            batch_edge_weights.append(ew)
            batch_features.append(x)
            batch_targets.append(y)
            batch_u.append(u)
            batch_jersey_numbers.append(jn)
            batch_masks.append(np.array([1] * T + [0] * pad_frames))

        arr_ei = np.array(batch_edge_indices)  # (B, T_max, 2, E)
        arr_ei = arr_ei.transpose(1, 2, 0, 3)  # (T_max, 2, B, E)
        batch_edge_indices_np = arr_ei.reshape(T_max, 2, -1)

        arr_ew = np.array(batch_edge_weights)  # (B, T_max, E)
        arr_ew = arr_ew.transpose(1, 0, 2)  # (T_max, B, E)
        batch_edge_weights_np = arr_ew.reshape(T_max, -1)

        arr_x = np.array(batch_features)  # (B, T_max, N, Node_dim)
        arr_x = arr_x.transpose(1, 0, 2, 3)  # (T_max, B, N, Node_dim)
        batch_features_np = arr_x.reshape(T_max, -1, arr_x.shape[-1])

        arr_y = np.array(batch_targets)  # (B, T_max, 1, 1)
        arr_y = arr_y.transpose(1, 0, 2, 3)  # (T_max, B, 1, 1)
        batch_targets_np = np.squeeze(arr_y, axis=2)  # (T_max, B, 1)

        arr_u = np.array(batch_u)  # (B, T_max, 1, Glob_dim)
        arr_u = arr_u.transpose(1, 0, 2, 3)  # (T_max, B, 1, Glob_dim)
        batch_u_np = np.squeeze(arr_u, axis=2)  # (T_max, B, Glob_dim)

        arr_jn = np.array(batch_jersey_numbers)  # (B, T_max, N)
        arr_jn = arr_jn.transpose(1, 0, 2)  # (T_max, B, N)
        batch_jn_np = arr_jn.reshape(T_max, -1)

        batch_masks_np = np.array(batch_masks).T  # (T_max, B)

        # For each time step, build an array that maps every node to the index
        # of the graph (in `batch`) it belongs to. This is equivalent to the
        # `batch` vector used in PyG for graphs, but replicated over time.
        for _ in range(T_max):
            timestep_batch = np.concatenate(
                [
                    np.full(batch[0].features[0].shape[0], i, dtype=np.int64)
                    for i in range(len(batch))
                ]
            )
            batches.append(timestep_batch)

        return DynamicGraphTemporalSignalBatch(
            edge_indices=batch_edge_indices_np,
            edge_weights=batch_edge_weights_np,
            features=batch_features_np,
            targets=batch_targets_np,
            batches=batches,
            masks=batch_masks_np,
            u=batch_u_np,
            jersey_numbers=batch_jn_np,
        )


def pad_chain(
    c: DynamicGraphTemporalSignal, num_pad_frames: int
) -> Tuple[List[np.ndarray], ...]:
    pad_ei = np.zeros_like(c.edge_indices[0])
    pad_ew = np.zeros_like(c.edge_weights[0])
    pad_x = np.zeros_like(c.features[0])
    pad_y = np.full_like(c.targets[0], -1)
    pad_u = np.zeros_like(c.u[0])
    pad_jn = np.zeros_like(c.jersey_numbers[0])

    padded_ei = list(c.edge_indices)
    padded_ew = list(c.edge_weights)
    padded_x = list(c.features)
    padded_y = list(c.targets)
    padded_u = list(c.u)
    padded_jn = list(c.jersey_numbers)

    for _ in range(num_pad_frames):
        padded_ei.append(pad_ei)
        padded_ew.append(pad_ew)
        padded_x.append(pad_x)
        padded_y.append(pad_y)
        padded_u.append(pad_u)
        padded_jn.append(pad_jn)

    return padded_ei, padded_ew, padded_x, padded_y, padded_u, padded_jn
