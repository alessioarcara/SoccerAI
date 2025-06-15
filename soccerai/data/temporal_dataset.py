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
    def __init__(
        self,
        temporal_chains: List[DynamicGraphTemporalSignal],
        transform: Optional[Callable] = None,
    ):
        self.temporal_chains = temporal_chains
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

            chains.append(
                DynamicGraphTemporalSignal(
                    edge_indices=edge_indices,
                    edge_weights=edge_weights,
                    features=node_features,
                    targets=targets,
                    u=global_features,
                )
            )

        return TemporalChainsDataset(temporal_chains=chains, transform=tmp_transform)

    @staticmethod
    def collate(batch: List[DynamicGraphTemporalSignal]):
        T_max = max(c.snapshot_count for c in batch)

        batch_edge_indices: List[np.ndarray] = []
        batch_edge_weights: List[np.ndarray] = []
        batch_features: List[np.ndarray] = []
        batch_targets: List[np.ndarray] = []
        for c in batch:
            T = c._set_snapshot_count

            pad_frames = T_max - T

            if pad_frames:
                padded_ei, padded_ew, padded_x, padded_y = pad_chain(c, pad_frames)
                batch_edge_indices += padded_ei
                batch_edge_weights += padded_ew
                batch_features += padded_x
                batch_targets += padded_y

        return DynamicGraphTemporalSignalBatch(
            edge_indices=batch_edge_indices,
            edge_weights=batch_edge_weights,
            features=batch_features,
            targets=batch_targets,
        )


def pad_chain(
    c: DynamicGraphTemporalSignal, num_pad_frames: int
) -> Tuple[Sequence[np.ndarray], ...]:
    pad_ei = np.zeros_like(c.edge_indices[0])
    pad_ew = np.zeros_like(c.edge_weights[0])
    pad_x = np.zeros_like(c.features[0])
    pad_y = np.zeros_like(c.targets[0])

    padded_ei = list(c.edge_indices)
    padded_ew = list(c.edge_weights)
    padded_x = list(c.features)
    padded_y = list(c.targets)

    for _ in range(num_pad_frames):
        padded_ei.append(pad_ei)
        padded_ew.append(pad_ew)
        padded_x.append(pad_x)
        padded_y.append(pad_y)

    return padded_ei, padded_ew, padded_x, padded_y
