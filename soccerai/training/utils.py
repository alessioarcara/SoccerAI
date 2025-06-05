from typing import Generic, List, Tuple, TypeVar

import numpy as np
import torch
from torch_geometric.seed import seed_everything
from torch_geometric_temporal.signal import (
    DynamicGraphTemporalSignal,
    DynamicGraphTemporalSignalBatch,
)

T = TypeVar("T")


def fix_random(seed: int):
    seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TopKStorage(Generic[T]):
    def __init__(self, k: int) -> None:
        self.k = k
        self._items: List[Tuple[float, T]] = []

    def add(self, entry: Tuple[float, T]) -> None:
        self._items.append(entry)
        self._items.sort(key=lambda x: x[0], reverse=True)
        if len(self._items) > self.k:
            self._items = self._items[: self.k]

    def clear(self) -> None:
        self._items.clear()

    def get_all_entries(self) -> List[Tuple[float, T]]:
        return list(self._items)


def temporal_collate_fn(
    batch: List[DynamicGraphTemporalSignal],
) -> Tuple[DynamicGraphTemporalSignalBatch, torch.Tensor]:
    max_len = max(seq.snapshot_count for seq in batch)
    batches = []
    for _ in range(max_len):
        timestep_batch = np.concatenate(
            [
                np.full(batch[0].features[0].shape[0], i, dtype=np.int64)
                for i in range(len(batch))
            ]
        )
        batches.append(timestep_batch)

    edge_indices = []
    edge_weights = []
    features = []
    targets = []
    masks = []
    for seq in batch:
        valid_len = seq.snapshot_count
        mask = np.array([1] * valid_len + [0] * (max_len - valid_len))
        padded_seq = pad_sequence_to_length(seq, max_len)
        edge_indices.append(np.array(padded_seq.edge_indices))
        edge_weights.append(np.array(padded_seq.edge_weights))
        features.append(np.array(padded_seq.features))
        targets.append(np.array(padded_seq.targets))
        masks.append(mask)

    edge_indices = np.array(edge_indices, dtype=np.int32).transpose(1, 0, 2, 3)
    edge_weights = np.array(edge_weights, dtype=np.float32).transpose(1, 0, 2, 3)
    features = np.array(features, dtype=np.float32).transpose(1, 0, 2, 3)
    targets = np.array(targets, dtype=np.float32).transpose(1, 0, 2, 3)
    masks = np.array(masks, dtype=np.bool).T

    return DynamicGraphTemporalSignalBatch(
        edge_indices=edge_indices,
        edge_weights=edge_weights,
        features=features,
        targets=targets,
        batches=batches,
        masks=masks,
    )


def pad_sequence_to_length(
    seq: DynamicGraphTemporalSignal, max_len: int
) -> DynamicGraphTemporalSignal:
    pad_len = max_len - seq.snapshot_count
    edge_indices_pad = np.zeros_like(seq.edge_indices[0])
    edge_weights_pad = np.zeros_like(seq.edge_weights[0])
    features_pad = np.zeros_like(seq.features[0])
    targets_pad = np.zeros_like(seq.targets[0])

    edge_indices_padded = seq.edge_indices + [edge_indices_pad] * pad_len
    edge_weights_padded = seq.edge_weights + [edge_weights_pad] * pad_len
    features_padded = seq.features + [features_pad] * pad_len
    targets_padded = seq.targets + [targets_pad] * pad_len

    return DynamicGraphTemporalSignal(
        edge_indices=edge_indices_padded,
        edge_weights=edge_weights_padded,
        features=features_padded,
        targets=targets_padded,
    )
