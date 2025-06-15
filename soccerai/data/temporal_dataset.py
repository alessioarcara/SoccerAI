from __future__ import annotations

from collections import defaultdict
from typing import Callable, List, Optional

from torch.utils.data import Dataset
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal

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
    def collate(chains: List[DynamicGraphTemporalSignal]):
        #        T_max = max(chain.snapshot_count for c in chains)

        # for c in chains:
        #     T = c._set_snapshot_count
        # print(batch)
        pass
