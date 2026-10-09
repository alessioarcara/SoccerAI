from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence

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
        temporal_chains: list[DynamicGraphTemporalSignal],
        num_features: int,
        num_global_features: int,
        feature_names: Sequence[str],
        transform: Callable | None = None,
    ):
        self.temporal_chains = temporal_chains
        self.num_features = num_features
        self.num_global_features = num_global_features
        self.feature_names = feature_names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.temporal_chains)

    def positive_weight(self) -> float | list[float]:
        """
        BCE weight balancing the positive and negative targets as the loss
        sees them with uniform frame weights: every chain weighs 1, split
        evenly over its frames. With chain-level targets this is #negative /
        #positive chains. Frames with unknown targets (-1) are left out.
        """
        targets = [np.concatenate(c.targets) for c in self.temporal_chains]
        if targets[0].shape[-1] == 1:
            return _balanced_weight(targets, column=0)
        # box decomposition: weights of the box, shot | box and shot | no box
        # heads, each over the frames it is trained on
        return [
            _balanced_weight(targets, column=0),
            _balanced_weight(targets, column=1, given_box=1.0),
            _balanced_weight(targets, column=1, given_box=0.0),
        ]

    def __getitem__(self, idx: int) -> DynamicGraphTemporalSignal:
        temporal_chain = self.temporal_chains[idx]

        if self.transform is not None:
            temporal_chain = self.transform(temporal_chain)

        return temporal_chain

    @staticmethod
    def from_worldcup_dataset(
        dataset: WorldCup2022Dataset,
        max_chain_len: int | None = None,
        shot_horizon: float | None = None,
        box_decomposition: bool = False,
    ) -> TemporalChainsDataset:
        """
        Group the frames of `dataset` by chain, in chronological order.

        With `max_chain_len` only the last frames of each chain are kept: the
        label is decided by how a chain ends, early frames of long chains are
        mostly padding for the rest of the batch, and the chain length itself
        differs between classes (positive chains lose their shot frame).

        Targets: with `shot_horizon = None` every frame carries the label of
        its chain. With a horizon (seconds) a frame is positive iff its chain
        shoots within `shot_horizon` seconds of it, so the target rises along
        a positive chain as the shot approaches and early frames, which
        cannot tell the classes apart, are not asked to. Every chain also
        stores its `chain_label` and the per-frame `time_to_shot` (+inf for
        negative chains), used by the chain-level and early-warning metrics.

        With `box_decomposition` (timeline datasets, which carry
        `time_to_box`) every frame gets two targets within the horizon,
        [the ball reaches the penalty area, the team shoots], for a model of
        P(shot) = P(box) P(shot | box) + (1 - P(box)) P(shot | no box). On
        timeline datasets the frames closer than the horizon to the end of the
        period have an unknown future: their targets are -1 (ignored).
        """
        if max_chain_len is not None and max_chain_len < 1:
            raise ValueError(f"max_chain_len must be >= 1 or None, got {max_chain_len}")
        if shot_horizon is not None and not shot_horizon > 0:
            raise ValueError(f"shot_horizon must be > 0 or None, got {shot_horizon}")
        if box_decomposition and shot_horizon is None:
            raise ValueError("box_decomposition needs a shot_horizon")

        tmp_transform = dataset.transform
        dataset.transform = None

        try:
            buckets = defaultdict(list)
            for data in dataset:
                chain_id = int(data.chain_id.item())
                buckets[chain_id].append(data)
        finally:
            dataset.transform = tmp_transform

        chains = []
        for chain_id, frames in buckets.items():
            ordered = sorted(frames, key=lambda f: int(f.event_index.item()))
            if max_chain_len is not None:
                ordered = ordered[-max_chain_len:]

            edge_indices = [f.edge_index.numpy() for f in ordered]
            node_features = [f.x.numpy() for f in ordered]
            global_features = [f.u.numpy() for f in ordered]
            chain_label = float(ordered[-1].y.item())
            time_to_shot = [
                np.full((1,), _time_to_shot(f, chain_label), dtype=np.float32)
                for f in ordered
            ]
            if shot_horizon is not None and np.isnan(time_to_shot).any():
                raise ValueError(
                    f"Chain {chain_id} has no time to shot: rebuild the dataset"
                )
            targets = (
                [f.y.numpy() for f in ordered]
                if shot_horizon is None
                else _horizon_targets(
                    ordered, time_to_shot, shot_horizon, box_decomposition
                )
            )
            # unweighted graphs (e.g. fully connected) get unit weights
            edge_weights = [
                f.edge_weight.numpy()
                if f.edge_weight is not None
                else np.ones(f.edge_index.shape[1], dtype=np.float32)
                for f in ordered
            ]
            jersey_numbers = [f.jersey_numbers.numpy() for f in ordered]

            chains.append(
                DynamicGraphTemporalSignal(
                    edge_indices=edge_indices,
                    edge_weights=edge_weights,
                    features=node_features,
                    targets=targets,
                    u=global_features,
                    jersey_numbers=jersey_numbers,
                    chain_label=[
                        np.full((1,), chain_label, dtype=np.float32) for _ in ordered
                    ],
                    time_to_shot=time_to_shot,
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
    def collate(batch: list[DynamicGraphTemporalSignal]):
        """
        Stack B chains into one `DynamicGraphTemporalSignalBatch`.

        At every time step the B graphs are laid out as one disjoint union
        (nodes of chain b occupy rows [b*N, (b+1)*N)), exactly like
        `torch_geometric.data.Batch.from_data_list`. The edge indices stored
        in each chain are local to the chain, so they must be shifted by b*N:
        `DynamicGraphTemporalSignalBatch` builds `Batch` objects from the raw
        arrays and performs no such offset itself. Shorter chains are padded
        at the end with masked frames.
        """
        T_max = max(c.snapshot_count for c in batch)
        num_nodes = batch[0].features[0].shape[0]

        batch_edge_indices = []
        batch_edge_weights = []
        batch_features = []
        batch_targets = []
        batch_u = []
        batch_jersey_numbers = []
        batch_chain_labels = []
        batch_time_to_shot = []
        batch_masks = []
        batches = []

        for b, c in enumerate(batch):
            T = c.snapshot_count
            pad_frames = T_max - T

            if c.features[0].shape[0] != num_nodes:
                raise ValueError("All graphs in a batch must have the same node count")

            ei, ew, x, y, u, jn, cl, tts = (
                pad_chain(c, pad_frames)
                if pad_frames
                else (
                    c.edge_indices,
                    c.edge_weights,
                    c.features,
                    c.targets,
                    c.u,
                    c.jersey_numbers,
                    c.chain_label,
                    c.time_to_shot,
                )
            )

            # local node ids -> position in the disjoint union of the time step
            ei = [e + b * num_nodes for e in ei]

            batch_edge_indices.append(ei)
            batch_edge_weights.append(ew)
            batch_features.append(x)
            batch_targets.append(y)
            batch_u.append(u)
            batch_jersey_numbers.append(jn)
            batch_chain_labels.append(cl)
            batch_time_to_shot.append(tts)
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

        # (B, T_max, 1) -> (T_max, B)
        batch_chain_labels_np = np.array(batch_chain_labels)[..., 0].T
        batch_time_to_shot_np = np.array(batch_time_to_shot)[..., 0].T

        batch_masks_np = np.array(batch_masks).T  # (T_max, B)

        # For each time step, build an array that maps every node to the index
        # of the graph (in `batch`) it belongs to. This is equivalent to the
        # `batch` vector used in PyG for graphs, but replicated over time.
        for _ in range(T_max):
            timestep_batch = np.concatenate(
                [np.full(num_nodes, i, dtype=np.int64) for i in range(len(batch))]
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
            chain_label=batch_chain_labels_np,
            time_to_shot=batch_time_to_shot_np,
        )


def pad_chain(
    c: DynamicGraphTemporalSignal, num_pad_frames: int
) -> tuple[list[np.ndarray], ...]:
    """
    Append `num_pad_frames` masked frames: zero features, target and chain
    label -1, unknown (NaN) time to shot and the edges of the last real frame
    with zero weight (so that no spurious self-loops are created once the
    node ids are offset per graph).
    """
    pad_ei = c.edge_indices[-1]
    pad_ew = np.zeros_like(c.edge_weights[-1])
    pad_x = np.zeros_like(c.features[0])
    pad_y = np.full_like(c.targets[0], -1)
    pad_u = np.zeros_like(c.u[0])
    pad_jn = np.zeros_like(c.jersey_numbers[0])
    pad_cl = np.full_like(c.chain_label[0], -1)
    pad_tts = np.full_like(c.time_to_shot[0], np.nan)

    pads = [pad_ei, pad_ew, pad_x, pad_y, pad_u, pad_jn, pad_cl, pad_tts]
    sequences = [
        c.edge_indices,
        c.edge_weights,
        c.features,
        c.targets,
        c.u,
        c.jersey_numbers,
        c.chain_label,
        c.time_to_shot,
    ]
    return tuple(
        list(seq) + [pad] * num_pad_frames
        for seq, pad in zip(sequences, pads, strict=True)
    )


def _horizon_targets(
    frames: list,
    time_to_shot: list[np.ndarray],
    horizon: float,
    box_decomposition: bool,
) -> list[np.ndarray]:
    """
    Per-frame targets within `horizon` seconds: [shot], or [box, shot] with
    `box_decomposition`; -1 when the period ends before the horizon.
    """
    targets = []
    for frame, tts in zip(frames, time_to_shot, strict=True):
        values = [float(tts[0] <= horizon)]
        if box_decomposition:
            time_to_box = getattr(frame, "time_to_box", None)
            if time_to_box is None:
                raise ValueError("box_decomposition needs a timeline dataset")
            values.insert(0, float(float(time_to_box) <= horizon))
        to_period_end = getattr(frame, "time_to_period_end", None)
        if to_period_end is not None and float(to_period_end) < horizon:
            values = [-1.0] * len(values)
        targets.append(np.asarray([values], dtype=np.float32))
    return targets


def _time_to_shot(frame, chain_label: float) -> float:
    """Seconds from `frame` to its chain's shot; NaN when unknown."""
    time_to_shot = getattr(frame, "time_to_shot", None)
    if time_to_shot is not None:
        return float(time_to_shot)
    return float("inf") if chain_label == 0 else float("nan")


def _balanced_weight(
    targets: list[np.ndarray], column: int, given_box: float | None = None
) -> float:
    """
    #negative / #positive targets in `column`, every chain weighing 1 split
    evenly over its known frames (optionally only those with box = given_box).
    """
    pos = neg = 0.0
    for chain_targets in targets:
        known = chain_targets[:, column] >= 0
        n_known = known.sum()
        if n_known == 0:
            continue
        if given_box is not None:
            known &= chain_targets[:, 0] == given_box
        values = chain_targets[known, column]
        pos += float((values == 1).sum()) / n_known
        neg += float((values == 0).sum()) / n_known
    return neg / max(pos, 1e-12)
