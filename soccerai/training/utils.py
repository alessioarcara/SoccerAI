from typing import Any, Dict, Generic, List, Sequence, Tuple, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from torch_geometric.data import Batch, Data
from torch_geometric.seed import seed_everything
from torch_geometric_temporal import Discrete_Signal

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


def plot_player_feature_importance(
    node_mask: np.ndarray,
    jersey_numbers: np.ndarray,
    feature_names: Sequence[str],
    positive_type: str,
    frame_idx: int,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        node_mask,
        ax=ax,
        cmap="coolwarm",
        cbar=False,
        xticklabels=feature_names,
        yticklabels=[str(int(num)) for num in jersey_numbers],
    )
    ax.set_title(f"{positive_type}_Frame_{frame_idx}", fontsize=14)
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel("Player Jersey Number", fontsize=10, labelpad=10)
    plt.tight_layout()
    return fig


def fig_to_numpy(
    fig: plt.Figure,
) -> np.ndarray:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img_np = np.asarray(buf)
    plt.close(fig)
    return img_np


def plot_average_feature_importance(
    node_masks: List[np.ndarray],
    feature_names: Sequence[str],
    num_frames: int,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    average_feature_importance = np.stack(node_masks).mean(axis=1)
    ax.boxplot(
        average_feature_importance,
        vert=False,
        flierprops=dict(marker="o", markersize=4, alpha=0.6),
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", edgecolor="gray"),
        medianprops=dict(color="orange", linewidth=2),
        tick_labels=feature_names,
    )
    ax.set_title(
        f"Feature importance over {num_frames} frames",
        fontsize=14,
    )
    plt.tight_layout()
    return fig


def build_dummy_inputs(
    bs: int, feat_dim: int, glob_dim: int, device: torch.device
) -> Dict[str, Any]:
    """
    Creates random tensors to feed `torch_geometric.nn.summary`.
    """
    num_nodes_total = 22 * bs
    num_edges_total = 11 * 22 * bs

    x = torch.rand((num_nodes_total, feat_dim), device=device)
    edge_index = torch.randint(
        0, num_nodes_total, (2, num_edges_total), dtype=torch.long, device=device
    )
    u = torch.rand((bs, glob_dim), device=device)
    batch = torch.tensor([[i] * 22 for i in range(bs)], device=device).view(-1)
    return dict(x=x, edge_index=edge_index, u=u, batch=batch)


def extract_chain(batch: Discrete_Signal, chain_idx: int) -> List[Data]:
    """
    NOTE - PyTorch Geometric Temporal assembles its batches manually instead of
    via `Batch.from_data_list`, so helper methods such as `get_example()` or
    `to_data_list()` raise a RuntimeError when you try to pull out a single
    graph.  The helper below works around that limitation with a (somewhat
    hacky) low-level extraction of the i-th graph.
    """

    def _extract_graph(snapshot: Batch) -> Data:
        node_mask = batch == chain_idx
        x = snapshot.x[node_mask]

        start_edge_idx = 22 * 11 * chain_idx
        end_edge_idx = 22 * 11 * (chain_idx + 1)
        edge_index = snapshot.edge_index[:, start_edge_idx:end_edge_idx]
        edge_attr = snapshot.edge_attr[start_edge_idx:end_edge_idx]

        u = snapshot.u[chain_idx]
        y = snapshot.y[chain_idx]

        return Data(x=x, edge_index=edge_index, edge_weight=edge_attr, u=u, y=y)

    return [_extract_graph(batch[t]) for t in range(batch.snapshot_count)]
