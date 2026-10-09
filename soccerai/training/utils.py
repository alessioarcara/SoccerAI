from collections.abc import Sequence
from typing import Any, Generic, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D
from mplsoccer import Pitch
from torch_geometric.data import Batch, Data
from torch_geometric.seed import seed_everything
from torch_geometric_temporal import Discrete_Signal

T = TypeVar("T")

_PITCH_KWARGS: dict = {
    "pitch_type": "metricasports",
    "pitch_length": 105,
    "pitch_width": 68,
    "pitch_color": "grass",
    "line_color": "white",
    "linewidth": 2,
}
_POSSESSION_COLOUR = "#2563eb"
_OPPOSITION_COLOUR = "#ef4444"
_CARRIER_COLOUR = "#facc15"


def fix_random(seed: int):
    seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TopKStorage(Generic[T]):
    def __init__(self, k: int) -> None:
        self.k = k
        self._items: list[tuple[float, T]] = []

    def add(self, entry: tuple[float, T]) -> None:
        self._items.append(entry)
        self._items.sort(key=lambda x: x[0], reverse=True)
        if len(self._items) > self.k:
            self._items = self._items[: self.k]

    def clear(self) -> None:
        self._items.clear()

    def get_all_entries(self) -> list[tuple[float, T]]:
        return list(self._items)


def _prepare_frame_data(
    data: Data,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, ...]:
    node_features = data.x.detach().cpu().numpy()
    x = node_features[:, feature_names.index("x")]
    y = node_features[:, feature_names.index("y")]

    teams = node_features[:, feature_names.index("is_possession_team_1")] > 0.5
    has_ball = node_features[:, feature_names.index("is_ball_carrier_1")] > 0.5

    face_colours = np.where(teams, _POSSESSION_COLOUR, _OPPOSITION_COLOUR)
    edge_colours = np.where(has_ball, _CARRIER_COLOUR, "#1f2937")

    jersey_numbers = data.jersey_numbers.detach().cpu().numpy()

    return x, y, jersey_numbers, face_colours, edge_colours


def _attacks_right(data: Data, feature_names: Sequence[str]) -> bool | None:
    # Goal direction refers to the attacked goal for every player. It also
    # works when attack normalisation is disabled; do not assume a direction
    # if goal features are unavailable.
    if "goal_cos" not in feature_names:
        return None
    direction = data.x[:, feature_names.index("goal_cos")].detach().cpu().numpy()
    direction = direction[np.isfinite(direction) & (np.abs(direction) > 1e-6)]
    return bool(np.median(direction) > 0) if direction.size else None


def _draw_frame(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    jersey_numbers: np.ndarray,
    face_colours: np.ndarray,
    edge_colours: np.ndarray,
    title: str,
    attacks_right: bool | None = None,
) -> None:
    ax.scatter(
        x,
        y,
        c=face_colours,
        ec=edge_colours,
        s=180,
        linewidths=np.where(edge_colours == _CARRIER_COLOUR, 2.5, 0.7).tolist(),
        zorder=3,
    )

    for xi, yi, num in zip(x, y, jersey_numbers):
        ax.text(
            xi,
            yi,
            str(num),
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            color="white",
            zorder=4,
        )

    if attacks_right is not None:
        start, end = (0.72, 0.92) if attacks_right else (0.28, 0.08)
        ax.annotate(
            "",
            xy=(end, 0.04),
            xytext=(start, 0.04),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": _CARRIER_COLOUR, "lw": 1.6},
        )
        ax.text(
            (start + end) / 2,
            0.075,
            "Attack",
            transform=ax.transAxes,
            color=_CARRIER_COLOUR,
            fontsize=7,
            ha="center",
        )

    ax.set_title(title, fontsize=10, pad=6)
    ax.axis("off")
    # metricasports draws y downwards; the data (like the labeling view in
    # `soccerai.data.visualize`) has y upwards
    ax.invert_yaxis()


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


def plot_average_feature_importance(
    node_masks: list[np.ndarray],
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


def plot_pitch_frames_grid(
    entries: Sequence[tuple[float, Data]],
    feature_names: Sequence[str],
    grid_params: dict[str, int],
    *,
    item_name: str = "Frame",
    title: str | None = None,
) -> plt.Figure:
    if not entries:
        raise ValueError("At least one frame is required to draw a pitch grid")
    if any(grid_params[key] <= 0 for key in ("nrows", "ncols", "figheight")):
        raise ValueError("Pitch grid dimensions must be positive")
    ncols = min(grid_params["ncols"], len(entries))
    nrows = (len(entries) + ncols - 1) // ncols
    row_height = max(2.5, grid_params["figheight"] / grid_params["nrows"])
    header_height = 0.7 if title else 0.45
    fig_height = nrows * row_height + header_height
    pitch = Pitch(**_PITCH_KWARGS)
    fig, axs = plt.subplots(
        nrows,
        ncols,
        squeeze=False,
        figsize=(max(6, ncols * row_height * pitch.ax_aspect), fig_height),
        layout="constrained",
    )
    fig.set_layout_engine("constrained", rect=(0, 0, 1, 1 - header_height / fig_height))
    if title:
        fig.text(
            0.02, 1 - 0.1 / fig_height, title, fontsize=12, fontweight="bold", va="top"
        )
    legend = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markerfacecolor=colour,
            markeredgecolor=edge,
            markeredgewidth=width,
            markersize=8,
            label=label,
        )
        for colour, edge, width, label in [
            (_POSSESSION_COLOUR, "#1f2937", 0.7, "Possession team"),
            (_OPPOSITION_COLOUR, "#1f2937", 0.7, "Opponents"),
            ("#64748b", _CARRIER_COLOUR, 2.5, "Ball carrier"),
        ]
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 1 - (0.30 if title else 0.05) / fig_height),
    )
    axes = axs.flatten()

    for i, (ax, (score, data)) in enumerate(zip(axes, entries), 1):
        pitch.draw(ax=ax)
        x, y, jerseys, face_c, edge_c = _prepare_frame_data(
            data,
            feature_names,
        )
        _draw_frame(
            ax,
            x,
            y,
            jerseys,
            face_c,
            edge_c,
            title=f"{item_name} {i} · p(shot)={score:.2f}",
            attacks_right=_attacks_right(data, feature_names),
        )

    for ax in axes[len(entries) :]:
        ax.set_visible(False)

    return fig


def plot_chain_frames(
    snapshots: Sequence[Data],
    scores: Sequence[float],
    feature_names: Sequence[str],
) -> np.ndarray:
    pitch = Pitch(**_PITCH_KWARGS)
    figs = []

    for i, (score, snapshot) in enumerate(zip(scores, snapshots), 1):
        fig, ax = pitch.draw()

        x, y, jerseys, face_c, edge_c = _prepare_frame_data(
            snapshot,
            feature_names,
        )

        _draw_frame(
            ax,
            x,
            y,
            jerseys,
            face_c,
            edge_c,
            title=f"Frame {i} · p(shot)={score:.2f}",
            attacks_right=_attacks_right(snapshot, feature_names),
        )

        figs.append(fig_to_numpy(fig))
        plt.close(fig)

    return np.array(figs).transpose(0, 3, 1, 2)


def fig_to_numpy(
    fig: plt.Figure,
) -> np.ndarray:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img_np = np.asarray(buf)
    plt.close(fig)
    return img_np


def build_dummy_inputs(
    bs: int, feat_dim: int, glob_dim: int, device: str | torch.device
) -> dict[str, Any]:
    """
    Creates random tensors to feed `torch_geometric.nn.summary`.
    """
    num_nodes_total = 22 * bs
    num_edges_total = 11 * 22 * bs

    x = torch.rand((num_nodes_total, feat_dim), device=device)
    edge_index = torch.randint(
        0, num_nodes_total, (2, num_edges_total), dtype=torch.long, device=device
    )
    edge_attr = torch.rand((num_edges_total), device=device)
    u = torch.rand((bs, glob_dim), device=device)
    batch = torch.tensor([[i] * 22 for i in range(bs)], device=device).view(-1)
    return dict(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, batch=batch)


def extract_chain(batch: Discrete_Signal, chain_idx: int) -> list[Data]:
    """
    NOTE - PyTorch Geometric Temporal assembles its batches manually instead of
    via `Batch.from_data_list`, so helper methods such as `get_example()` or
    `to_data_list()` raise a RuntimeError when you try to pull out a single
    graph.  The helper below works around that limitation with a (somewhat
    hacky) low-level extraction of the i-th graph.
    """

    def _extract_graph(snapshot: Batch) -> Data:
        node_mask = snapshot.batch == chain_idx
        x = snapshot.x[node_mask]
        jersey_numbers = snapshot.jersey_numbers[node_mask]

        # nodes of graph `chain_idx` are contiguous: map their ids back to 0..N-1
        node_offset = int(node_mask.nonzero().min())
        edge_mask = node_mask[snapshot.edge_index[0]]
        edge_index = snapshot.edge_index[:, edge_mask] - node_offset
        edge_attr = (
            snapshot.edge_attr[edge_mask] if snapshot.edge_attr is not None else None
        )

        u = snapshot.u[chain_idx].unsqueeze(0)
        y = snapshot.y[chain_idx]

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            u=u,
            jersey_numbers=jersey_numbers,
            y=y,
        )

    return [_extract_graph(batch[t]) for t in range(batch.snapshot_count)]
