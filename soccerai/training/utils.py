import os
import tempfile
from typing import Generic, List, Tuple, TypeVar

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch_geometric.seed import seed_everything

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


def make_heatmap_video_opencv(
    node_masks: List[np.ndarray],
    feature_names: List[str],
    label_name: str,
    fps: int = 1,
) -> str:
    tmpdir = tempfile.mkdtemp(prefix="explainer_vid_")
    frame_paths: List[str] = []

    for i, mask in enumerate(node_masks):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            mask,
            ax=ax,
            cmap="coolwarm",
            cbar=True,
            xticklabels=feature_names,
            yticklabels=False,
        )
        ax.set_title(f"{label_name} Frame {i + 1}", fontsize=12)
        ax.set_xlabel("Features")
        plt.tight_layout()

        path = os.path.join(tmpdir, f"frame_{i:03d}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        frame_paths.append(path)

    first = cv2.imread(frame_paths[0])
    h, w, _ = first.shape

    fourcc = cv2.VideoWriter_fourcc(*"VP90")
    video_path = os.path.join(tmpdir, f"{label_name}.webm")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    for fp in frame_paths:
        img = cv2.imread(fp)
        writer.write(img)
    writer.release()

    return video_path


def plot_feature_importance_distribution(
    node_masks: List[np.ndarray],
    feature_names: List[str],
    label_name: str,
    actual_n: int,
) -> Tuple[str, plt.Figure]:
    per_frame_importance = np.stack(node_masks).mean(axis=1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(
        per_frame_importance,
        vert=False,
        flierprops=dict(marker="o", markersize=4, alpha=0.6),
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", edgecolor="gray"),
        medianprops=dict(color="orange", linewidth=2),
    )
    ax.set_yticks(np.arange(1, len(feature_names) + 1))
    ax.set_yticklabels(feature_names)
    ax.set_xlabel("Feature Importance")
    ax.set_title(
        f"Feature importance distribution ({label_name}, over {actual_n} frames)",
        fontsize=14,
    )
    plt.tight_layout()

    key = f"explain/feature_importance_box_{label_name.replace(' ', '_').lower()}"
    return key, fig
