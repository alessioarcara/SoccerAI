from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.collections import LineCollection
from mplsoccer import Pitch
from mplsoccer.dimensions import center_scale_dims
from torch_geometric.data import Batch
from torchmetrics.functional.classification import binary_precision_recall_curve

from soccerai.training.utils import TopKStorage


class Metric(ABC):
    @abstractmethod
    def update(
        self, preds_probs: torch.Tensor, true_labels: torch.Tensor, batch: Batch
    ) -> None:
        pass

    @abstractmethod
    def compute(self) -> List[Tuple[str, float]]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def plot(self) -> Optional[Tuple[str, plt.Figure]]:
        pass


class BinaryConfusionMatrix(Metric):
    def __init__(self, thr: float = 0.5, beta: float = 1):
        self.thr = thr
        self.beta = beta
        self.reset()

    def update(
        self, preds_probs: torch.Tensor, true_labels: torch.Tensor, batch: Batch
    ) -> None:
        preds_labels_flat = (preds_probs >= self.thr).view(-1).long()
        true_labels_flat = true_labels.view(-1).long()

        for t, p in zip(true_labels_flat, preds_labels_flat):
            self.cm[t, p] += 1

    def compute(self) -> List[Tuple[str, float]]:
        tn, fp = self.cm[0, 0].item(), self.cm[0, 1].item()
        fn, tp = self.cm[1, 0].item(), self.cm[1, 1].item()

        total = tn + fp + fn + tp
        results: List[Tuple[str, float]] = []

        # Accuracy
        accuracy = (tp + tn) / total if total > 0 else 0.0
        results.append(("accuracy", accuracy))

        # Fbeta-score
        beta2 = self.beta**2
        denom = (1 + beta2) * tp + beta2 * fn + fp
        fbeta = ((1 + beta2) * tp / denom) if denom > 0 else 0.0
        results.append((f"f{self.beta}_score", fbeta))

        return results

    def reset(self) -> None:
        self.cm = torch.zeros((2, 2), dtype=torch.int64)

    def plot(self) -> Optional[Tuple[str, plt.Figure]]:
        cm_np = self.cm.cpu().numpy()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm_np,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            cbar=False,
            annot_kws={"fontsize": 14},
        )
        ax.set_xlabel("Predicted Label", fontsize=16)
        ax.set_ylabel("True Label", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=12)
        plt.tight_layout()
        return "confusion_matrix", fig


class BinaryPrecisionRecallCurve(Metric):
    def __init__(self):
        self.reset()

    def update(
        self, preds_probs: torch.Tensor, true_labels: torch.Tensor, batch: Batch
    ) -> None:
        self.all_preds_probs.append(preds_probs.detach().view(-1).cpu())
        self.all_true_labels.append(true_labels.detach().view(-1).cpu())

    def compute(self) -> List[Tuple[str, float]]:
        return []

    def reset(self):
        self.all_preds_probs = []
        self.all_true_labels = []

    def plot(self) -> Optional[Tuple[str, plt.Figure]]:
        all_preds_probs_flat = torch.cat(self.all_preds_probs)
        all_true_labels_flat = torch.cat(self.all_true_labels).long()
        p, r, thresholds = binary_precision_recall_curve(
            all_preds_probs_flat, all_true_labels_flat
        )
        points = np.stack([r, p], axis=1)
        segments_list = np.stack([points[:-1], points[1:]], axis=1).tolist()
        lc = LineCollection(
            segments_list,
            cmap="rainbow",
            norm=plt.Normalize(
                vmin=thresholds.min().item(),
                vmax=thresholds.max().item(),
            ),
            linewidth=2,
        )
        lc.set_array(thresholds)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax)
        cbar.ax.tick_params(labelsize=12)
        ax.set_xlabel("Recall", fontsize=16)
        ax.set_ylabel("Precision", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.grid(True)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        return "Precision-Recall Curve", fig


class PositiveFrameCollector(Metric):
    def __init__(self, thr: float = 0.5, max_samples: int = 12):
        self.thr = thr
        self.storage: TopKStorage[Batch] = TopKStorage(max_samples)

    def update(
        self, preds_probs: torch.Tensor, true_labels: torch.Tensor, batch: Batch
    ) -> None:
        probs_np = preds_probs.detach().cpu().numpy()
        labels_np = true_labels.detach().cpu().numpy()

        pos_indices = np.where((probs_np > self.thr) & (labels_np == 1))[0]

        for i in pos_indices:
            self.storage.add((probs_np[i], batch[i]))

    def compute(self) -> List[Tuple[str, float]]:
        return []

    def reset(self) -> None:
        self.storage.clear()

    def plot(self) -> Optional[Tuple[str, plt.Figure]]:
        dim = center_scale_dims(
            pitch_width=68, pitch_length=105, width=2, length=2, invert_y=False
        )
        pitch = Pitch(
            pitch_type=dim, pitch_color="grass", line_color="white", linewidth=2
        )

        fig, axs = pitch.grid(
            nrows=3,
            ncols=4,
            figheight=12,
            grid_height=0.95,
            grid_width=0.95,
            bottom=0.025,
            endnote_height=0,
            title_height=0,
        )

        axes = axs.flatten()
        entries = self.storage.get_all_entries()

        for ax, (score, graph) in zip(axes, entries):
            node_features = graph.x.detach().cpu().numpy()

            coords = node_features[:, :2]
            teams = node_features[:, 2].astype(int)
            has_ball = node_features[:, 3].astype(bool)

            face_colours = np.where(teams == 0, "red", "blue")
            edge_colours = np.where(has_ball, "white", face_colours)

            ax.scatter(*coords.T, c=face_colours, ec=edge_colours, s=150)
            ax.set_title(f"Prob: {float(score):.3f}", fontsize=14, pad=5)
            ax.axis("off")

        for ax in axes[len(entries) :]:
            ax.set_visible(False)

        return "positive_frames", fig
