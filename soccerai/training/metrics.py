from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.collections import LineCollection
from mplsoccer import Pitch
from torch_geometric.data import Batch
from torchmetrics.functional.classification import binary_precision_recall_curve

from soccerai.training.trainer_config import Config, MetricsConfig
from soccerai.training.utils import TopKStorage

T = TypeVar("T")


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
    def __init__(self, cfg: MetricsConfig):
        self.cfg = cfg
        self.reset()

    def update(
        self, preds_probs: torch.Tensor, true_labels: torch.Tensor, batch: Batch
    ) -> None:
        preds_labels_flat = (preds_probs >= self.cfg.thr).view(-1).long()
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
        beta2 = self.cfg.fbeta**2
        denom = (1 + beta2) * tp + beta2 * fn + fp
        fbeta = ((1 + beta2) * tp / denom) if denom > 0 else 0.0
        results.append((f"f{self.cfg.fbeta}_score", fbeta))

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


class FrameCollector(Metric):
    def __init__(
        self,
        target_label: int,
        cfg: Config,
        feature_names: List[str],
    ):
        self.cfg = cfg
        self.feature_names = feature_names
        self.target_label = target_label
        self.storage: TopKStorage = TopKStorage(self.cfg.collector.n_frames)

    def update(
        self,
        preds_probs: torch.Tensor,
        true_labels: torch.Tensor,
        batch: Batch,
    ) -> None:
        probs_np = preds_probs.detach().cpu().numpy()
        labels_np = true_labels.detach().cpu().numpy()

        indices = np.where(
            (probs_np >= self.cfg.metrics.thr) & (labels_np == self.target_label)
        )[0]

        for i in indices:
            self.storage.add((float(probs_np[i]), batch[i]))

    def compute(self) -> List[Tuple[str, float]]:
        return []

    def reset(self) -> None:
        self.storage.clear()

    def plot(self) -> Optional[Tuple[str, plt.Figure]]:
        entries = self.storage.get_all_entries()

        if not entries:
            return None

        x_col_idx = self.feature_names.index("x")
        possession_team_col_idx = self.feature_names.index("is_possession_team_1")
        ball_carrier_col_idx = self.feature_names.index("is_ball_carrier_1")

        pitch = Pitch(
            pitch_type="metricasports",
            pitch_length=105,
            pitch_width=68,
            pitch_color="grass",
            line_color="white",
            linewidth=2,
        )
        fig, axs = pitch.grid(
            nrows=self.cfg.collector.n_rows,
            ncols=self.cfg.collector.n_cols,
            figheight=self.cfg.collector.fig_height,
            grid_height=0.95,
            grid_width=0.95,
            bottom=0.025,
            endnote_height=0,
            title_height=0,
        )
        axes = axs.flatten()

        for i, (ax, (score, data)) in enumerate(zip(axes, entries), start=1):
            node_features = data.x.detach().cpu().numpy()
            jersey_numbers = data.jersey_numbers.detach().cpu().numpy()

            xy_coords = node_features[:, x_col_idx : x_col_idx + 2]
            teams = node_features[:, possession_team_col_idx].astype(int)
            has_ball = node_features[:, ball_carrier_col_idx].astype(bool)

            face_colours = np.where(teams == 0, "red", "blue")
            edge_colours = np.where(has_ball, "white", face_colours)

            ax.scatter(*xy_coords.T, c=face_colours, ec=edge_colours, s=200)

            for (xi, yi), jersey_num in zip(xy_coords, jersey_numbers):
                ax.text(
                    xi,
                    yi,
                    str(jersey_num),
                    fontsize=8,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    color="white",
                )

            ax.set_title(
                f"Frame {i} — Conf. {score:.2f}",
                fontsize=12,
                pad=4,
            )
            ax.axis("off")
            ax.invert_yaxis()

        for ax in axes[len(entries) :]:
            ax.set_visible(False)

        return f"{'tp' if self.target_label == 1 else 'fp'}_frames", fig
