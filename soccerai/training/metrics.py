from abc import ABC, abstractmethod
from typing import Generic, List, Optional, Sequence, Tuple, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.collections import LineCollection
from torch_geometric.data import Batch, Data
from torch_geometric_temporal.signal import Discrete_Signal
from torchmetrics.functional.classification import binary_precision_recall_curve

from soccerai.training.trainer_config import Config, MetricsConfig
from soccerai.training.utils import TopKStorage, extract_chain, plot_pitch_frames_grid

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
    def plot(self) -> List[Tuple[str, plt.Figure]]:
        pass


class BinaryConfusionMatrix(Metric):
    def __init__(self, cfg: MetricsConfig, ignore_value: Optional[int] = None):
        self.cfg = cfg
        self.ignore_value = ignore_value
        self.reset()

    def update(
        self, preds_probs: torch.Tensor, true_labels: torch.Tensor, batch: Batch
    ) -> None:
        preds_labels_flat = (preds_probs >= self.cfg.thr).view(-1).long()
        true_labels_flat = true_labels.view(-1).long()

        if self.ignore_value is not None:
            mask = true_labels_flat != self.ignore_value
            preds_labels_flat = preds_labels_flat[mask]
            true_labels_flat = true_labels_flat[mask]

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

    def plot(self) -> List[Tuple[str, plt.Figure]]:
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
        return [("confusion_matrix", fig)]


class BinaryPrecisionRecallCurve(Metric):
    def __init__(self, ignore_value: Optional[int] = None):
        self.ignore_value = ignore_value
        self.reset()

    def update(
        self, preds_probs: torch.Tensor, true_labels: torch.Tensor, batch: Batch
    ) -> None:
        preds_flat = preds_probs.detach().view(-1).cpu()
        labels_flat = true_labels.detach().view(-1).cpu()

        if self.ignore_value is not None:
            mask = labels_flat != self.ignore_value
            preds_flat = preds_flat[mask]
            labels_flat = labels_flat[mask]

        self.all_preds_probs.append(preds_flat)
        self.all_true_labels.append(labels_flat)

    def compute(self) -> List[Tuple[str, float]]:
        return []

    def reset(self):
        self.all_preds_probs = []
        self.all_true_labels = []

    def plot(self) -> List[Tuple[str, plt.Figure]]:
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
        return [("Precision-Recall Curve", fig)]


class Collector(Metric, Generic[T]):
    def __init__(self, target_label: int, cfg: Config, feature_names: Sequence[str]):
        self.cfg = cfg
        self.target_label = target_label
        self.positive_type = "TP" if self.target_label == 1 else "FP"
        self.feature_names = feature_names
        self.storage: TopKStorage[T] = TopKStorage(self.cfg.collector.n_frames)

    @property
    def frames(self) -> List[Tuple[float, T]]:
        return self._fetch_frames()

    @abstractmethod
    def _fetch_frames(self) -> List[Tuple[float, T]]: ...

    def __len__(self) -> int:
        return len(self.storage._items)

    def compute(self) -> List[Tuple[str, float]]:
        return []

    def reset(self) -> None:
        self.storage.clear()


class FrameCollector(Collector[Data]):
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

    def plot(self) -> List[Tuple[str, plt.Figure]]:
        entries = self.storage.get_all_entries()

        if not entries:
            return []

        fig = plot_pitch_frames_grid(
            entries, self.feature_names, self.cfg.pitch_grid.model_dump()
        )

        return [(f"{self.positive_type}_frames", fig)]

    def _fetch_frames(self):
        return self.storage.get_all_entries()


class ChainCollector(Collector[Tuple[np.ndarray, List[Data]]]):
    def update(
        self,
        preds_probs: torch.Tensor,
        true_labels: torch.Tensor,
        batch: Discrete_Signal,
    ) -> None:
        probs_np = preds_probs.detach().cpu().numpy()
        labels_np = true_labels.detach().cpu().numpy()

        last_t = batch.masks.sum(axis=0)  # (B,)

        for i, t in enumerate(last_t):
            conf = probs_np[t - 1, i]

            if (conf > self.cfg.metrics.thr) & (labels_np[0, i] == self.target_label):
                chain_predictions = probs_np[:t, i]
                chain = extract_chain(batch[:t], i)
                self.storage.add(
                    (
                        float(conf),
                        (chain_predictions, chain),
                    )
                )

    def plot(self) -> List[Tuple[str, plt.Figure]]:
        chain_predictions = [entry[1][0] for entry in self.storage.get_all_entries()]
        if not chain_predictions:
            return []

        max_len = max(map(len, chain_predictions))
        padded_chain_predictions = np.asarray(
            [
                np.pad(pred, (0, max_len - len(pred)), constant_values=np.nan)
                for pred in chain_predictions
            ]
        )

        cell_side = 0.5
        fig_width = max(6, max_len * cell_side)
        fig_height = max(3.75, len(chain_predictions) * cell_side)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        sns.heatmap(
            padded_chain_predictions,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            vmin=0,
            vmax=1,
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            square=True,
        )
        ax.set_xlabel("Time step")
        ax.set_ylabel("Chain #")
        ax.tick_params(axis="both", length=0)
        ax.set_title(
            "Temporal evolution of each chain predictions",
            fontsize=12,
            pad=10,
        )
        fig.tight_layout()

        pitch_grid_fig = plot_pitch_frames_grid(
            self.frames, self.feature_names, self.cfg.pitch_grid.model_dump()
        )

        # plot_pitch_frames_grid([(0.0, frame) for frame in chain], self.feature_names, {"figheight": 12, "nrows": int(len(chain) / 5), "ncols": 5}).savefig("test.png")

        return [
            (f"{self.positive_type}_chains", fig),
            (f"{self.positive_type}_frames", pitch_grid_fig),
        ]

    def _fetch_frames(self):
        # Last data of each collected chain
        return [(entry[0], entry[1][1][-1]) for entry in self.storage.get_all_entries()]

    @property
    def highest_confidence_chain(self):
        entries = self.storage.get_all_entries()
        if not entries:
            return None, (None, None)
        return self.storage.get_all_entries()[0]
