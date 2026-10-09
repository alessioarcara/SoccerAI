from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import (
    Any,
    Generic,
    Literal,
    TypeVar,
)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from eztrain import Image, Video
from matplotlib.collections import LineCollection
from torch_geometric.data import Batch, Data
from torch_geometric_temporal.signal import Discrete_Signal
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_average_precision,
    binary_precision_recall_curve,
)

from soccerai.training.utils import (
    TopKStorage,
    extract_chain,
    plot_chain_frames,
    plot_pitch_frames_grid,
)

T = TypeVar("T")


def chain_level_predictions(
    preds_probs: torch.Tensor, chain_labels: Any, masks: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reduce per-frame predictions of a temporal batch, shape (T_max, B), to one
    prediction per chain, taken at the last valid (non padded) frame, paired
    with the chain labels (shape (T_max, B), constant along a chain).

    The chain label is a property of the whole chain, so scoring every frame
    (including the first ones, which are indistinguishable between classes)
    would measure a different task: `EarlyWarning` measures how early the
    prediction rises instead.
    """
    masks = torch.as_tensor(np.asarray(masks), dtype=torch.bool)  # (T, B)
    last_valid = masks.sum(dim=0) - 1  # (B,)
    chain_idx = torch.arange(masks.shape[1])

    preds_last = preds_probs[
        last_valid.to(preds_probs.device), chain_idx.to(preds_probs.device)
    ]
    # chain label, always valid at t = 0
    labels_last = torch.as_tensor(np.asarray(chain_labels))[0].long()
    return preds_last, labels_last


class Metric(ABC):
    """
    A metric for `eztrain.MetricCollection`: `compute` returns named scalars,
    `plot` named media, both logged under the evaluated split.
    """

    # True for metrics that need the per-frame outputs of a temporal batch;
    # the others receive one prediction per chain (see `TemporalTrainer`)
    frame_level: bool = False

    @abstractmethod
    def update(
        self, preds_probs: torch.Tensor, true_labels: torch.Tensor, batch: Batch
    ) -> None:
        pass

    @abstractmethod
    def compute(self) -> dict[str, float]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    def plot(self) -> dict[str, Image | Video]:
        """Named figures (`Image`) and frame sequences (`Video`) to log."""
        return {}


class BinaryConfusionMatrix(Metric):
    def __init__(
        self,
        threshold: float = 0.5,
        fbeta: float = 1.0,
        ignore_value: int | None = None,
        mode: Literal["pos", "both"] = "pos",
    ):
        self.threshold = threshold
        self.fbeta = fbeta
        self.ignore_value = ignore_value
        self.mode = mode
        self.reset()

    def update(
        self, preds_probs: torch.Tensor, true_labels: torch.Tensor, batch: Batch
    ) -> None:
        preds_labels_flat = (preds_probs >= self.threshold).view(-1).long().cpu()
        true_labels_flat = true_labels.view(-1).long().cpu()

        if self.ignore_value is not None:
            mask = true_labels_flat != self.ignore_value
            preds_labels_flat = preds_labels_flat[mask]
            true_labels_flat = true_labels_flat[mask]

        self.cm += torch.bincount(
            true_labels_flat * 2 + preds_labels_flat, minlength=4
        ).view(2, 2)

    def _get_fbeta(self, tp: float, fp: float, fn: float) -> float:
        beta2 = self.fbeta**2
        denom = (1 + beta2) * tp + beta2 * fn + fp
        fbeta = ((1 + beta2) * tp / denom) if denom > 0 else 0.0
        return fbeta

    def compute(self) -> dict[str, float]:
        tn, fp = self.cm[0, 0].item(), self.cm[0, 1].item()
        fn, tp = self.cm[1, 0].item(), self.cm[1, 1].item()

        total = tn + fp + fn + tp
        results: dict[str, float] = {}

        # Accuracy
        accuracy = (tp + tn) / total if total > 0 else 0.0
        results["accuracy"] = accuracy

        # Pos F-beta
        fbeta_pos = self._get_fbeta(tp=tp, fp=fp, fn=fn)
        results[f"f{self.fbeta}_pos_score"] = fbeta_pos

        if self.mode == "both":
            # Neg F-beta
            fbeta_neg = self._get_fbeta(tp=tn, fp=fn, fn=fp)
            results[f"f{self.fbeta}_neg_score"] = fbeta_neg

        return results

    def reset(self) -> None:
        self.cm = torch.zeros((2, 2), dtype=torch.int64)

    def plot(self) -> dict[str, Image | Video]:
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
        return {"confusion_matrix": Image(fig)}


class BinaryPrecisionRecallCurve(Metric):
    def __init__(self, ignore_value: int | None = None):
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

    def compute(self) -> dict[str, float]:
        all_preds_probs_flat = torch.cat(self.all_preds_probs)
        all_true_labels_flat = torch.cat(self.all_true_labels).long()

        ap = binary_average_precision(all_preds_probs_flat, all_true_labels_flat)
        auroc = binary_auroc(all_preds_probs_flat, all_true_labels_flat)
        return {"average_precision": ap.item(), "auroc": auroc.item()}

    def reset(self):
        self.all_preds_probs = []
        self.all_true_labels = []

    def plot(self) -> dict[str, Image | Video]:
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
        return {"precision_recall_curve": Image(fig)}


class Collector(Metric, Generic[T]):
    frame_level = True

    """
    Keep the `n_frames` most confident examples of class `target_label`
    predicted above `threshold`, and plot them on a pitch grid. With
    `n_frames = 0` nothing is collected.
    """

    def __init__(
        self,
        target_label: int,
        feature_names: Sequence[str],
        n_frames: int,
        threshold: float = 0.5,
        grid_nrows: int = 6,
        grid_ncols: int = 4,
        grid_figheight: int = 12,
    ):
        self.target_label = target_label
        self.positive_type = "TP" if self.target_label == 1 else "FP"
        self.feature_names = feature_names
        self.threshold = threshold
        self.pitch_grid = {
            "nrows": grid_nrows,
            "ncols": grid_ncols,
            "figheight": grid_figheight,
        }
        self.storage: TopKStorage[T] = TopKStorage(n_frames)

    @property
    def frames(self) -> list[tuple[float, T]]:
        return self._fetch_frames()

    @abstractmethod
    def _fetch_frames(self) -> list[tuple[float, T]]: ...

    def __len__(self) -> int:
        return len(self.storage._items)

    def compute(self) -> dict[str, float]:
        return {}

    def reset(self) -> None:
        self.storage.clear()


class FrameCollector(Collector[Data]):
    def update(
        self,
        preds_probs: torch.Tensor,
        true_labels: torch.Tensor,
        batch: Batch,
    ) -> None:
        if self.storage.k == 0:
            return
        probs_np = preds_probs.detach().cpu().numpy()
        labels_np = true_labels.detach().cpu().numpy()

        indices = np.where(
            (probs_np >= self.threshold) & (labels_np == self.target_label)
        )[0]

        for i in indices:
            self.storage.add((float(probs_np[i]), batch[i]))

    def plot(self) -> dict[str, Image | Video]:
        entries = self.storage.get_all_entries()

        if not entries:
            return {}

        fig = plot_pitch_frames_grid(entries, self.feature_names, self.pitch_grid)

        return {f"{self.positive_type}_frames": Image(fig)}

    def _fetch_frames(self):
        return self.storage.get_all_entries()


class ChainCollector(Collector[tuple[np.ndarray, list[Data]]]):
    def update(
        self,
        preds_probs: torch.Tensor,
        true_labels: torch.Tensor,
        batch: Discrete_Signal,
    ) -> None:
        if self.storage.k == 0:
            return
        probs_np = preds_probs.detach().cpu().numpy()
        chain_labels = np.asarray(batch.chain_label)[0]

        last_t = batch.masks.sum(axis=0)  # (B,)

        for i, t in enumerate(last_t):
            conf = probs_np[t - 1, i]

            if (conf >= self.threshold) & (chain_labels[i] == self.target_label):
                chain_predictions = probs_np[:t, i]
                chain = extract_chain(batch[:t], i)
                self.storage.add(
                    (
                        float(conf),
                        (chain_predictions, chain),
                    )
                )

    def plot(self) -> dict[str, Image | Video]:
        chain_predictions = [entry[1][0] for entry in self.storage.get_all_entries()]
        if not chain_predictions:
            return {}

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
            xticklabels=range(1, max_len + 1),
            yticklabels=range(1, len(chain_predictions) + 1),
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
            self.frames,
            self.feature_names,
            self.pitch_grid,
            item_name="Chain",
            title=f"{self.positive_type} · Last valid frame of each chain",
        )

        scores_np, snapshots = self.highest_confidence_chain[1]
        chain_frames_np = plot_chain_frames(
            snapshots, scores_np.tolist(), self.feature_names
        )

        return {
            f"{self.positive_type}_chains_predictions": Image(fig),
            f"{self.positive_type}_chains_last_frames": Image(pitch_grid_fig),
            f"{self.positive_type}_chains_frames": Video(chain_frames_np, fps=1),
        }

    def _fetch_frames(self):
        # Last data of each collected chain
        return [(entry[0], entry[1][1][-1]) for entry in self.storage.get_all_entries()]

    @property
    def highest_confidence_chain(self):
        entries = self.storage.get_all_entries()
        if not entries:
            return

        return entries[0]


class EarlyWarning(Metric):
    """
    How early the prediction flags the actions that end in a shot.

    An alarm fires at the first frame whose probability exceeds a threshold;
    the threshold is set on the negative chains so that at most
    `false_alarm_rate` of them ever raise one. On the positive chains this
    gives:

    - `early_recall`: fraction of chains flagged before their shot;
    - `early_recall_<min_lead>s`: fraction flagged at least `min_lead`
      seconds before the shot;
    - `early_median_lead_s`: median warning time of the flagged chains;
    - `auroc_lead_<lo>-<hi>s`: frame-level AUROC of the positive frames
      `lo`-`hi` seconds before their shot against every negative frame, i.e.
      how separable the classes are that far ahead.

    Needs the per-frame `time_to_shot` and `chain_label` of the batch.
    """

    frame_level = True

    def __init__(
        self,
        false_alarm_rate: float = 0.1,
        min_lead: float = 3.0,
        lead_bins: Sequence[float] = (0, 2, 4, 6, 10, 15, 30),
    ):
        if not 0 < false_alarm_rate < 1:
            raise ValueError(
                f"false_alarm_rate must be in (0, 1), got {false_alarm_rate}"
            )
        self.false_alarm_rate = false_alarm_rate
        self.min_lead = min_lead
        self.lead_bins = list(lead_bins)
        self.reset()

    def reset(self) -> None:
        # one (label, per-frame probabilities, per-frame time to shot) per chain
        self.chains: list[tuple[int, np.ndarray, np.ndarray]] = []

    def update(
        self, preds_probs: torch.Tensor, true_labels: torch.Tensor, batch: Batch
    ) -> None:
        probs = preds_probs.detach().cpu().numpy()  # (T_max, B)
        lengths = np.asarray(batch.masks).sum(axis=0)
        chain_labels = np.asarray(batch.chain_label)[0]
        time_to_shot = np.asarray(batch.time_to_shot)
        for b, length in enumerate(lengths):
            self.chains.append(
                (int(chain_labels[b]), probs[:length, b], time_to_shot[:length, b])
            )

    def _threshold(self) -> float | None:
        peaks = [p.max() for label, p, _ in self.chains if label == 0]
        if not peaks:
            return None
        return float(np.quantile(peaks, 1 - self.false_alarm_rate))

    def _leads(self, threshold: float) -> np.ndarray:
        """Warning time of every positive chain (NaN when never flagged)."""
        leads = []
        for label, probs, time_to_shot in self.chains:
            if label != 1:
                continue
            alarms = np.flatnonzero(probs > threshold)
            leads.append(time_to_shot[alarms[0]] if alarms.size else np.nan)
        return np.asarray(leads, dtype=float)

    def compute(self) -> dict[str, float]:
        threshold = self._threshold()
        if threshold is None or not any(label == 1 for label, _, _ in self.chains):
            return {}
        leads = self._leads(threshold)
        flagged = ~np.isnan(leads)
        results = {
            "early_threshold": threshold,
            "early_recall": float(flagged.mean()),
            f"early_recall_{self.min_lead:g}s": float(
                (np.nan_to_num(leads, nan=-1.0) >= self.min_lead).mean()
            ),
            "early_median_lead_s": float(np.median(leads[flagged]))
            if flagged.any()
            else 0.0,
        }

        neg_probs = np.concatenate([p for label, p, _ in self.chains if label == 0])
        pos_probs = np.concatenate([p for label, p, _ in self.chains if label == 1])
        pos_lead = np.concatenate([t for label, _, t in self.chains if label == 1])
        for lo, hi in zip(self.lead_bins[:-1], self.lead_bins[1:]):
            in_bin = pos_probs[(pos_lead >= lo) & (pos_lead < hi)]
            if in_bin.size == 0:
                continue
            preds = torch.from_numpy(np.concatenate([in_bin, neg_probs]))
            labels = torch.cat(
                [torch.ones(in_bin.size), torch.zeros(neg_probs.size)]
            ).long()
            results[f"auroc_lead_{lo:g}-{hi:g}s"] = binary_auroc(preds, labels).item()
        return results

    def plot(self) -> dict[str, Image | Video]:
        threshold = self._threshold()
        if threshold is None:
            return {}
        leads = self._leads(threshold)
        if leads.size == 0:
            return {}

        grid = np.linspace(0.0, self.lead_bins[-1], 121)
        recall = [(np.nan_to_num(leads, nan=-1.0) >= g).mean() for g in grid]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(grid, recall, color="#2a78d6", linewidth=2, drawstyle="steps-post")
        ax.axvline(self.min_lead, color="#52514e", linewidth=1, linestyle="--")
        at_min = (np.nan_to_num(leads, nan=-1.0) >= self.min_lead).mean()
        ax.annotate(
            f"{at_min:.0%} flagged ≥ {self.min_lead:g} s ahead",
            xy=(self.min_lead, at_min),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=12,
            color="#0b0b0b",
        )
        ax.set_xlabel("Warning time before the shot (s)", fontsize=16)
        ax.set_ylabel("Shot chains flagged", fontsize=16)
        ax.set_title(
            f"Early warning at {self.false_alarm_rate:.0%} false alarms",
            fontsize=14,
        )
        ax.set_xlim(0, self.lead_bins[-1])
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(True, color="#e5e5e3", linewidth=0.8)
        ax.tick_params(axis="both", which="major", labelsize=12)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        plt.tight_layout()
        return {"early_warning_curve": Image(fig)}
