from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch


class Metric(ABC):
    @abstractmethod
    def update(self, preds_probs: torch.Tensor, true_labels: torch.Tensor) -> None:
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


class ConfusionMatrix(Metric):
    def __init__(self, thr: float = 0.5, beta: float = 1):
        self.thr = thr
        self.beta = beta
        self.reset()

    def update(self, preds_probs, true_labels) -> None:
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
        return super().reset()

    def plot(self) -> Optional[Tuple[str, plt.Figure]]:
        cm_np = self.cm.cpu().numpy()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_np, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        plt.tight_layout()
        return "confusion_matrix", fig
