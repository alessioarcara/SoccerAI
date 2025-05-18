from abc import ABC, abstractmethod

import torch


class Metric(ABC):
    @abstractmethod
    def update(self, preds_probs: torch.Tensor, true_labels: torch.Tensor):
        pass

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class BinaryAccuracy(Metric):
    def __init__(self, thr: float = 0.5):
        self.thr = thr
        self.reset()

    def update(self, preds_probs: torch.Tensor, true_labels: torch.Tensor):
        pred_labels = (preds_probs >= self.thr).float()

        correct_in_batch = (pred_labels == true_labels).sum().item()

        self.correct_count += correct_in_batch
        self.total_count += true_labels.numel()

    def compute(self):
        if self.total_count == 0:
            return 0.0
        return self.correct_count / self.total_count

    def reset(self):
        self.correct_count = 0
        self.total_count = 0
