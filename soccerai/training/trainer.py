from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from eztrain import EpochTrainer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch_geometric.data import Batch
from torch_geometric_temporal.signal import Discrete_Signal

from soccerai.training.metrics import chain_level_predictions

BatchEvalResult = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def resolve_device(device: str) -> str:
    """`auto` picks the GPU when there is one."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class BaseTrainer(EpochTrainer, ABC):
    """
    EzTrain epoch trainer for graph models: EzTrain owns the loop (epochs,
    periodic evaluation, callbacks, run identity and resume, logging), this
    class the optimisation and evaluation steps.

    Every evaluation scores the training split too (`train/...`, in eval
    mode) and then the validation split (`val/...`): the loss is averaged
    per example, so a smaller last batch does not count more. During an
    epoch the mean training-mode loss and the learning rate are logged as
    `train/step_loss` and `train/lr`.

    All other keyword arguments (`train_loader`, `val_loader`, `metrics`,
    `max_iterations`, `eval_freq`, `callbacks`, `logger`, `run_name`,
    `resume_from`, `config`) go to `eztrain.EpochTrainer`.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        pos_weight: float | Sequence[float] | None = None,
        aux_loss_weight: float = 1.0,
        feature_names: Sequence[str] | None = None,
        device: str = "auto",
        evaluate_train: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.device = resolve_device(device)
        # moving the model keeps the same Parameter objects: the optimizer
        # built from them stays valid
        self.model: nn.Module = model.to(self.device)
        self.optim = optimizer
        self.scheduler = scheduler
        self.aux_loss_weight = aux_loss_weight
        self.feature_names = list(feature_names or [])
        self.evaluate_train = evaluate_train
        self.pos_weight = (
            None
            if pos_weight is None
            else torch.as_tensor(pos_weight, dtype=torch.float32, device=self.device)
        )
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self._loss_sum = 0.0
        self._n_examples = 0

    @property
    def checkpointables(self) -> Mapping[str, Any]:
        return {
            "model": self.model,
            "optimizer": self.optim,
            "scheduler": self.scheduler,
        }

    # --- optimisation and evaluation steps ----------------------------------

    @abstractmethod
    def _train_step(self, item: Any) -> torch.Tensor:
        """Forward, backward and optimizer step; returns the loss."""

    @abstractmethod
    def _eval_step(self, item: Any) -> BatchEvalResult:
        """Returns (loss, prediction_probabilities, true_labels)."""

    def _aux_loss(self) -> torch.Tensor:
        """
        Auxiliary loss exposed by the model after its forward pass (e.g. the
        DiffPool link/entropy regularisers), scaled by the configured weight.
        Either a scalar or one value per graph of the batch, shape (B,).
        """
        aux = getattr(self.model, "aux_loss", None)
        if aux is None:
            return torch.zeros((), device=self.device)
        return self.aux_loss_weight * aux

    def _per_example(
        self, preds_probs: torch.Tensor, true_labels: torch.Tensor, item: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One prediction and label per training example (graph or chain)."""
        return preds_probs, true_labels

    @staticmethod
    def _num_examples(item: Any) -> int:
        """Number of training examples (graphs or chains) in a batch."""
        if isinstance(item, Discrete_Signal):
            return int(np.asarray(item.masks).shape[1])
        return int(item.num_graphs)

    # --- eztrain.EpochTrainer -------------------------------------------------

    def train_iteration(self, iteration: int) -> Mapping[str, Any]:
        # evaluation switches the model to eval mode: re-enable dropout and
        # batch-norm updates for every epoch
        self.model.train()
        logs = dict(super().train_iteration(iteration))
        logs["train/lr"] = self.scheduler.get_last_lr()[0]
        return logs

    def train_step(self, batch: Any) -> Mapping[str, float]:
        return {"step_loss": self._train_step(batch).item()}

    def evaluate(self) -> Mapping[str, Any]:
        self.model.eval()
        with torch.inference_mode():
            logs: dict[str, Any] = {}
            if self.evaluate_train:
                logs.update(self.evaluate_split(self.train_loader, "train"))
            logs.update(super().evaluate())
        return logs

    def evaluate_split(self, loader: Iterable[Any], prefix: str) -> dict[str, Any]:
        self._loss_sum, self._n_examples = 0.0, 0
        results = super().evaluate_split(loader, prefix)
        results[f"{prefix}/loss"] = self._loss_sum / max(self._n_examples, 1)
        return results

    def eval_step(self, batch: Any) -> Mapping[str, float]:
        loss, preds_probs, true_labels = self._eval_step(batch)
        # the loss is a mean over the examples of the batch: weight it by
        # the batch size so that a smaller last batch does not count more
        n_examples = self._num_examples(batch)
        self._loss_sum += loss.item() * n_examples
        self._n_examples += n_examples

        example_preds, example_labels = self._per_example(
            preds_probs, true_labels, batch
        )
        for metric in self.metrics:
            if getattr(metric, "frame_level", False):
                metric.update(preds_probs, true_labels, batch)
            else:
                metric.update(example_preds, example_labels, batch)
        return {}


class Trainer(BaseTrainer):
    def _train_step(self, batch: Batch) -> torch.Tensor:
        self.optim.zero_grad(set_to_none=True)
        out = self.model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_weight=batch.edge_weight,
            edge_attr=batch.edge_attr,
            u=batch.u,
            batch=batch.batch,
            batch_size=batch.num_graphs,
        )
        loss: torch.Tensor = self.criterion(out, batch.y) + self._aux_loss().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optim.step()
        self.scheduler.step()
        return loss

    def _eval_step(self, batch: Batch) -> BatchEvalResult:
        out = self.model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_weight=batch.edge_weight,
            edge_attr=batch.edge_attr,
            u=batch.u,
            batch=batch.batch,
            batch_size=batch.num_graphs,
        )
        loss = self.criterion(out, batch.y) + self._aux_loss().mean()
        preds_probs = torch.sigmoid(out)
        true_labels = batch.y.cpu().long()
        return loss, preds_probs, true_labels


class TemporalTrainer(BaseTrainer):
    """
    Trainer over batches of chains. `gamma` discounts the per-frame losses
    towards the start of a chain (the last frame weighs 1, the one before
    `gamma`, ...; 1 weighs every frame alike).

    `rank_loss_weight` adds the ranking loss of Ma, Sigal and Sclaroff
    (CVPR 2016) on the positive chains: the predicted probability should
    never drop as a dangerous action unfolds, so every frame is penalised by
    how far it falls below the highest probability of the frames before it.
    Negative chains are left free, since an action can become dangerous and
    then be stopped.

    `lead_weight` (per-frame shot targets) multiplies the loss of a frame
    whose team shoots within `lead_horizon` seconds by
    1 + lead_weight * time_to_shot / lead_horizon: the earliest warnings,
    the hardest and most useful ones, weigh the most.
    """

    def __init__(
        self,
        *,
        gamma: float = 0.1,
        rank_loss_weight: float = 0.0,
        lead_weight: float = 0.0,
        lead_horizon: float = 8.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.gamma = gamma
        self.rank_loss_weight = rank_loss_weight
        self.lead_weight = lead_weight
        self.lead_horizon = lead_horizon

    def _per_example(
        self,
        preds_probs: torch.Tensor,
        true_labels: torch.Tensor,
        item: Discrete_Signal,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return chain_level_predictions(preds_probs, item.chain_label, item.masks)

    def _frame_loss(
        self, out: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Per-graph loss and shot logit of one time step; unknown targets (-1)
        are clamped here and weighed 0 by the caller.

        One output: BCE on the shot target. Three outputs (box decomposition,
        targets [box, shot]): logits of P(box), P(shot | box) and
        P(shot | no box), each head trained on the frames it conditions on;
        the shot logit returned combines them,
        P(shot) = P(box) P(shot | box) + (1 - P(box)) P(shot | no box),
        after removing the bias of the positive-class weights.
        """
        y = y.clamp(min=0)
        pw = self.pos_weight
        if out.shape[-1] == 1:
            loss = F.binary_cross_entropy_with_logits(
                out, y, reduction="none", pos_weight=pw
            ).squeeze(1)
            return loss, out.squeeze(-1)

        box, shot = y[:, 0], y[:, 1]
        head_pw = [None] * 3 if pw is None else list(pw.reshape(-1))

        def bce(logit: torch.Tensor, target: torch.Tensor, i: int) -> torch.Tensor:
            return F.binary_cross_entropy_with_logits(
                logit, target, reduction="none", pos_weight=head_pw[i]
            )

        loss = (
            bce(out[:, 0], box, 0)
            + box * bce(out[:, 1], shot, 1)
            + (1 - box) * bce(out[:, 2], shot, 2)
        )
        # a head trained with weight w on its positives predicts w times the
        # true odds: undo it before combining the probabilities
        if pw is not None:
            out = out - torch.log(pw.reshape(-1))
        p_box, p_shot_box, p_shot_no_box = torch.sigmoid(out).unbind(-1)
        p_shot = p_box * p_shot_box + (1 - p_box) * p_shot_no_box
        return loss, torch.logit(p_shot, eps=1e-6)

    def _lead_multiplier(self, signal: Discrete_Signal) -> torch.Tensor:
        """(T_max, B) loss multipliers favouring the frames far from the shot."""
        time_to_shot = torch.as_tensor(
            np.asarray(signal.time_to_shot), dtype=torch.float32, device=self.device
        )
        ahead = (time_to_shot / self.lead_horizon).nan_to_num(nan=2.0)
        return torch.where(ahead <= 1, 1 + self.lead_weight * ahead.clamp(min=0), 1.0)

    @staticmethod
    def _rank_loss(
        logits: torch.Tensor, masks: torch.Tensor, positive: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean over the frames of every positive chain of
        `max(0, max_{s<t} p_s - p_t)`, averaged over the chains of the batch
        (negative chains count as zero). Shapes: logits and masks (T_max, B),
        positive (B,).
        """
        probs = torch.sigmoid(logits)
        # padded frames sit at the end of a chain: they never raise the
        # running maximum of a real frame
        prev_max = torch.cummax(probs, dim=0).values[:-1]
        drops = F.relu(prev_max - probs[1:]) * masks[1:]
        per_chain = drops.sum(dim=0) / masks.sum(dim=0).clamp(min=1)
        return (per_chain * positive).mean()

    def _compute_signal_loss_and_last_pred(
        self, signal: Discrete_Signal
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masks = torch.tensor(signal.masks, dtype=torch.bool, device=self.device).T
        B, T_max = masks.shape
        # frames with a known target (-1: padding, or a future past the end
        # of the period); censored frames sit at the end of a chain
        targets = np.asarray(signal.targets)  # (T_max, B, n_targets)
        known = masks & torch.as_tensor(targets[..., 0] >= 0, device=self.device).T

        # ----------------- Computing discount weights --------------------
        lengths = masks.sum(dim=1)  # (B,)
        n_known = known.sum(dim=1)  # (B,)
        T = torch.arange(T_max, device=self.device)  # (T_max,)

        # (B, 1) − (T_max,) => (B, T_max) tramite broadcasting
        exps = (n_known - 1).unsqueeze(1) - T

        weights = torch.where(known, self.gamma ** exps.float(), 0.0)
        weights /= weights.sum(dim=1, keepdim=True).clamp(min=1e-12)

        weights = weights.T.contiguous()  # (T_max, B)
        # ------------------------------------------------------------------

        loss_per_timestep = torch.empty_like(weights)
        pred_per_timestep = torch.empty_like(weights)

        h = None
        c = None
        aux_per_timestep = torch.zeros_like(weights)
        for t, snapshot in enumerate(signal):
            snapshot.to(self.device, non_blocking=True)

            out, h, c = self.model(
                x=snapshot.x,
                edge_index=snapshot.edge_index,
                edge_weight=snapshot.edge_attr,
                edge_attr=snapshot.edge_attr,
                u=snapshot.u,
                batch=snapshot.batch,
                batch_size=snapshot.num_graphs,
                prev_h=h,
                prev_c=c,
            )

            loss_per_timestep[t], pred_per_timestep[t] = self._frame_loss(
                out, snapshot.y
            )
            aux_per_timestep[t] = self._aux_loss().expand(B)

        # auxiliary loss averaged over the real frames of every chain, so that
        # padded (all-zero) snapshots contribute neither loss nor gradient
        valid = masks.T.to(aux_per_timestep.dtype)  # (T_max, B)
        aux_loss = (aux_per_timestep * valid).sum(dim=0) / lengths.clamp(min=1)

        if self.lead_weight:
            loss_per_timestep = loss_per_timestep * self._lead_multiplier(signal)
        loss = (loss_per_timestep * weights).sum(dim=0).mean() + aux_loss.mean()
        if self.rank_loss_weight:
            positive = torch.as_tensor(
                np.asarray(signal.chain_label)[0] == 1, device=self.device
            )
            loss = loss + self.rank_loss_weight * self._rank_loss(
                pred_per_timestep, known.T.to(pred_per_timestep.dtype), positive
            )

        return loss, pred_per_timestep

    def _train_step(self, batch: Discrete_Signal) -> torch.Tensor:
        self.optim.zero_grad(set_to_none=True)
        loss, _ = self._compute_signal_loss_and_last_pred(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optim.step()
        self.scheduler.step()
        return loss

    def _eval_step(self, batch: Discrete_Signal) -> BatchEvalResult:
        loss, out = self._compute_signal_loss_and_last_pred(batch)
        preds_probs = torch.sigmoid(out)
        true_labels = torch.from_numpy(batch.targets).long().squeeze(2).contiguous()
        return loss, preds_probs, true_labels
