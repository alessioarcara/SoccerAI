from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data.dataloader import DataLoader as TorchDataLoader
from torch_geometric.data import Batch
from torch_geometric_temporal.signal import Discrete_Signal
from tqdm import tqdm

from soccerai.training.callbacks import Callback, EarlyStoppingCallback
from soccerai.training.metrics import Metric
from soccerai.training.trainer_config import Config

BatchEvalResult = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def compute_pos_weight(labels: np.ndarray) -> float:
    """#negatives / #positives, the BCE weight that balances the classes."""
    labels = np.asarray(labels).reshape(-1)
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    return n_neg / max(n_pos, 1.0)


class BaseTrainer(ABC):
    def __init__(
        self,
        cfg: Config,
        model: nn.Module,
        train_loader: TorchDataLoader,
        device: str,
        feature_names: Sequence[str] | None = None,
        val_loader: TorchDataLoader | None = None,
        metrics: list[Metric] | None = None,
        callbacks: list[Callback] | None = None,
        pos_weight: float | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.model: nn.Module = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.feature_names = feature_names or []
        self.metrics = metrics or []
        self.callbacks = callbacks or []
        self.pos_weight = (
            None
            if pos_weight is None
            else torch.tensor(float(pos_weight), device=self.device)
        )
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.optim = AdamW(
            self.model.parameters(),
            lr=self.cfg.trainer.lr,
            weight_decay=self.cfg.trainer.wd,
        )
        # warm-up to `max_lr` over the first 10% of the steps, then anneal
        self.scheduler = OneCycleLR(
            self.optim,
            max_lr=cfg.trainer.max_lr or cfg.trainer.lr,
            total_steps=cfg.trainer.n_epochs * len(self.train_loader),
            pct_start=0.1,
        )
        self.history: dict[str, Any] = {}

    @abstractmethod
    def _train_step(self, item: Any) -> torch.Tensor:
        """
        Returns: loss
        """
        ...

    @abstractmethod
    def _eval_step(self, item: Any) -> BatchEvalResult:
        """
        Returns: (loss, prediction_probabilities, true_labels)
        """
        ...

    def _get_data_iterable(self, split: str) -> TorchDataLoader | None:
        return self.train_loader if split == "train" else self.val_loader

    def _aux_loss(self) -> torch.Tensor | float:
        """
        Auxiliary loss exposed by the model after its forward pass (e.g. the
        DiffPool link/entropy regularisers), scaled by the configured weight.
        """
        aux = getattr(self.model, "aux_loss", None)
        if aux is None:
            return 0.0
        return self.cfg.trainer.aux_loss_weight * aux

    @staticmethod
    def _num_examples(item: Any) -> int:
        """Number of training examples (graphs or chains) in a batch."""
        if isinstance(item, Discrete_Signal):
            return int(np.asarray(item.masks).shape[1])
        return int(item.num_graphs)

    def _on_training_end(self) -> None:
        """
        Hook called at the end of training.
        """
        for cb in self.callbacks:
            cb.on_train_end(self)

    def _on_eval_end(self) -> None:
        """
        Hook called at the end of an evaluation.
        """
        for cb in self.callbacks:
            cb.on_eval_end(self)

    def _check_for_early_stop(self) -> bool:
        return any(
            isinstance(cb, EarlyStoppingCallback) and cb.should_stop
            for cb in self.callbacks
        )

    def train(self, run_name: str):
        wandb.init(
            project=self.cfg.project_name, name=run_name, config=self.cfg.model_dump()
        )
        wandb.watch(self.model, log="all", log_freq=100)
        try:
            for epoch in tqdm(
                range(1, self.cfg.trainer.n_epochs + 1), desc="Epoch", colour="green"
            ):
                # `eval()` switches the model to eval mode at the end of every
                # epoch: dropout / batch-norm must be re-enabled for training.
                self.model.train()
                train_iterable = self._get_data_iterable("train")
                assert train_iterable is not None

                for item in tqdm(
                    train_iterable,
                    total=len(train_iterable),
                    desc=f"Epoch {epoch} Batches",
                    leave=False,
                    colour="blue",
                ):
                    loss = self._train_step(item)
                    self._wandb_log(
                        {
                            "train/step_loss": loss.item(),
                            "train/lr": self.scheduler.get_last_lr()[0],
                        }
                    )

                if epoch % self.cfg.trainer.eval_rate == 0:
                    self.eval("train")
                    self.eval("val")
                    self._on_eval_end()

                    if self._check_for_early_stop():
                        logger.info("Early stopping triggered! No improvement.")
                        break

            self._on_training_end()

        finally:
            wandb.finish()

    @torch.inference_mode()
    def eval(self, split: Literal["train", "val"]) -> None:
        self.model.eval()
        iterable = self._get_data_iterable(split)

        if iterable is None:
            logger.warning(
                "No data to evaluate for the '{}' split. Skipping evaluation.", split
            )
            return

        num_items = len(iterable)

        total_loss = 0.0
        total_examples = 0
        for m in self.metrics:
            m.reset()

        for item in tqdm(
            iterable,
            total=num_items,
            desc=f"Evaluating {split}",
            leave=False,
            colour="red",
        ):
            loss, preds_probs, true_labels = self._eval_step(item)
            # the loss is a mean over the examples of the batch: weight it by
            # the batch size so that a smaller last batch does not count more
            n_examples = self._num_examples(item)
            total_loss += loss.item() * n_examples
            total_examples += n_examples

            for m in self.metrics:
                m.update(preds_probs, true_labels, item)

        mean_loss = total_loss / max(total_examples, 1)

        if split == "val":
            self.history["val_loss"] = mean_loss
        log_dict: dict[str, Any] = {f"{split}/loss": mean_loss}

        for m in self.metrics:
            for name, value in m.compute():
                if split == "val":
                    self.history[f"val_{name}"] = value
                log_dict[f"{split}/{name}"] = value

            for name, visual in m.plot():
                if isinstance(visual, plt.Figure):
                    log_dict[f"{split}/{name}"] = wandb.Image(visual)
                    plt.close(visual)
                else:
                    log_dict[f"{split}/{name}"] = wandb.Video(
                        visual, fps=1, format="mp4"
                    )

        self._wandb_log(log_dict)

    @staticmethod
    def _wandb_log(payload: dict[str, Any]) -> None:
        """Log to W&B only when a run is active (eval can run standalone)."""
        if wandb.run is not None:
            wandb.log(payload)


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
        loss: torch.Tensor = self.criterion(out, batch.y) + self._aux_loss()
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
        loss = self.criterion(out, batch.y) + self._aux_loss()
        preds_probs = torch.sigmoid(out)
        true_labels = batch.y.cpu().long()
        return loss, preds_probs, true_labels


class TemporalTrainer(BaseTrainer):
    def _compute_signal_loss_and_last_pred(
        self, signal: Discrete_Signal
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masks = torch.tensor(signal.masks, dtype=torch.bool, device=self.device).T
        B, T_max = masks.shape

        # ----------------- Computing discount weights --------------------
        lengths = masks.sum(dim=1)  # (B,)
        T = torch.arange(T_max, device=self.device)  # (T_max,)

        # (B, 1) − (T_max,) => (B, T_max) tramite broadcasting
        exps = (lengths - 1).unsqueeze(1) - T

        weights = torch.where(masks, self.cfg.trainer.gamma ** exps.float(), 0.0)
        weights /= weights.sum(dim=1, keepdim=True).clamp(min=1e-12)

        weights = weights.T.contiguous()  # (T_max, B)
        # ------------------------------------------------------------------

        loss_per_timestep = torch.empty_like(weights)
        pred_per_timestep = torch.empty_like(weights)

        h = None
        c = None
        aux_loss: torch.Tensor | float = 0.0
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

            loss_per_timestep[t] = F.binary_cross_entropy_with_logits(
                out, snapshot.y, reduction="none", pos_weight=self.pos_weight
            ).squeeze(1)
            pred_per_timestep[t] = out.squeeze(-1)
            aux_loss = aux_loss + self._aux_loss()

        loss = (loss_per_timestep * weights).sum(dim=0).mean() + aux_loss / T_max

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
