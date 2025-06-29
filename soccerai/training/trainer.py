from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data.dataloader import DataLoader as TorchDataLoader
from torch_geometric.data import Batch
from torch_geometric_temporal.signal import Discrete_Signal
from tqdm import tqdm

import wandb
from soccerai.training.callbacks import Callback
from soccerai.training.metrics import Metric
from soccerai.training.trainer_config import Config

BatchEvalResult = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class BaseTrainer(ABC):
    def __init__(
        self,
        cfg: Config,
        model: nn.Module,
        train_loader: TorchDataLoader,
        device: str,
        feature_names: Optional[Sequence[str]] = None,
        val_loader: Optional[TorchDataLoader] = None,
        metrics: Optional[List[Metric]] = None,
        callbacks: Optional[List[Callback]] = None,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.model: nn.Module = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.feature_names = feature_names or []
        self.metrics = metrics or []
        self.callbacks = callbacks or []
        self.criterion = nn.BCEWithLogitsLoss()
        self.optim = AdamW(
            self.model.parameters(),
            lr=self.cfg.trainer.lr,
            weight_decay=self.cfg.trainer.wd,
        )
        self.scheduler = OneCycleLR(
            self.optim,
            max_lr=cfg.trainer.lr * 10,
            total_steps=cfg.trainer.n_epochs * len(self.train_loader),
            pct_start=0.1,
        )

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

    def _get_data_iterable(self, split: str) -> Optional[TorchDataLoader]:
        return self.train_loader if split == "train" else self.val_loader

    def _on_training_end(self) -> None:
        """
        Hook called at the end of training.
        """
        for cb in self.callbacks:
            cb.on_train_end(self)

    def train(self, run_name: str):
        wandb.init(
            project=self.cfg.project_name, name=run_name, config=self.cfg.model_dump()
        )
        wandb.watch(self.model, log="all", log_freq=100)
        try:
            for epoch in tqdm(
                range(1, self.cfg.trainer.n_epochs + 1), desc="Epoch", colour="green"
            ):
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
                    self.scheduler.step()
                    wandb.log(
                        {
                            "train/step_loss": loss.item(),
                            "train/lr": self.scheduler.get_last_lr()[0],
                        }
                    )

                if epoch % self.cfg.trainer.eval_rate == 0:
                    self.eval("train")
                    self.eval("val")

            self._on_training_end()

        finally:
            wandb.finish()

    @torch.inference_mode()
    def eval(self, split: str) -> None:
        self.model.eval()
        iterable = self._get_data_iterable(split)

        if iterable is None:
            logger.warning(
                "No data to evaluate for the '{}' split. Skipping evaluation.", split
            )
            return

        num_items = len(iterable)

        total_loss = 0.0
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
            total_loss += loss.item()

            for m in self.metrics:
                m.update(preds_probs, true_labels, item)

        mean_loss = total_loss / num_items
        log_dict: Dict[str, Any] = {f"{split}/loss": mean_loss}

        for m in self.metrics:
            for name, value in m.compute():
                log_dict[f"{split}/{name}"] = value
            for name, visual in m.plot():
                if isinstance(visual, plt.Figure):
                    log_dict[f"{split}/{name}"] = wandb.Image(visual)
                    plt.close(visual)
                else:
                    log_dict[f"{split}/{name}"] = wandb.Video(
                        visual, fps=1, format="mp4"
                    )

        wandb.log(log_dict)


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
        loss: torch.Tensor = self.criterion(out, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optim.step()
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
        loss = self.criterion(out, batch.y)
        preds_probs = torch.sigmoid(out)
        true_labels = batch.y.cpu().long()
        return loss, preds_probs, true_labels


class TemporalTrainer(BaseTrainer):
    def _compute_signal_loss_and_last_pred(
        self, signal: Discrete_Signal
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
                out, snapshot.y, reduction="none"
            ).squeeze(1)
            pred_per_timestep[t] = out.squeeze(-1)

        loss = (loss_per_timestep * weights).sum(dim=0).mean()

        return loss, pred_per_timestep

    def _train_step(self, batch: Discrete_Signal) -> torch.Tensor:
        self.optim.zero_grad(set_to_none=True)
        loss, _ = self._compute_signal_loss_and_last_pred(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optim.step()
        return loss

    def _eval_step(self, batch: Discrete_Signal) -> BatchEvalResult:
        loss, out = self._compute_signal_loss_and_last_pred(batch)
        preds_probs = torch.sigmoid(out)
        true_labels = torch.from_numpy(batch.targets).long().squeeze(2).contiguous()
        return loss, preds_probs, true_labels
