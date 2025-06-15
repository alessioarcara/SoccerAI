from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader as TorchDataLoader
from torch_geometric.data import Batch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric_temporal.signal import Discrete_Signal
from tqdm import tqdm

import wandb
from soccerai.training.metrics import Metric, PositiveFrameCollector
from soccerai.training.trainer_config import Config


class BaseTrainer(ABC):
    def __init__(
        self,
        cfg: Config,
        model: nn.Module,
        train_loader: TorchDataLoader,
        device: str,
        feature_names: Optional[Sequence[str]] = None,
        val_loader: Optional[TorchDataLoader] = None,
        metrics: List[Metric] = [],
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.model: nn.Module = model.to(self.device)
        # self.model = torch.compile(self.model)  # type: ignore[arg-type]
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.feature_names: List[str] = list(feature_names or [])
        self.metrics: List[Metric] = metrics or []
        self.criterion = nn.BCEWithLogitsLoss()
        self.optim = AdamW(
            self.model.parameters(),
            lr=self.cfg.trainer.lr,
            weight_decay=self.cfg.trainer.wd,
        )

    @abstractmethod
    def _train_step(self, item: Any) -> torch.Tensor:
        """
        Returns: loss
        """
        ...

    @abstractmethod
    def _eval_step(self, item: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (loss, prediction_probabilities, true_labels)
        """
        ...

    def _get_data_iterable(self, split: str) -> Optional[TorchDataLoader]:
        return self.train_loader if split == "train" else self.val_loader

    def _on_epoch_start(self) -> None:
        """
        Hook called at the start of each epoch.
        """
        self.model.train()

    def _on_training_end(self) -> None:
        """
        Hook called at the end of training.
        """
        ...

    def train(self, run_name: str):
        wandb.init(
            project=self.cfg.project_name, name=run_name, config=self.cfg.model_dump()
        )
        wandb.watch(self.model, log="all", log_freq=100)
        try:
            for epoch in tqdm(
                range(1, self.cfg.trainer.n_epochs + 1), desc="Epoch", colour="green"
            ):
                self._on_epoch_start()

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
                    wandb.log({"train/step_loss": loss.item()})

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
            results = m.compute()
            for name, value in results:
                log_dict[f"{split}/{name}"] = value
            plot_result = m.plot()
            if plot_result:
                name, fig = plot_result
                log_dict[f"{split}/{name}"] = wandb.Image(fig)
                plt.close(fig)

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
        self.optim.step()
        return loss

    def _eval_step(
        self, batch: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def _on_training_end(self):
        self._explain()

    def _explain(self) -> None:
        if (
            not (
                pfc := next(
                    (m for m in self.metrics if isinstance(m, PositiveFrameCollector)),
                    None,
                )
            )
            or not pfc.storage._items
        ):
            logger.warning("Nothing to explain")
            return

        _, data = pfc.storage._items[0]

        explainer = Explainer(
            model=self.model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="binary_classification",
                task_level="graph",
                return_type="raw",
            ),
        )

        with torch.enable_grad():
            x = data.x.clone().float().requires_grad_(True)
            u = data.u.clone().float().requires_grad_(True)
            edge_index = data.edge_index.clone()
            edge_weight = data.edge_weight.clone().float().requires_grad_(True)

            explanation = explainer(
                x=x,
                edge_index=edge_index,
                u=u,
                edge_weight=edge_weight,
            )

        node_mask = explanation.node_mask.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.heatmap(
            node_mask,
            ax=ax,
            cmap="coolwarm",
            cbar=False,
            xticklabels=self.feature_names,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center", fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

        fig.tight_layout()

        wandb.log({"explain/node_feats_heatmap": wandb.Image(fig)})
        plt.close(fig)


class TemporalTrainer(BaseTrainer):
    def _compute_signal_loss_and_last_pred(
        self, signal: Discrete_Signal
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # T = signal.snapshot_count
        # weights = torch.tensor(
        #     [self.cfg.trainer.gamma ** (T - 1 - t) for t in range(T)],
        #     device=self.device,
        # )
        # weights /= weights.sum()

        loss = torch.zeros((), device=self.device)
        valid_sum = 0
        h = None
        last_pred = None

        for t, snapshot in enumerate(signal):
            x = snapshot.x.to(self.device, non_blocking=True).float()
            y = snapshot.y.to(self.device, non_blocking=True).float()
            edge_index = snapshot.edge_index.to(self.device, non_blocking=True).long()
            edge_attr = snapshot.edge_attr.to(self.device, non_blocking=True).float()
            u = snapshot.u.to(self.device, non_blocking=True).float()
            batch = snapshot.batch.to(self.device, non_blocking=True).long()
            mask = snapshot.masks.to(self.device, non_blocking=True).bool()

            out, h = self.model(
                x=x,
                edge_index=edge_index,
                edge_weight=edge_attr,
                edge_attr=edge_attr,
                u=u,
                batch=batch,
                batch_size=snapshot.num_graphs,
                prev_h=h,
            )

            if last_pred is None:
                last_pred = torch.zeros_like(out)

            last_pred[mask] = out[mask]

            loss += (
                F.binary_cross_entropy_with_logits(out, y, reduction="none")
                * mask.unsqueeze(1)
            ).sum()
            valid_sum += mask.sum()

        loss /= valid_sum

        return loss, last_pred

    def _train_step(self, batch: Discrete_Signal) -> torch.Tensor:
        self.optim.zero_grad(set_to_none=True)
        loss, _ = self._compute_signal_loss_and_last_pred(batch)
        loss.backward()
        self.optim.step()
        return loss

    def _eval_step(
        self, batch: Discrete_Signal
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss, out = self._compute_signal_loss_and_last_pred(batch)
        preds_probs = torch.sigmoid(out)
        true_labels = batch[0].y.cpu().long()
        return loss, preds_probs, true_labels
