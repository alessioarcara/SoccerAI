import random
from abc import ABC, abstractmethod
from typing import Any, Collection, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from loguru import logger
from torch.optim import AdamW
from torch_geometric.data import Batch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.loader import DataLoader
from torch_geometric_temporal.signal import Discrete_Signal
from tqdm import tqdm

import wandb
from soccerai.training.metrics import FrameCollector, Metric
from soccerai.training.trainer_config import Config
from soccerai.training.utils import (
    make_heatmap_video_opencv,
    plot_feature_importance_distribution,
)

Signals = List[Discrete_Signal]


class BaseTrainer(ABC):
    def __init__(
        self,
        cfg: Config,
        model: nn.Module,
        device: str,
        metrics: List[Metric] = [],
    ):
        self.cfg = cfg
        # self.model: nn.Module = torch.compile(model.to(device))  # type: ignore
        self.model = model.to(device)
        self.device = device
        self.metrics = metrics
        self.optim = AdamW(
            self.model.parameters(), lr=cfg.trainer.lr, weight_decay=cfg.trainer.wd
        )
        self.criterion = nn.BCEWithLogitsLoss()

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

    @abstractmethod
    def _get_data_iterable(self, split: str) -> Optional[Collection[Any]]: ...

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
    def __init__(
        self,
        cfg: Config,
        model: nn.Module,
        train_loader: DataLoader,
        device: str,
        feature_names: List[str],
        val_loader: Optional[DataLoader] = None,
        metrics: List[Metric] = [],
    ):
        super().__init__(cfg, model, device, metrics)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.feature_names = feature_names

    def _get_data_iterable(self, split: str) -> Optional[Collection[Any]]:
        return self.train_loader if split == "train" else self.val_loader

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
        collectors = [m for m in self.metrics if isinstance(m, FrameCollector)]
        if not collectors or all(len(c.storage._items) == 0 for c in collectors):
            logger.warning("Nothing to explain")
            return

        for coll in collectors:
            label = coll.target_label
            entries = coll.storage.get_all_entries()
            desired_n = coll.storage.k
            actual_n = min(desired_n, len(entries))

            if actual_n == 0:
                continue

            entries = entries[:actual_n]

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

            if actual_n == 1:
                score, (batch, idx) = entries[0]
                data = batch.to_data_list()[idx]
                with torch.enable_grad():
                    exp = explainer(
                        x=data.x.clone().float().requires_grad_(True),
                        edge_index=data.edge_index,
                        u=data.u.clone().float().requires_grad_(True),
                        edge_weight=data.edge_weight.clone()
                        .float()
                        .requires_grad_(True),
                    )
                single_mask = exp.node_mask.detach().cpu().numpy()

                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    single_mask,
                    ax=ax,
                    cmap="coolwarm",
                    cbar=False,
                    xticklabels=self.feature_names,
                    yticklabels=[str(int(n)) for n in data.jersey_num.cpu().numpy()],
                )
                ax.set_title(f"Single frame (score={score:.3f})", fontsize=14)
                ax.tick_params(axis="x", rotation=90)
                plt.tight_layout()

                wandb.log({f"explain/single_label_{label}": wandb.Image(fig)})
                plt.close(fig)
                continue

            node_masks = []
            jerseys_list = []

            for _, (batch, idx) in entries:
                data = batch.to_data_list()[idx]
                jersey_nums = [str(int(n)) for n in data.jersey_num.cpu().numpy()]
                jerseys_list.append(jersey_nums)

                with torch.enable_grad():
                    exp = explainer(
                        x=data.x.clone().float().requires_grad_(True),
                        edge_index=data.edge_index,
                        u=data.u.clone().float().requires_grad_(True),
                        edge_weight=data.edge_weight.clone()
                        .float()
                        .requires_grad_(True),
                    )
                node_masks.append(exp.node_mask.detach().cpu().numpy())

            key, fig = plot_feature_importance_distribution(
                node_masks,
                self.feature_names,
                label_name=("TruePositive" if label == 1 else "FalsePositive"),
                actual_n=actual_n,
            )
            wandb.log({key: wandb.Image(fig)})
            plt.close(fig)

            video_path = make_heatmap_video_opencv(
                node_masks,
                self.feature_names,
                jerseys_list,
                label_name=("TP" if label == 1 else "FP"),
                fps=1,
            )
            wandb.log(
                {
                    f"explain/video_{'TP' if label == 1 else 'FP'}": wandb.Video(
                        video_path, fps=1
                    )
                }
            )


class TemporalTrainer(BaseTrainer):
    def __init__(
        self,
        cfg: Config,
        model: nn.Module,
        train_signals: Signals,
        device: str,
        val_signals: Optional[Signals] = None,
        metrics: List[Metric] = [],
    ) -> None:
        super().__init__(cfg, model, device, metrics)
        self.train_signals = train_signals
        self.val_signals = val_signals

    def _get_data_iterable(self, split: str) -> Optional[Collection[Any]]:
        return self.train_signals if split == "train" else self.val_signals

    def _compute_signal_loss_and_last_pred(
        self, signal: Discrete_Signal
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T = signal.snapshot_count
        weights = torch.tensor(
            [self.cfg.trainer.gamma ** (T - 1 - t) for t in range(T)],
            device=self.device,
        )
        weights /= weights.sum()

        loss = torch.scalar_tensor(0.0, device=self.device)
        h = None

        for t, snapshot in enumerate(signal):
            x = snapshot.x.to(self.device, non_blocking=True)
            y = snapshot.y.to(self.device, non_blocking=True)
            edge_index = snapshot.edge_index.to(self.device, non_blocking=True)
            edge_attr = snapshot.edge_attr.to(self.device, non_blocking=True)
            u = snapshot.u.to(self.device, non_blocking=True)

            out, h = self.model(
                x=x,
                edge_index=edge_index,
                edge_weight=edge_attr,
                edge_attr=edge_attr,
                u=u,
                prev_h=h,
            )

            loss += weights[t] * self.criterion(out, y)

        return loss, out

    def _train_step(self, signal: Discrete_Signal) -> torch.Tensor:
        self.optim.zero_grad(set_to_none=True)
        loss, _ = self._compute_signal_loss_and_last_pred(signal)
        loss.backward()
        self.optim.step()
        return loss

    def _eval_step(
        self, signal: Discrete_Signal
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss, out = self._compute_signal_loss_and_last_pred(signal)
        preds_probs = torch.sigmoid(out)
        true_labels = signal[0].y.cpu().long()
        return loss, preds_probs, true_labels

    def _on_epoch_start(self):
        train_iterable = self._get_data_iterable("train")
        assert train_iterable is not None
        random.shuffle(train_iterable)
