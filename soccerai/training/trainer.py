from typing import List, Optional

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import wandb
from soccerai.training.metrics import Metric
from soccerai.training.trainer_config import TrainerConfig


class Trainer:
    def __init__(
        self,
        cfg: TrainerConfig,
        model: nn.Module,
        train_loader: DataLoader,
        device: str,
        val_loader: Optional[DataLoader] = None,
        metrics: List[Metric] = [],
    ):
        self.cfg = cfg
        self.model: nn.Module = torch.compile(model.to(device))  # type: ignore
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.metrics = metrics

        self.optim = AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        self.criterion = nn.BCEWithLogitsLoss()

    def train(self, run_name: str):
        wandb.init(
            project=self.cfg.project_name, name=run_name, config=self.cfg.model_dump()
        )
        try:
            for epoch in tqdm(
                range(1, self.cfg.n_epochs + 1), desc="Epoch", colour="green"
            ):
                self.model.train()
                for batch in tqdm(
                    self.train_loader,
                    total=len(self.train_loader),
                    desc=f"Epoch {epoch} Batches",
                    leave=False,
                    colour="blue",
                ):
                    batch_loss = self._train_step(batch)
                    wandb.log({"train/batch_loss": batch_loss.item()})

                if epoch % self.cfg.eval_rate == 0:
                    self.eval("train")
                    if self.val_loader:
                        self.eval("val")

        finally:
            wandb.finish()

    def _train_step(self, batch: Batch) -> torch.Tensor:
        self.optim.zero_grad(set_to_none=True)
        out = self.model(batch)
        loss: torch.Tensor = self.criterion(out, batch.y)
        loss.backward()
        self.optim.step()
        return loss

    @torch.inference_mode()
    def eval(self, split: str) -> None:
        self.model.eval()
        loader = self.val_loader if split == "val" else self.train_loader

        assert loader is not None

        total_loss = 0.0
        for metric in self.metrics:
            metric.reset()
        conf_matrix = torch.zeros(2, 2, dtype=torch.int, device="cpu")

        batch: Batch
        for batch in tqdm(
            loader,
            total=len(loader),
            desc=f"Evaluating {split}",
            leave=False,
            colour="red",
        ):
            out = self.model(batch)
            batch_loss = self.criterion(out, batch.y)
            total_loss += batch_loss

            preds_probs = torch.sigmoid(out)
            preds_labels = (preds_probs.cpu() > 0.5).long()
            true_labels = batch.y.cpu().long()

            for metric in self.metrics:
                metric.update(preds_probs, batch.y)

            true_labels_flat = true_labels.view(-1)
            preds_labels_flat = preds_labels.view(-1)
            for i in range(true_labels_flat.shape[0]):
                conf_matrix[true_labels_flat[i], preds_labels_flat[i]] += 1

        mean_loss = total_loss / len(loader)

        log_dict = {f"{split}/total_loss": mean_loss}
        for metric in self.metrics:
            metric_name = metric.__class__.__name__
            log_dict[f"{split}/{metric_name}"] = metric.compute()

        if split == "val":
            conf_matrix_df = pl.DataFrame(conf_matrix.numpy())
            fig, ax = plt.subplots(figsize=(8, 6))
            heatmap = sns.heatmap(
                conf_matrix_df,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
            )
            plt.title(f"{split.capitalize()} Confusion Matrix")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            log_dict[f"{split}/confusion_matrix"] = wandb.Image(heatmap.get_figure())
            plt.close(fig)

        wandb.log(log_dict)
