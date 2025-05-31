from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from loguru import logger
from torch.optim import AdamW
from torch_geometric.data import Batch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import wandb
from soccerai.training.metrics import Metric, PositiveFrameCollector
from soccerai.training.trainer_config import TrainerConfig


class Trainer:
    def __init__(
        self,
        cfg: TrainerConfig,
        model: nn.Module,
        train_loader: DataLoader,
        device: str,
        feature_names: List[str],
        val_loader: Optional[DataLoader] = None,
        metrics: List[Metric] = [],
    ):
        self.cfg = cfg
        self.model: nn.Module = torch.compile(model.to(device))  # type: ignore
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.metrics = metrics
        self.feature_names = feature_names

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

            self._explain()

        finally:
            wandb.finish()

    def _train_step(self, batch: Batch) -> torch.Tensor:
        self.optim.zero_grad(set_to_none=True)
        out = self.model(
            x=batch.x,
            edge_index=batch.edge_index,
            batch=batch.batch,
            batch_size=batch.num_graphs,
        )
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
        for m in self.metrics:
            m.reset()

        batch: Batch
        for batch in tqdm(
            loader,
            total=len(loader),
            desc=f"Evaluating {split}",
            leave=False,
            colour="red",
        ):
            out = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                batch=batch.batch,
                batch_size=batch.num_graphs,
            )
            batch_loss = self.criterion(out, batch.y)
            total_loss += batch_loss.item()

            preds_probs = torch.sigmoid(out)
            true_labels = batch.y.cpu().long()

            for m in self.metrics:
                m.update(preds_probs, true_labels, batch)

        mean_loss = total_loss / len(loader)

        # Log metrics and plots to  W&B
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
            explanation = explainer(
                x=data.x.clone().float().requires_grad_(True),
                edge_index=data.edge_index.clone(),
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
