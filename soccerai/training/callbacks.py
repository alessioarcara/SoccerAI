from abc import ABC
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from torch_geometric.explain import Explainer, GNNExplainer

import wandb
from soccerai.training.metrics import Collector
from soccerai.training.utils import (
    fig_to_numpy,
    plot_average_feature_importance,
    plot_player_feature_importance,
)


class Callback(ABC):
    def on_train_end(self, trainer): ...
    def on_eval_end(self, trainer): ...


class ExplainerCallback(Callback):
    def on_train_end(self, trainer):
        collectors = [
            m for m in trainer.metrics if isinstance(m, Collector) and len(m) > 0
        ]
        if not collectors:
            logger.warning("Nothing to explain")
            return

        explainer = Explainer(
            model=trainer.model,
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

        for c in collectors:
            pos_type = "TP" if c.target_label == 1 else "FP"

            collected_frames = c.frames
            num_frames = len(c)

            node_masks, figs = [], []
            for i, (_, data) in enumerate(collected_frames):
                data = data.clone()
                data.x = data.x.float().requires_grad_(True)
                data.u = data.u.float().requires_grad_(True)

                with torch.enable_grad():
                    explanation = explainer(
                        x=data.x,
                        edge_index=data.edge_index,
                        u=data.u,
                        edge_weight=data.edge_weight,
                    )

                    node_masks.append(explanation.node_mask.detach().cpu().numpy())
                    figs.append(
                        fig_to_numpy(
                            plot_player_feature_importance(
                                node_masks[i],
                                data.jersey_numbers,
                                trainer.feature_names,
                                pos_type,
                                i + 1,
                            )
                        )
                    )

            figs_np = np.stack(figs).transpose(0, 3, 1, 2)
            wandb.log(
                {f"explain/video_{pos_type}": wandb.Video(figs_np, fps=1, format="mp4")}
            )

            fig = plot_average_feature_importance(
                node_masks,
                trainer.feature_names,
                num_frames,
            )
            wandb.log(
                {f"explain/{pos_type}_average_feature_importance": wandb.Image(fig)}
            )
            plt.close(fig)


class ValidationCheckpointCallback(Callback):
    """
    Every time validation ends, if the monitored metric improves,
    updates the best value and logs the improvement.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        artifact_name: str = "best_val_model",
        artifact_type: str = "model",
    ):
        self.monitor = monitor
        self.best = float("inf") if mode == "min" else -float("inf")
        self.mode = mode
        self.artifact_name = artifact_name
        self.artifact_type = artifact_type

    def on_eval_end(self, trainer):
        current = trainer.metrics.get(self.monitor)
        if current is None:
            logger.warning(
                f"Metric {self.monitor} not found in trainer.metrics; skipping update."
            )
            return

        improved = current < self.best if self.mode == "min" else current > self.best
        if not improved:
            logger.debug(
                f"No improvement in {self.monitor}: current={current:.4f} vs best={self.best:.4f}."
            )
            return

        old_best = self.best
        self.best = current
        logger.info(
            f"Validation metric {self.monitor} improved from {old_best:.4f} to {self.best:.4f}."
        )


class FinalModelSaverCallback(Callback):
    def __init__(
        self,
        model_name: str,
        artifact_name: str = "final_model",
        out_dir: str = "checkpoints",
    ):
        self.model_name = model_name
        self.artifact_name = artifact_name
        self.out_dir = Path(out_dir) / model_name
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def on_train_end(self, trainer):
        run_id = wandb.run.id if wandb.run else ""
        filename = self.out_dir / f"{self.artifact_name}_{run_id}.pth"

        torch.save(trainer.model.state_dict(), str(filename))
        logger.info("Saved final model checkpoint to {}", filename)

        artifact = wandb.Artifact(
            name=f"{self.artifact_name}-{self.model_name}-{run_id}",
            type="model",
        )
        artifact.add_file(str(filename))
        wandb.run.log_artifact(artifact)
        logger.success(
            "Logged final artifact '{}' for model '{}' to W&B run {}",
            artifact.name,
            self.model_name,
            run_id,
        )
