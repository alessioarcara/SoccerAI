import copy
from abc import ABC
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from torch_geometric.explain import Explainer, GNNExplainer

import wandb
from soccerai.training.metrics import Collector
from soccerai.training.trainer_config import Config
from soccerai.training.utils import (
    fig_to_numpy,
    plot_average_feature_importance,
    plot_player_feature_importance,
)


class Callback(ABC):
    def on_train_end(self, trainer): ...
    def on_eval_end(self, trainer): ...


def build_callbacks(cfg: Config) -> List[Callback]:
    callbacks: List[Callback] = []

    if cfg.trainer.early_stopping_callback:
        callbacks.append(
            EarlyStoppingCallback(**cfg.trainer.early_stopping_callback.model_dump())
        )

    if cfg.trainer.model_saving_callback:
        callbacks.append(
            ModelSavingCallback(
                **cfg.trainer.model_saving_callback.model_dump(),
                model_name=cfg.run_name,
            )
        )

    if not cfg.model.use_temporal:
        callbacks.append(ExplainerCallback())

    return callbacks


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


class ModelMonitorCallback(Callback):
    def __init__(self, history_key: str, minimize: bool):
        super().__init__()
        self.history_key = history_key
        self.best = float("inf") if minimize else 0.0
        self.minimize = minimize

    def on_eval_end(self, trainer) -> bool:
        curr = trainer.history.get(self.history_key)
        if curr is None:
            logger.warning(f"{self.history_key} not found in history; skipping.")
            return False

        improved = curr < self.best if self.minimize else curr > self.best
        if improved:
            self.best = curr
            return True
        return False


class EarlyStoppingCallback(ModelMonitorCallback):
    def __init__(self, history_key: str, minimize: bool, patience: int):
        super().__init__(history_key, minimize)
        self.patience = patience
        self.counter = 0
        self.should_stop = False

    def on_eval_end(self, trainer):
        improved = super().on_eval_end(trainer)

        if improved:
            self.counter = 0
        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.should_stop = True


class ModelSavingCallback(ModelMonitorCallback):
    def __init__(self, history_key: str, minimize: bool, model_name: str):
        super().__init__(history_key, minimize)
        self.out_dir = Path("checkpoints") / model_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

    def on_eval_end(self, trainer):
        improved = super().on_eval_end(trainer)

        if improved:
            self.best_model = copy.deepcopy(trainer.model.state_dict())

    def on_train_end(self, trainer):
        checkpoint_name = f"{wandb.run.id}_{self.history_key}_{self.best:0.4f}.pth"
        checkpoint_path = self.out_dir / checkpoint_name
        torch.save(self.best_model, checkpoint_path)
        logger.info(f"Saved model checkpoint to {checkpoint_path}")

        artifact = wandb.Artifact(name=self.model_name, type="model")
        artifact.add_file(str(checkpoint_path), name=checkpoint_name)
        wandb.run.log_artifact(artifact)
