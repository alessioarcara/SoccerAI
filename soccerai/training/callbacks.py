from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from torch_geometric.explain import Explainer, GNNExplainer

import wandb
from soccerai.training.metrics import FrameCollector
from soccerai.training.utils import (
    fig_to_numpy,
    plot_average_feature_importance,
    plot_player_feature_importance,
)


class Callback(ABC):
    def on_train_end(self, trainer): ...


class ExplainerCallback(Callback):
    def on_train_end(self, trainer):
        collectors = [m for m in trainer.metrics if isinstance(m, FrameCollector)]
        if not collectors or all(len(c.storage._items) == 0 for c in collectors):
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

            entries = c.storage.get_all_entries()
            num_entries = len(entries)

            if num_entries == 0:
                continue

            node_masks, frames = [], []
            for i, (_, data) in enumerate(entries):
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
                    frames.append(
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

            video_np = np.stack(frames).transpose(0, 3, 1, 2)
            wandb.log(
                {
                    f"explain/video_{pos_type}": wandb.Video(
                        video_np, fps=1, format="mp4"
                    )
                }
            )

            fig = plot_average_feature_importance(
                node_masks,
                trainer.feature_names,
                num_entries,
            )
            wandb.log(
                {f"explain/{pos_type}_average_feature_importance": wandb.Image(fig)}
            )
            plt.close(fig)
