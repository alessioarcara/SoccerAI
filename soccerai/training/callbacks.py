from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from torch_geometric.explain import Explainer, GNNExplainer

import wandb
from soccerai.training.metrics import ChainCollector, Collector
from soccerai.training.utils import (
    fig_to_numpy,
    plot_average_feature_importance,
    plot_player_feature_importance,
)


class Callback(ABC):
    def on_train_end(self, trainer): ...


class ExplainerCallback(Callback):
    def on_train_end(self, trainer):
        collectors = [m for m in trainer.metrics if isinstance(m, Collector)]
        if not collectors or all(len(c) == 0 for c in collectors):
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

            if num_frames == 0:
                continue

            node_masks, figs_np = [], []
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
                    figs_np.append(
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

            video_np = np.stack(figs_np).transpose(0, 3, 1, 2)
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
                num_frames,
            )
            wandb.log(
                {f"explain/{pos_type}_average_feature_importance": wandb.Image(fig)}
            )
            plt.close(fig)


class BestChainExplainerCallback(Callback):
    def on_train_end(self, trainer):
        chain_collectors = [m for m in trainer.metrics if isinstance(m, ChainCollector)]
        if not chain_collectors:
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

        for c in chain_collectors:
            entries = c.frames  # List[(score, (probs_seq, frames_list))]
            if not entries:
                continue

            best_confidence, (_, best_chain) = entries[0]

            frames_np = []
            for t, data in enumerate(best_chain):
                d = data.clone().to(trainer.device)
                d.x = d.x.float().requires_grad_(True)
                d.u = d.u.float().requires_grad_(True)
                d.edge_weight = d.edge_weight.float()

                with torch.enable_grad():
                    explanation = explainer(
                        x=d.x,
                        edge_index=d.edge_index,
                        u=d.u,
                        edge_weight=d.edge_weight,
                    )

                mask = explanation.node_mask.detach().cpu().numpy()
                fig = plot_player_feature_importance(
                    mask,
                    d.jersey_numbers.cpu().numpy(),
                    trainer.feature_names,
                    title=f"Best Chain — Frame {t + 1}",
                    confidence=best_confidence,
                )
                frames_np.append(fig_to_numpy(fig))
                plt.close(fig)

            video_np = np.stack(frames_np).transpose(0, 3, 1, 2)
            wandb.log(
                {
                    f"explain/temporal_best_chain_label_{c.target_label}": wandb.Video(
                        video_np, fps=1, format="mp4"
                    )
                }
            )
