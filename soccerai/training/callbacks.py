from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
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

        class ModelForExplainer(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model

            def forward(
                self,
                x,
                edge_index,
                u=None,
                edge_weight=None,
                batch=None,
                batch_size=None,
                prev_h=None,
            ):
                out, _ = self.base_model(
                    x=x,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    edge_attr=None,
                    u=u,
                    batch=batch,
                    batch_size=batch_size,
                    prev_h=prev_h,
                )
                return out

        explainer = Explainer(
            model=ModelForExplainer(trainer.model).to(trainer.device),
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

        jersey_idx = None
        if "jerseyNum" in trainer.feature_names:
            jersey_idx = trainer.feature_names.index("jerseyNum")

        for c in chain_collectors:
            pos_type = "TP" if c.target_label == 1 else "FP"
            _, (_, best_chain) = c.highest_confidence_chain

            # List of (score, (probs_seq, List[Data]))
            if not best_chain:
                continue

            h = None
            frames_np = []

            for frame_idx, data in enumerate(best_chain):
                d = data.clone().to(trainer.device)

                N = d.x.size(0)
                if hasattr(d, "jersey_numbers"):
                    jersey_numbers = d.jersey_numbers.cpu().numpy()
                elif jersey_idx is not None:
                    jersey_numbers = d.x[:, jersey_idx].long().cpu().numpy()
                else:
                    jersey_numbers = np.arange(N)

                d.x = d.x.float().requires_grad_(True)
                d.u = d.u.float().requires_grad_(True)
                d.edge_weight = d.edge_weight.float()
                batch_idx = torch.zeros(N, dtype=torch.long, device=trainer.device)

                with torch.enable_grad():
                    explanation = explainer(
                        x=d.x,
                        edge_index=d.edge_index,
                        u=d.u,
                        edge_weight=d.edge_weight,
                        batch=batch_idx,
                        batch_size=1,
                        prev_h=h,
                    )

                mask = explanation.node_mask.detach().cpu().numpy()

                _, h = trainer.model(
                    x=d.x,
                    edge_index=d.edge_index,
                    edge_weight=d.edge_weight,
                    edge_attr=None,
                    u=d.u,
                    batch=batch_idx,
                    batch_size=1,
                    prev_h=h,
                )
                h = h.detach()

                fig = plot_player_feature_importance(
                    mask,
                    jersey_numbers,
                    trainer.feature_names,
                    positive_type=pos_type,
                    frame_idx=frame_idx + 1,
                )
                frames_np.append(fig_to_numpy(fig))
                plt.close(fig)

            if not frames_np:
                logger.warning(f"No frames to explain for chain {pos_type}")
                continue

            video_np = np.stack(frames_np).transpose(0, 3, 1, 2)
            wandb.log(
                {
                    f"explain/temporal_best_chain_label_{c.target_label}": wandb.Video(
                        video_np, fps=1, format="mp4"
                    )
                }
            )
