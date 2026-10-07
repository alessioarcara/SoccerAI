import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data.dataloader import DataLoader as TorchDataLoader
from tqdm import tqdm

from soccerai.data.converters import create_graph_converter
from soccerai.data.dataset import WorldCup2022Dataset
from soccerai.data.temporal_dataset import TemporalChainsDataset
from soccerai.models.models import build_model
from soccerai.training.checkpoint import (
    checkpoint_config,
    find_best_checkpoint,
    load_checkpoint,
)
from soccerai.training.metrics import BinaryConfusionMatrix, BinaryPrecisionRecallCurve
from soccerai.training.trainer_config import Config, MetricsConfig
from soccerai.training.utils import fix_random

NUM_WORKERS = (os.cpu_count() or 1) - 1


def load_config_from_wandb(run_id: str) -> Config:
    """Fallback for checkpoints that predate the self-contained format."""
    import wandb

    logger.info("Checkpoint has no config: fetching run {} from W&B", run_id)
    run = wandb.Api().run(f"soccerai/soccerai/{run_id}")
    return Config(**{k: v for k, v in run.config.items() if not k.startswith("_")})


def evaluate(
    model: nn.Module,
    loader: TorchDataLoader,
    device: torch.device,
    threshold: float,
    fbeta: float,
):
    model.eval()

    cm = BinaryConfusionMatrix(MetricsConfig(thr=threshold, fbeta=fbeta), mode="both")
    ap = BinaryPrecisionRecallCurve()

    cm.reset()
    ap.reset()

    with torch.inference_mode():
        for signal in tqdm(loader, desc="signals", leave=False):
            h = c = None
            last_preds: Optional[torch.Tensor] = None

            for snapshot in signal:
                snapshot = snapshot.to(device, non_blocking=True)
                mask = snapshot.masks.bool()

                out, h, c = model(
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

                if last_preds is None:
                    last_preds = torch.zeros_like(out)

                last_preds[mask] = out[mask]

            assert last_preds is not None

            preds_probs = torch.sigmoid(last_preds)
            true_labels = torch.tensor(signal.targets[0], device=device)

            cm.update(preds_probs, true_labels, snapshot)
            ap.update(preds_probs, true_labels, snapshot)

        cm_results = cm.compute()
        ap_results = ap.compute()

        print("Evaluation results:")
        for name, value in cm_results + ap_results:
            print(f"{name}: {value:.4f}")

        cm_np = cm.cm.cpu().numpy()
        tn, fp = cm_np[0, 0], cm_np[0, 1]
        fn, tp = cm_np[1, 0], cm_np[1, 1]

        print("\n" + "Confusion Matrix".center(35))
        print("\n" + " " * 14 + "Predicted")
        print(" " * 15 + "0     1")
        print(" " * 11 + "┌─────┬─────┐")
        print("  True   0 │ {:>3} │ {:>3} │".format(tn, fp))
        print("  Label    ├─────┼─────┤")
        print("         1 │ {:>3} │ {:>3} │".format(fn, tp))
        print(" " * 11 + "└─────┴─────┘\n")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: {}", device)

    model_dir = Path("checkpoints") / args.name
    best = find_best_checkpoint(model_dir)
    if best is None:
        logger.warning("No checkpoints were found in {}", model_dir)
        raise SystemExit(1)

    best_run_id, ckpt_path = best
    logger.info("Evaluating checkpoint {}", ckpt_path)

    payload = load_checkpoint(ckpt_path)
    cfg = checkpoint_config(payload)
    if cfg is None:
        cfg = load_config_from_wandb(best_run_id)
    fix_random(cfg.seed)

    converter = create_graph_converter(
        cfg.data.connection_mode, cfg.data.edge_length_scale
    )
    ds = WorldCup2022Dataset(
        split="val",
        root="soccerai/data/resources",
        converter=converter,
        cfg=cfg.data,
        random_state=cfg.seed,
    )

    model = build_model(cfg, ds)

    chain_ds = TemporalChainsDataset.from_worldcup_dataset(ds, cfg.data.max_chain_len)

    loader = TorchDataLoader(
        chain_ds,
        collate_fn=TemporalChainsDataset.collate,
        shuffle=False,
        batch_size=cfg.trainer.bs,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
        prefetch_factor=4,
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)

    evaluate(model, loader, device, args.threshold, args.fbeta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        help="Experiment directory under 'checkpoints/'",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument("--fbeta", type=float, default=1.0)

    args = parser.parse_args()
    main(args)
