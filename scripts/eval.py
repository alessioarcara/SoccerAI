import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data.dataloader import DataLoader as TorchDataLoader
from tqdm import tqdm

import wandb
import wandb.errors
from soccerai.data.converters import create_graph_converter
from soccerai.data.dataset import WorldCup2022Dataset
from soccerai.data.temporal_dataset import TemporalChainsDataset
from soccerai.models.models import build_model
from soccerai.training.metrics import BinaryConfusionMatrix
from soccerai.training.trainer_config import Config

NUM_WORKERS = (os.cpu_count() or 1) - 1


def find_best_checkpoint(model_dir: Path) -> Optional[Tuple[str, Path]]:
    """Return the `(wandb_run_id, checkpoint_path)` with the lowest metric.

    The checkpoint filenames are expected to follow
    `{run_id}_{metric_key}_{metric_name}_{metric_value}.pth`.
    """
    ckpts = list(model_dir.glob("*.pth"))
    if not ckpts:
        return None

    best: Optional[Tuple[str, Path, float]] = None
    for path in ckpts:
        parts = path.stem.split("_")
        run_id = parts[0]
        metric_value = float(parts[3])
        if best is None or metric_value < best[2]:
            best = (run_id, path, metric_value)

    if best is None:
        return None

    wandb_id, ckpt_path, _ = best
    return wandb_id, ckpt_path


def evaluate(
    model: nn.Module, loader: TorchDataLoader, device: torch.device, cfg: Config
):
    model.eval()
    m = BinaryConfusionMatrix(cfg.metrics)

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
            preds_probs = torch.sigmoid(out[mask])
            true_labels = snapshot.y[mask].cpu().long()
            m.update(preds_probs, true_labels, snapshot)

        print(m.compute())


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: {}", device)

    model_dir = Path("checkpoints") / args.name
    best = find_best_checkpoint(model_dir)
    if best is None:
        logger.warning("No checkpoints were found in {}", model_dir)
        raise SystemExit(1)

    best_run_id, ckpt_path = best

    # load W&B config
    api = wandb.Api()
    try:
        run = api.run(f"soccerai/soccerai/{best_run_id}")
    except wandb.errors.CommError:
        logger.exception("Failed to fetch run {} from W&B", best_run_id)
        raise SystemExit(1)

    cfg = Config(**run.config)

    converter = create_graph_converter(cfg.data.connection_mode)
    ds = WorldCup2022Dataset(
        split="val",
        root="soccerai/data/resources",
        converter=converter,
        cfg=cfg.data,
        random_state=cfg.seed,
        force_reload=True,
    )

    model = build_model(cfg, ds)

    chain_ds = TemporalChainsDataset.from_worldcup_dataset(ds)

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
    model.load_state_dict(torch.load(ckpt_path))
    model.to(device)

    evaluate(model, loader, device, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        help="Experiment directory under 'checkpoints/'",
    )

    args = parser.parse_args()
    main(args)
