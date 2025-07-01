import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from loguru import logger
from torch.utils.data.dataloader import DataLoader as TorchDataLoader
from tqdm import tqdm

import wandb
from soccerai.data.converters import create_graph_converter
from soccerai.data.dataset import WorldCup2022Dataset
from soccerai.data.temporal_dataset import TemporalChainsDataset
from soccerai.models.models import build_model
from soccerai.training.trainer_config import Config

NUM_WORKERS = (os.cpu_count() or 1) - 1


def find_best_checkpoint(model_dir: Path) -> Optional[Tuple[str, str]]:
    checkpoint_files = list(model_dir.glob("*.pth"))

    best_run_id: str | None = None
    best_metric: float = np.inf

    found = None
    for ckpt_path in checkpoint_files:
        # example: k5ed88ly_val_loss_0.5446.pth
        # the cktpt name is made of run_id, history key and the history value
        # ["k5ed88ly", "val", "loss", "0.5446"]
        parts = ckpt_path.stem.split("_")
        run_id = parts[0]
        metric_value = float(parts[3])

        if metric_value < best_metric:
            best_run_id = run_id
            best_metric = metric_value
            found = ckpt_path

    if best_run_id is None:
        return None

    return best_run_id, found


def evaluate(model, loader):
    for signal in tqdm(loader):
        masks = torch.tensor(signal.masks, dtype=torch.bool, device="cuda").T
        h = None
        c = None

        preds_per_timestep = torch.empty_like(masks)
        for t, snapshot in enumerate(tqdm(signal)):
            snapshot.to("cuda", non_blocking=True)
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
            preds_per_timestep[t] = out.unsqueeze(-1)


def main(args):
    api = wandb.Api()

    model_dir = Path("checkpoints") / args.name

    best_run_id, model_ckpt_path = find_best_checkpoint(model_dir)

    if best_run_id is None:
        logger.info("no ckpt found.")
        return

    run = api.run(f"soccerai/soccerai/{best_run_id}")

    cfg = Config(**run.config)

    converter = create_graph_converter(cfg.data.connection_mode)
    val_ds = WorldCup2022Dataset(
        split="val",
        root="soccerai/data/resources",
        converter=converter,
        cfg=cfg.data,
        random_state=cfg.seed,
        force_reload=True,
    )

    model = build_model(cfg, val_ds)
    model.load_state_dict(torch.load(model_ckpt_path))
    model.eval()
    model.to("cuda")

    loader_kwargs = dict(
        batch_size=cfg.trainer.bs,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    val_ds = TemporalChainsDataset.from_worldcup_dataset(val_ds)

    val_loader = TorchDataLoader(
        val_ds,
        collate_fn=TemporalChainsDataset.collate,
        shuffle=False,
        **loader_kwargs,
    )

    evaluate(model, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        help="the name of the best model to choose",
    )
    parser.add_argument("--device")
    args = parser.parse_args()
    main(args)
