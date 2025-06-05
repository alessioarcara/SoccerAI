import argparse
import os

import torch
from loguru import logger
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader, PrefetchLoader

from soccerai.data.converters import create_graph_converter
from soccerai.data.dataset import WorldCup2022Dataset
from soccerai.data.temporal_dataset import TemporalGraphDataset
from soccerai.models.tgcn import RecurrentGCN
from soccerai.training.metrics import (
    BinaryConfusionMatrix,
    BinaryPrecisionRecallCurve,
    PositiveFrameCollector,
)
from soccerai.training.trainer import Trainer
from soccerai.training.trainer_config import build_cfg
from soccerai.training.utils import fix_random, temporal_collate_fn

NUM_WORKERS = (os.cpu_count() or 1) - 1
CONFIG_PATH = "configs/base.yaml"
torch.set_float32_matmul_precision("high")


def main(args):
    cfg = build_cfg(CONFIG_PATH)
    fix_random(cfg.seed)
    converter = create_graph_converter(cfg.connection_mode)

    train_ds = WorldCup2022Dataset(
        root="soccerai/data/resources",
        converter=converter,
        force_reload=args.reload,
        split="train",
        cfg=cfg,
        # transform=Compose([RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.5)]),
    )
    val_ds = WorldCup2022Dataset(
        root="soccerai/data/resources",
        converter=converter,
        split="val",
        cfg=cfg,
    )

    logger.success(
        "Datasets loaded successfully → train graphs: {}, val graphs: {}",
        len(train_ds),
        len(val_ds),
    )
    feature_names = train_ds.feature_names
    if cfg.use_temporal_sequences:
        train_ds = TemporalGraphDataset(train_ds.to_temporal_iterator())
        val_ds = TemporalGraphDataset(val_ds.to_temporal_iterator())
        train_loader = TorchDataLoader(
            train_ds, batch_size=cfg.bs, shuffle=True, collate_fn=temporal_collate_fn
        )
        val_loader = TorchDataLoader(
            val_ds, batch_size=cfg.bs, shuffle=True, collate_fn=temporal_collate_fn
        )
    else:
        common_loader_kwargs = dict(
            batch_size=cfg.bs,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
        )
        train_loader = PrefetchLoader(
            DataLoader(
                train_ds,
                shuffle=True,
                **common_loader_kwargs,
            ),
        )
        val_loader = PrefetchLoader(
            DataLoader(
                val_ds,
                shuffle=False,
                **common_loader_kwargs,
            ),
        )
    model = RecurrentGCN(train_ds[0].features[0].shape[1], cfg.dim, 1, cfg.bs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        feature_names=feature_names,
        metrics=[
            BinaryConfusionMatrix(),
            BinaryPrecisionRecallCurve(),
            PositiveFrameCollector(),
        ],
    )
    trainer.train(args.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reload",
        action="store_true",
        help="If set, forces the dataset to be re-created",
    )
    parser.add_argument(
        "--name", type=str, help="The name of the W&B run", default="debug"
    )
    args = parser.parse_args()
    main(args)
