import argparse
import os

import torch
from loguru import logger
from torch_geometric.loader import DataLoader, PrefetchLoader

from soccerai.data.converters import create_graph_converter
from soccerai.data.dataset import WorldCup2022Dataset
from soccerai.models import GCN
from soccerai.training.metrics import (
    BinaryConfusionMatrix,
    BinaryPrecisionRecallCurve,
    PositiveFrameCollector,
)
from soccerai.training.trainer import Trainer
from soccerai.training.trainer_config import build_cfg
from soccerai.training.utils import fix_random

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

    _ = train_ds.to_temporal_graph()

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
    model = GCN(train_ds.num_node_features, cfg.dim, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        feature_names=train_ds.feature_names,
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
