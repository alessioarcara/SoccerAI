import argparse
import os

from loguru import logger
from torch_geometric.loader import DataLoader, PrefetchLoader

from soccerai.data.converters import ConnectionMode, ShotPredictionGraphConverter
from soccerai.data.dataset import WorldCup2022Dataset, split_dataset
from soccerai.models import GCN
from soccerai.training.metrics import BinaryAccuracy
from soccerai.training.trainer import Trainer
from soccerai.training.trainer_config import build_cfg

NUM_WORKERS = (os.cpu_count() or 1) - 1
CONFIG_PATH = "configs/example.yaml"

cfg = build_cfg(CONFIG_PATH)
common_loader_kwargs = dict(
    batch_size=cfg.bs,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)


def main(args):
    converter = ShotPredictionGraphConverter(ConnectionMode.FULLY_CONNECTED)

    dataset = WorldCup2022Dataset(
        root="soccerai/data/resources",
        converter=converter,
        force_reload=args.reload,
    )
    logger.success(f"Dataset loaded successfully. Number of graphs: {len(dataset)}")

    train_dataset, val_dataset = split_dataset(dataset, cfg.val_ratio)

    train_loader = PrefetchLoader(
        DataLoader(
            train_dataset,
            shuffle=True,
            **common_loader_kwargs,
        ),
    )
    val_loader = PrefetchLoader(
        DataLoader(
            val_dataset,
            shuffle=False,
            **common_loader_kwargs,
        ),
    )
    model = GCN(dataset.num_node_features, 256, 1)

    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cuda",
        metrics=[BinaryAccuracy()],
    )
    trainer.train("debug")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reload",
        action="store_true",
        help="If set, forces the dataset to be re-created",
    )
    args = parser.parse_args()
    main(args)
