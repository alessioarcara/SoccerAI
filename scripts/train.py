import argparse
import os

from loguru import logger
from torch_geometric.data import DataLoader

from soccerai.data.converters import ConnectionMode, ShotPredictionGraphConverter
from soccerai.data.dataset import WorldCup2022Dataset
from soccerai.models import GCN
from soccerai.training.trainer import Trainer
from soccerai.training.trainer_config import build_cfg


def main(args):
    converter = ShotPredictionGraphConverter(ConnectionMode.FULLY_CONNECTED)

    dataset = WorldCup2022Dataset(
        root="soccerai/data/resources",
        converter=converter,
        force_reload=args.reload,
    )
    logger.success(f"Dataset loaded successfully. Number of graphs: {len(dataset)}")

    cfg = build_cfg("configs/example.yaml")

    loader = DataLoader(
        dataset,
        cfg.bs,
        num_workers=os.cpu_count() - 1,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )
    model = GCN(dataset.num_node_features, 256, 1)

    trainer = Trainer(cfg, model, loader, "cuda")
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
