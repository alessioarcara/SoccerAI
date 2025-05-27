import argparse
import os

import torch
from loguru import logger
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.loader import DataLoader, PrefetchLoader
from torch_geometric.transforms import Compose

from soccerai.data.converters import ConnectionMode, ShotPredictionGraphConverter
from soccerai.data.dataset import WorldCup2022Dataset
from soccerai.models import GIN
from soccerai.training.metrics import (
    BinaryConfusionMatrix,
    BinaryPrecisionRecallCurve,
    PositiveFrameCollector,
)
from soccerai.training.trainer import Trainer
from soccerai.training.trainer_config import build_cfg
from soccerai.training.transforms import RandomHorizontalFlip, RandomVerticalFlip
from soccerai.training.utils import fix_random

NUM_WORKERS = (os.cpu_count() or 1) - 1
CONFIG_PATH = "configs/example.yaml"
torch.set_float32_matmul_precision("high")


def main(args):
    cfg = build_cfg(CONFIG_PATH)
    fix_random(cfg.seed)
    converter = ShotPredictionGraphConverter(ConnectionMode.FULLY_CONNECTED)

    train_dataset = WorldCup2022Dataset(
        root="soccerai/data/resources",
        converter=converter,
        force_reload=args.reload,
        split="train",
        val_ratio=cfg.val_ratio,
        transform=Compose([RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.5)]),
    )
    val_dataset = WorldCup2022Dataset(
        root="soccerai/data/resources",
        converter=converter,
        force_reload=args.reload,
        split="val",
        val_ratio=cfg.val_ratio,
    )

    logger.success(
        "Datasets loaded successfully → train graphs: {}, val graphs: {}",
        len(train_dataset),
        len(val_dataset),
    )

    common_loader_kwargs = dict(
        batch_size=cfg.bs,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

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
    model = GIN(train_dataset.num_node_features, cfg.dim, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        metrics=[
            BinaryConfusionMatrix(),
            BinaryPrecisionRecallCurve(),
            PositiveFrameCollector(),
        ],
    )
    trainer.train(args.name)

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="binary_classification",
            task_level="graph",
            return_type="raw",
        ),
    )

    sample = val_dataset[0].to(device)

    explanation = explainer(x=sample.x, edge_index=sample.edge_index)

    torch.set_printoptions(profile="full", linewidth=200)

    # Supponendo che edge_mask e node_mask siano i tuoi tensori:
    print("Edge mask:\n", explanation.edge_mask)
    print("\nNode mask:\n", explanation.node_mask)

    # (facoltativo) Torna alle impostazioni di default dopo la stampa
    torch.set_printoptions(profile="default")


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
