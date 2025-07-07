import argparse
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data.dataloader import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.loader import PrefetchLoader
from torch_geometric.nn import summary

from soccerai.data.converters import create_graph_converter
from soccerai.data.dataset import WorldCup2022Dataset
from soccerai.data.temporal_dataset import TemporalChainsDataset
from soccerai.models.models import build_model
from soccerai.training.callbacks import build_callbacks
from soccerai.training.metrics import (
    BinaryConfusionMatrix,
    BinaryPrecisionRecallCurve,
    ChainCollector,
    FrameCollector,
)
from soccerai.training.trainer import TemporalTrainer, Trainer
from soccerai.training.trainer_config import build_config
from soccerai.training.utils import build_dummy_inputs, fix_random

CONFIG_DIR = Path("configs")

torch.set_float32_matmul_precision("high")

NUM_WORKERS = 4


def main(args):
    cfg = build_config(CONFIG_DIR)
    fix_random(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    converter = create_graph_converter(cfg.data.connection_mode)
    ds_kwargs = dict(
        root="soccerai/data/resources",
        converter=converter,
        cfg=cfg.data,
        random_state=cfg.seed,
    )

    train_ds = WorldCup2022Dataset(split="train", force_reload=args.reload, **ds_kwargs)
    val_ds = WorldCup2022Dataset(split="val", **ds_kwargs)

    logger.success(
        "Datasets loaded successfully → train graphs: {}, val graphs: {}",
        len(train_ds),
        len(val_ds),
    )

    model = build_model(cfg, train_ds)
    loader_kwargs = dict(
        batch_size=cfg.trainer.bs,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    callbacks = build_callbacks(cfg)
    if cfg.model.use_temporal:
        train_ds = TemporalChainsDataset.from_worldcup_dataset(train_ds)
        val_ds = TemporalChainsDataset.from_worldcup_dataset(val_ds)

        train_loader = TorchDataLoader(
            train_ds,
            collate_fn=TemporalChainsDataset.collate,
            shuffle=True,
            **loader_kwargs,
        )
        val_loader = TorchDataLoader(
            val_ds,
            collate_fn=TemporalChainsDataset.collate,
            shuffle=False,
            **loader_kwargs,
        )
        trainer = TemporalTrainer(
            cfg=cfg,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            feature_names=train_ds.feature_names,
            metrics=[
                BinaryConfusionMatrix(cfg.metrics, -1),
                BinaryPrecisionRecallCurve(-1),
                ChainCollector(1, cfg, train_ds.feature_names),
                ChainCollector(0, cfg, train_ds.feature_names),
            ],
            callbacks=callbacks,
        )

    else:
        train_loader = PrefetchLoader(
            PyGDataLoader(
                train_ds,
                shuffle=True,
                **loader_kwargs,
            ),
        )
        val_loader = PrefetchLoader(
            PyGDataLoader(
                val_ds,
                shuffle=False,
                **loader_kwargs,
            ),
        )

        trainer = Trainer(
            cfg=cfg,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            feature_names=train_ds.feature_names,
            metrics=[
                BinaryConfusionMatrix(cfg.metrics),
                BinaryPrecisionRecallCurve(),
                FrameCollector(1, cfg, train_ds.feature_names),
                FrameCollector(0, cfg, train_ds.feature_names),
            ],
            callbacks=callbacks,
        )

    print(
        summary(
            model,
            **build_dummy_inputs(
                cfg.trainer.bs,
                train_ds.num_features,
                train_ds.num_global_features,
                device,
            ),
        )
    )

    trainer.train(cfg.run_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reload",
        action="store_true",
        help="If set, forces the dataset to be re-created",
    )
    args = parser.parse_args()
    main(args)
