import argparse
from typing import Any

import torch
from loguru import logger
from torch_geometric.nn import summary

from soccerai.config import DEFAULT_CONFIGS, SCHEMA_PATH, build_config
from soccerai.training.utils import build_dummy_inputs

torch.set_float32_matmul_precision("high")


def main(args: argparse.Namespace) -> None:
    overrides: dict[str, Any] = {}
    if args.reload:
        overrides["train_ds"] = {"_init_args_": {"force_reload": True}}
    if args.resume_from:
        overrides["resume_from"] = args.resume_from
    cfg, raw = build_config(args.configs, overrides or None, args.schema)

    logger.success(
        "Datasets loaded → train chains: {}, val chains: {}",
        len(cfg.train_chains),
        len(cfg.val_chains),
    )
    if cfg.pos_weight is not None:
        logger.info("Positive class weight: {:.3f}", cfg.pos_weight)

    trainer = cfg.trainer
    print(
        summary(
            cfg.model,
            **build_dummy_inputs(
                cfg.batch_size,
                cfg.train_ds.num_features,
                cfg.train_ds.num_global_features,
                trainer.device,
            ),
        )
    )
    # the merged YAML is logged to the tracker and stored in the checkpoints
    trainer.config = raw
    trainer.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[str(p) for p in DEFAULT_CONFIGS],
        help="YAML files merged in order (later files win)",
    )
    parser.add_argument("--schema", default=str(SCHEMA_PATH))
    parser.add_argument(
        "--resume-from",
        help="Run id (<run_name>_<timestamp>) to continue, or to fork when "
        "run_name differs",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="If set, forces the dataset to be re-created",
    )
    main(parser.parse_args())
