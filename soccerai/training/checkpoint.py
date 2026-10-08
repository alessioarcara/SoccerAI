from __future__ import annotations

import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from eztrain import RunType, run_id_base
from loguru import logger

if TYPE_CHECKING:
    from soccerai.training.trainer import BaseTrainer

# 2: `config` is the merged EzConfy YAML of the run (1: pydantic Config dump)
CHECKPOINT_FORMAT = 2


def save_checkpoint(
    path: Path,
    state_dict: Mapping[str, torch.Tensor],
    run_config: Mapping[str, Any],
    feature_names: Sequence[str],
    history_key: str,
    best_value: float,
    metrics: Mapping[str, float] | None = None,
) -> None:
    """
    Save a self-contained checkpoint: weights plus everything needed to
    rebuild the model and its dataset (the merged YAML configuration, feature
    names) without querying the experiment tracker.
    """
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
            "config": dict(run_config),
            "feature_names": list(feature_names),
            "history_key": history_key,
            "best_value": float(best_value),
            # every validation metric at the epoch the checkpoint comes from
            "metrics": {k: float(v) for k, v in (metrics or {}).items()},
        },
        path,
    )


def load_checkpoint(path: Path) -> dict[str, Any]:
    """
    Load a checkpoint saved by `save_checkpoint`; bare state dicts written
    by older versions are wrapped into the same structure (without config).
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload
    return {"state_dict": payload, "config": None, "feature_names": None}


def find_best_checkpoint(
    model_dir: Path, include_legacy: bool = False
) -> tuple[str, Path] | None:
    """
    Return `(run_id, path)` of the checkpoint with the lowest monitored value
    among the `best_<key>_<value>.pth` files under `model_dir` (searched
    recursively; the run id is the name of the folder holding the file).

    Checkpoints of older formats (bare state dicts, pydantic configs) are
    skipped unless `include_legacy` is set: their configuration cannot be
    rebuilt by the current code.
    """
    candidates: list[tuple[float, str, Path]] = []
    for path in model_dir.rglob("*.pth"):
        try:
            value = float(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        candidates.append((value, path.parent.name, path))

    # best value first, so that usually a single file has to be loaded
    for _, run_id, path in sorted(candidates, key=lambda c: c[0]):
        if include_legacy or load_checkpoint(path).get("format") == CHECKPOINT_FORMAT:
            return run_id, path
    return None


class TorchCheckpointer:
    """
    EzTrain `Checkpointer` (scheduled by `eztrain.CheckpointCallback`) for
    `<root>/<run name>/<run id>/`:

    - `last.pth`: model, optimizer, scheduler, iteration and history after
      every save, to CONTINUE a run (`resume_from` with the same run name) or
      FORK it (weights only, under a new name);
    - `best_<monitor>_<value>.pth`: a self-contained checkpoint (see
      `save_checkpoint`) of the best `monitor` value so far, the one
      `scripts/eval.py` picks. The previous best file is replaced.
    """

    def __init__(
        self,
        *,
        monitor: str = "val/loss",
        mode: Literal["min", "max"] = "min",
        root: str | Path = "checkpoints",
        log_artifact: bool = True,
    ) -> None:
        self.monitor = monitor
        self.mode = mode
        self.root = Path(root)
        self.log_artifact = log_artifact
        self.run_dir: Path | None = None
        self.best: float | None = None
        self.best_path: Path | None = None

    def setup(self, trainer: BaseTrainer) -> None:
        run = trainer.run
        self.run_dir = self.root / run.name / run.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if run.restore_dir is None:
            return

        last = self.root / run_id_base(run.restore_dir) / run.restore_dir / "last.pth"
        state = torch.load(last, map_location="cpu", weights_only=False)
        trainer.model.load_state_dict(state["model"])
        if run.run_type is RunType.CONTINUE:
            trainer.optim.load_state_dict(state["optimizer"])
            trainer.scheduler.load_state_dict(state["scheduler"])
            trainer.start_iteration = int(state["iteration"])
            trainer.history.update(state["history"])
        logger.info("Restored {} ({})", last, run.run_type.name)

    def save(
        self, trainer: BaseTrainer, iteration: int, metrics: Mapping[str, Any]
    ) -> None:
        assert self.run_dir is not None, "setup() was not called"
        scalars = {k: float(v) for k, v in metrics.items() if _is_number(v)}
        torch.save(
            {
                "model": trainer.model.state_dict(),
                "optimizer": trainer.optim.state_dict(),
                "scheduler": trainer.scheduler.state_dict(),
                "iteration": iteration,
                "history": scalars,
            },
            self.run_dir / "last.pth",
        )

        value = scalars.get(self.monitor)
        if value is None or not self._improves(value):
            return
        self.best = value
        if self.best_path is not None:
            self.best_path.unlink(missing_ok=True)
        key = re.sub(r"[^\w.-]", "-", self.monitor)
        self.best_path = self.run_dir / f"best_{key}_{value:.4f}.pth"
        save_checkpoint(
            self.best_path,
            trainer.model.state_dict(),
            trainer.config or {},
            trainer.feature_names,
            self.monitor,
            value,
            metrics=scalars,
        )
        logger.info("Saved best checkpoint {}", self.best_path)

    def close(self) -> None:
        wandb = sys.modules.get("wandb")
        if (
            not self.log_artifact
            or self.best_path is None
            or wandb is None
            or wandb.run is None
        ):
            return
        artifact = wandb.Artifact(name=self.best_path.parent.parent.name, type="model")
        artifact.add_file(str(self.best_path), name=self.best_path.name)
        wandb.run.log_artifact(artifact)

    def _improves(self, value: float) -> bool:
        if self.best is None:
            return True
        return value < self.best if self.mode == "min" else value > self.best


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
