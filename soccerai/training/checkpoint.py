from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

from soccerai.training.trainer_config import Config

CHECKPOINT_FORMAT = 1


def save_checkpoint(
    path: Path,
    state_dict: Mapping[str, torch.Tensor],
    cfg: Config,
    feature_names: Sequence[str],
    history_key: str,
    best_value: float,
) -> None:
    """
    Save a self-contained checkpoint: weights plus everything needed to
    rebuild the model and its dataset (config, feature names) without
    querying the experiment tracker.
    """
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
            "config": cfg.model_dump(),
            "feature_names": list(feature_names),
            "history_key": history_key,
            "best_value": float(best_value),
        },
        path,
    )


def load_checkpoint(path: Path) -> Dict[str, Any]:
    """
    Load a checkpoint saved by `save_checkpoint`; bare state dicts written
    by older versions are wrapped into the same structure (without config).
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload
    return {"state_dict": payload, "config": None, "feature_names": None}


def checkpoint_config(payload: Mapping[str, Any]) -> Optional[Config]:
    cfg = payload.get("config")
    return None if cfg is None else Config(**cfg)


def find_best_checkpoint(model_dir: Path) -> Optional[Tuple[str, Path]]:
    """
    Return `(wandb_run_id, path)` of the checkpoint with the lowest monitored
    value among `<run_id>_<key>_<value>.pth` files under `model_dir`
    (searched recursively).
    """
    best: Optional[Tuple[str, Path, float]] = None
    for path in model_dir.rglob("*.pth"):
        run_id, _, rest = path.stem.partition("_")
        try:
            value = float(rest.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if best is None or value < best[2]:
            best = (run_id, path, value)

    return None if best is None else (best[0], best[1])
