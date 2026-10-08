from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

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
    Return `(wandb_run_id, path)` of the checkpoint with the lowest monitored
    value among `<run_id>_<key>_<value>.pth` files under `model_dir`
    (searched recursively).

    Checkpoints of older formats (bare state dicts, pydantic configs) are
    skipped unless `include_legacy` is set: their configuration cannot be
    rebuilt by the current code.
    """
    candidates: list[tuple[float, str, Path]] = []
    for path in model_dir.rglob("*.pth"):
        run_id, _, rest = path.stem.partition("_")
        try:
            value = float(rest.rsplit("_", 1)[-1])
        except ValueError:
            continue
        candidates.append((value, run_id, path))

    # best value first, so that usually a single file has to be loaded
    for _, run_id, path in sorted(candidates, key=lambda c: c[0]):
        if include_legacy or load_checkpoint(path).get("format") == CHECKPOINT_FORMAT:
            return run_id, path
    return None
