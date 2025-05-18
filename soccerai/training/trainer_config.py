from pathlib import Path

import yaml
from pydantic import BaseModel


class TrainerConfig(BaseModel):
    project_name: str
    bs: int
    lr: float
    wd: float
    n_epochs: int
    eval_rate: int


def _load_yaml(path: str | Path):
    return yaml.safe_load(Path(path).expanduser().read_text()) or {}


def build_cfg(*yaml_paths: str | Path) -> TrainerConfig:
    merged = {}
    for p in yaml_paths:
        merged.update(_load_yaml(p))
    return TrainerConfig(**merged)
