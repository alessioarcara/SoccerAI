from pathlib import Path

import yaml
from pydantic import BaseModel


class ModelConfig(BaseModel):
    model_name: str
    dmid: int
    dhid: int


class TrainerConfig(BaseModel):
    bs: int
    lr: float
    wd: float
    n_epochs: int
    eval_rate: int
    gamma: float


class DataConfig(BaseModel):
    val_ratio: float
    include_goal_features: bool
    use_macro_roles: bool
    connection_mode: str


class ExplainConfig(BaseModel):
    n_frames: int = 12
    thr: float = 0.5
    grid_nrows: int = 3
    grid_ncols: int = 4
    grid_figheight: float = 12.0
    grid_pitch_type: str = "metricasports"
    log_best_single: bool = True


class Config(BaseModel):
    project_name: str
    seed: int
    use_temporal: bool
    model: ModelConfig
    trainer: TrainerConfig
    data: DataConfig
    explain: ExplainConfig


def _load_yaml(path: str | Path):
    return yaml.safe_load(Path(path).expanduser().read_text()) or {}


def build_cfg(*yaml_paths: str | Path) -> Config:
    merged = {}
    for p in yaml_paths:
        merged.update(_load_yaml(p))
    return Config(**merged)
