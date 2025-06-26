from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class BackboneConfig(BaseModel):
    name: Literal["gcn", "gcn2"]
    drop: float
    dhid: int
    dout: int
    n_layers: int
    skip_stride: int
    norm: Literal["none", "layer", "graph", "instance"]


class NeckConfig(BaseModel):
    readout: Literal["sum", "mean"]
    dhid: int
    k: int


class HeadConfig(BaseModel):
    n_layers: int
    drop: float
    dout: int


class ModelConfig(BaseModel):
    name: Literal["gnn", "tgnn"]
    backbone: BackboneConfig
    neck: NeckConfig
    head: HeadConfig


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
    include_ball_features: bool
    use_macro_roles: bool
    use_augmentations: bool
    use_regression_imputing: bool
    use_pca_on_roster_cols: bool
    mask_non_possession_shooting_stats: bool
    connection_mode: str


class CollectorConfig(BaseModel):
    n_frames: int


class PitchGridConfig(BaseModel):
    nrows: int
    ncols: int
    figheight: int


class MetricsConfig(BaseModel):
    thr: float
    fbeta: float


class Config(BaseModel):
    project_name: str
    seed: int
    model: ModelConfig
    trainer: TrainerConfig
    data: DataConfig
    collector: CollectorConfig
    metrics: MetricsConfig
    pitch_grid: PitchGridConfig


def _load_yaml(path: str | Path):
    return yaml.safe_load(Path(path).expanduser().read_text()) or {}


def build_cfg(*yaml_paths: str | Path) -> Config:
    merged = {}
    for p in yaml_paths:
        merged.update(_load_yaml(p))
    return Config(**merged)
