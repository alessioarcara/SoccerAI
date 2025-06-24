from pathlib import Path

import yaml
from pydantic import BaseModel


class ModelConfig(BaseModel):
    model_name: str
    dmid: int
    dhid: int
    backbone: str
    num_heads: int
    num_layers: int
    dropout_layer: float
    dropout_head: float


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
    use_temporal: bool
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
