from pathlib import Path
from typing import Annotated, Any, Dict, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field

from soccerai.models.typings import (
    AggregationType,
    NormalizationType,
    ReadoutType,
    ResidualSumMode,
)

PathLike = str | Path


class BackboneCommon(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dhid: int
    dout: int
    drop: float
    norm: NormalizationType


class GCNConfig(BackboneCommon):
    type: Literal["gcn"]


class GCN2Config(BackboneCommon):
    type: Literal["gcn2"]
    n_layers: int
    residual_sum_mode: ResidualSumMode


class GraphSAGEConfig(BackboneCommon):
    type: Literal["graphsage"]
    n_layers: int
    residual_sum_mode: ResidualSumMode
    aggr_type: AggregationType
    l2_norm: bool


class GINEConfig(BackboneCommon):
    type: Literal["gine"]
    n_layers: int
    residual_sum_mode: ResidualSumMode


BackboneConfig = Annotated[
    Union[GCNConfig, GCN2Config, GraphSAGEConfig, GINEConfig],
    Field(discriminator="type"),
]


class NeckConfig(BaseModel):
    readout: ReadoutType
    dhid: int


class HeadConfig(BaseModel):
    n_layers: int
    drop: float


class ModelConfig(BaseModel):
    use_temporal: bool
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
    run_name: str
    seed: int
    model: ModelConfig
    trainer: TrainerConfig
    data: DataConfig
    collector: CollectorConfig
    metrics: MetricsConfig
    pitch_grid: PitchGridConfig


def _load_yaml(path: PathLike):
    return yaml.safe_load(Path(path).expanduser().read_text()) or {}


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge dict `b` into dict `a`
    """
    result = a.copy()
    for k, v in b.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def build_config(config_dir: Path) -> Config:
    """
    Load and deeply merge multiple YAML configuration files.

    Steps:
    1. Load the base configuration from 'base.yaml'.
    2. Determine the model name from the base config and load its specific YAML file.
    3. Merge the base and model-specific configs.
    4. Instantiate and return a Config object with the merged settings.
    """
    base_yaml_path = config_dir / "base.yaml"
    base_cfg_dict = _load_yaml(base_yaml_path)

    model_yaml_path = config_dir / f"{base_cfg_dict['run_name']}.yaml"
    model_cfg_dict = _load_yaml(model_yaml_path)

    merged_dict = _deep_merge(base_cfg_dict, model_cfg_dict)
    return Config(**merged_dict)
