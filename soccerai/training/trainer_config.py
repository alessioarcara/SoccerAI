from pathlib import Path
from typing import Annotated, Any, Dict, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field

from soccerai.models.typings import (
    AggregationType,
    NormalizationType,
    ReadoutType,
    RNNType,
    TemporalMode,
)

PathLike = str | Path


class StrictModel(BaseModel):
    """
    Base for every config section: unknown keys are an error, so that a typo
    (e.g. `training:` instead of `trainer:`) cannot be silently ignored.
    """

    model_config = ConfigDict(extra="forbid")


class BackboneCommon(StrictModel):
    n_layers: int
    dout: int
    drop: float
    norm: NormalizationType


class GCNConfig(BackboneCommon):
    type: Literal["gcn"]
    plus: bool


class GCN2Config(BackboneCommon):
    type: Literal["gcn2"]


class GraphSAGEConfig(BackboneCommon):
    type: Literal["graphsage"]
    aggr_type: AggregationType
    l2_norm: bool


class GATv2Config(BackboneCommon):
    type: Literal["gatv2"]
    use_edge_attr: bool
    num_heads: int
    edge_dropout: float


class GINEConfig(BackboneCommon):
    type: Literal["gine"]
    train_eps: bool
    plus: bool


class GraphGPSConfig(BackboneCommon):
    type: Literal["graphgps"]
    heads: int
    attn_drop: float


class DiffPoolConfig(BackboneCommon):
    type: Literal["diffpool"]
    dhid: int
    pooling_ratio: float
    dhid_multiplier: int


BackboneConfig = Annotated[
    Union[
        GCNConfig,
        GCN2Config,
        GraphSAGEConfig,
        GATv2Config,
        GINEConfig,
        GraphGPSConfig,
        DiffPoolConfig,
    ],
    Field(discriminator="type"),
]


class NeckConfig(StrictModel):
    rnn_type: RNNType
    readout: ReadoutType
    glob_dout: int
    rnn_din: int
    rnn_dout: int
    mode: TemporalMode
    raw_features_proj: bool
    proj_dout: int


class HeadConfig(StrictModel):
    n_layers: int
    din: int
    drop: float


class ModelConfig(StrictModel):
    use_temporal: bool
    use_hierarchical: bool
    backbone: BackboneConfig
    neck: NeckConfig
    head: HeadConfig


class ModelMonitorCallbackConfig(StrictModel):
    history_key: str
    minimize: bool


class EarlyStoppingCallbackConfig(ModelMonitorCallbackConfig):
    patience: int


class ModelSavingCallbackConfig(ModelMonitorCallbackConfig):
    pass


class TrainerConfig(StrictModel):
    bs: int
    lr: float
    wd: float
    n_epochs: int
    eval_rate: int
    gamma: float
    # weight of the auxiliary loss exposed by some models (DiffPool)
    aux_loss_weight: float = 1.0
    early_stopping_callback: Optional[EarlyStoppingCallbackConfig] = None
    model_saving_callback: Optional[ModelSavingCallbackConfig] = None


class DataConfig(StrictModel):
    val_ratio: float
    # "chronological": last games (knock-out stage) validate; "random": seeded
    split_mode: Literal["chronological", "random"] = "chronological"
    # games whose chains are all positive would only distort the class prior
    drop_games_without_negatives: bool = True
    # keep positive chains only if their last action is within this distance
    # (metres) of the attacked goal line, like the negatives (None = keep all)
    goal_window_for_positives: Optional[float] = 25.0
    include_goal_features: bool
    include_ball_features: bool
    use_macro_roles: bool
    use_augmentations: bool
    use_regression_imputing: bool
    use_pca_on_roster_cols: bool
    mask_non_possession_shooting_stats: bool
    connection_mode: str
    # length scale (metres) of the proximity edge weights exp(-d / scale)
    edge_length_scale: float = 10.0
    # mirror frames so that the possession team always attacks towards x = 105
    normalize_attack_direction: bool = True
    # keep only the last frames of every chain (None = whole chain)
    max_chain_len: Optional[int] = 12
    # per-player scraped statistics (weight, market value, shooting record,
    # age); constant per player, they let the model identify players
    use_roster_features: bool = False
    # match clock (seconds) as a global feature
    use_match_clock: bool = False


class CollectorConfig(StrictModel):
    n_frames: int


class PitchGridConfig(StrictModel):
    nrows: int
    ncols: int
    figheight: int


class MetricsConfig(StrictModel):
    thr: float
    fbeta: float


class Config(StrictModel):
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
