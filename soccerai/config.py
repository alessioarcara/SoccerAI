"""
Configuration loading: YAML files validated by `configs/schema.yaml` and
instantiated by EzConfy (`_target_type_` / `_init_args_`).
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml
from ezconfy import ConfigBuilder

from soccerai.generated import ConfigModel
from soccerai.training.utils import fix_random

PathLike = str | Path

CONFIG_DIR = Path("configs")
SCHEMA_PATH = CONFIG_DIR / "schema.yaml"
DEFAULT_CONFIGS: list[PathLike] = [
    CONFIG_DIR / "base.yaml",
    CONFIG_DIR / "models" / "gcn.yaml",
]


def _peek_seed(
    config_paths: Sequence[PathLike], overrides: Mapping[str, Any] | None
) -> int | None:
    """The `seed` the merged configuration will have (later files win)."""
    seed = None
    for path in config_paths:
        seed = (yaml.safe_load(Path(path).read_text()) or {}).get("seed", seed)
    if overrides and "seed" in overrides:
        seed = overrides["seed"]
    return seed


def build_config(
    config_paths: Sequence[PathLike] = DEFAULT_CONFIGS,
    overrides: dict[str, Any] | None = None,
    schema_path: PathLike = SCHEMA_PATH,
) -> tuple[ConfigModel, dict[str, Any]]:
    """
    Merge `config_paths` (later files win) and `overrides`, instantiate every
    object and validate the result against the schema.

    The random generators are seeded before anything is built, so model
    initialisation is reproducible. Returns the built configuration and the
    merged YAML (stored in checkpoints, logged to W&B).
    """
    seed = _peek_seed(config_paths, overrides)
    if seed is not None:
        fix_random(seed)
    cfg, raw = ConfigBuilder.from_files(
        config_paths=list(config_paths),
        overrides=overrides,
        schema_path=schema_path,
        return_raw_config=True,
    )  # type: ignore[misc]
    return cast(ConfigModel, cfg), cast(dict[str, Any], raw)


def build_config_from_dict(
    raw: dict[str, Any], tmp_dir: PathLike, schema_path: PathLike = SCHEMA_PATH
) -> tuple[ConfigModel, dict[str, Any]]:
    """Rebuild a configuration from a merged YAML dict (e.g. a checkpoint's)."""
    path = Path(tmp_dir) / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return build_config([path], schema_path=schema_path)
