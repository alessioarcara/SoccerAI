from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from soccerai.training.trainer_config import build_config

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _write_configs(tmp_path: Path, run_name: str, model_cfg: dict) -> Path:
    base = yaml.safe_load((REPO_CONFIGS / "base.yaml").read_text())
    base["run_name"] = run_name
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    (tmp_path / f"{run_name}.yaml").write_text(yaml.safe_dump(model_cfg))
    return tmp_path


@pytest.mark.parametrize(
    "model_yaml",
    sorted(p.name for p in REPO_CONFIGS.glob("*.yaml") if p.name != "base.yaml"),
)
def test_repository_configs_are_valid(tmp_path, model_yaml):
    run_name = model_yaml[: -len(".yaml")]
    model_cfg = yaml.safe_load((REPO_CONFIGS / model_yaml).read_text())
    cfg = build_config(_write_configs(tmp_path, run_name, model_cfg))
    assert cfg.run_name == run_name
    assert cfg.model.backbone.type == run_name


def test_unknown_section_is_rejected(tmp_path):
    model_cfg = yaml.safe_load((REPO_CONFIGS / "gcn.yaml").read_text())
    model_cfg["training"] = {"lr": 1e-5}  # typo for `trainer`
    with pytest.raises(ValidationError, match="training"):
        build_config(_write_configs(tmp_path, "gcn", model_cfg))


def test_unknown_nested_key_is_rejected(tmp_path):
    model_cfg = yaml.safe_load((REPO_CONFIGS / "gcn.yaml").read_text())
    model_cfg["model"]["use_cell_state"] = True
    with pytest.raises(ValidationError, match="use_cell_state"):
        build_config(_write_configs(tmp_path, "gcn", model_cfg))


def test_model_yaml_overrides_base(tmp_path):
    model_cfg = yaml.safe_load((REPO_CONFIGS / "gcn.yaml").read_text())
    model_cfg["trainer"] = {"bs": 7}
    cfg = build_config(_write_configs(tmp_path, "gcn", model_cfg))
    assert cfg.trainer.bs == 7
    assert (
        cfg.trainer.n_epochs
        == yaml.safe_load((REPO_CONFIGS / "base.yaml").read_text())["trainer"][
            "n_epochs"
        ]
    )
