"""
Synthetic stand-ins for the processed datasets, so that the repository
configurations can be built (and trained) end to end without the real data.

`build_stub_config` builds `configs/base.yaml` + `configs/models/<model>.yaml`
with the datasets swapped for the stubs below, CPU loaders without workers and
checkpoints written under `tmp_path`.
"""

from pathlib import Path
from typing import Any

from ezconfy import ConfigBuilder
from test_temporal_collate import N_FEAT, make_chain

from soccerai.config import CONFIG_DIR, SCHEMA_PATH, build_config
from soccerai.data.dataset import WorldCup2022Dataset
from soccerai.data.temporal_dataset import TemporalChainsDataset

N_GLOBAL = 3
# the carrier readout looks the flag up by name: use the first column
FEATURE_NAMES = ["is_ball_carrier_1"] + [f"f{i}" for i in range(1, N_FEAT)]
MODELS = sorted(p.stem for p in (CONFIG_DIR / "models").glob("*.yaml"))


class StubDataset(WorldCup2022Dataset):
    """A `WorldCup2022Dataset` that accepts its arguments and loads nothing."""

    num_node_features = N_FEAT
    num_features = N_FEAT
    num_global_features = N_GLOBAL
    feature_names = FEATURE_NAMES
    carrier_feature_idx = 0

    def __init__(self, split: str = "train", **_: Any) -> None:  # noqa: D107
        # no super().__init__(): nothing is processed or loaded
        self.split = split

    def positive_weight(self) -> float:
        return 1.0


class StubChains(TemporalChainsDataset):
    """Four short synthetic chains (two per class) per split."""

    @staticmethod
    def from_worldcup_dataset(dataset, max_chain_len=None) -> TemporalChainsDataset:
        offset = 0 if dataset.split == "train" else 10
        chains = [
            make_chain(3, 1.0, offset),
            make_chain(2, 0.0, offset + 1),
            make_chain(1, 1.0, offset + 2),
            make_chain(4, 0.0, offset + 3),
        ]
        return TemporalChainsDataset(chains, N_FEAT, N_GLOBAL, FEATURE_NAMES)


def stub_overrides(tmp_path: Path) -> dict[str, Any]:
    loader = {
        "_init_args_": {
            "batch_size": 2,
            "num_workers": 0,
            "persistent_workers": False,
            "prefetch_factor": None,
            "pin_memory": False,
        }
    }
    return {
        "device": "cpu",
        "n_epochs": 1,
        "collector_frames": 0,  # pitch plots need the real feature names
        "train_ds": {"_target_type_": "stubs:StubDataset"},
        "val_ds": {"_target_type_": "stubs:StubDataset"},
        "train_chains": {"_target_type_": "stubs:StubChains"},
        "val_chains": {"_target_type_": "stubs:StubChains"},
        "train_loader": loader,
        "val_loader": loader,
        "callbacks": [
            "...",
            {"_id_": "checkpoint", "_init_args_": {"out_dir": str(tmp_path)}},
        ],
    }


def build_stub_config(
    model: str, tmp_path: Path, overrides: dict[str, Any] | None = None
):
    merged = ConfigBuilder._deep_merge(stub_overrides(tmp_path), overrides or {})
    return build_config(
        [CONFIG_DIR / "base.yaml", CONFIG_DIR / "models" / f"{model}.yaml"],
        merged,
        SCHEMA_PATH,
    )
