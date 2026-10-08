"""
Integration tests on the real (committed) parquet: they build the processed
dataset with the repository configuration, so the first run takes a while.
"""

from pathlib import Path

import torch
from torch_geometric.data import Data

from soccerai.data.converters import BipartiteGraphConverter, create_graph_converter
from soccerai.data.dataset import WorldCup2022Dataset
from soccerai.training.trainer_config import build_config
from soccerai.training.transforms import RandomVerticalFlip

CONFIG_DIR = Path("configs")


def _dataset(split: str = "train") -> WorldCup2022Dataset:
    # same arguments as scripts/train.py, so that the training cache is reused
    full_cfg = build_config(CONFIG_DIR)
    cfg = full_cfg.data
    converter = create_graph_converter(cfg.connection_mode, cfg.edge_length_scale)
    return WorldCup2022Dataset(
        root="soccerai/data/resources",
        converter=converter,
        force_reload=False,
        split=split,
        cfg=cfg,
        random_state=full_cfg.seed,
    )


def test_bipartite_graph_creation():
    dataset = _dataset()
    assert isinstance(dataset.converter, BipartiteGraphConverter)
    data = dataset[0]
    assert isinstance(data, Data)
    assert data.x.shape == (22, len(dataset.feature_names))
    assert data.edge_index.shape == (2, 2 * 11 * 11)
    assert int(data.edge_index.max()) == 21


def test_augmentations_flip_the_pitch_width():
    dataset = _dataset()
    dataset.transform = None  # sample without the dataset's own random flips
    y_idx = dataset.feature_names.index("y")
    original = dataset[0].x.clone()
    flipped = RandomVerticalFlip(dataset.feature_names, p=1.0)(dataset[0])
    assert torch.allclose(flipped.x[:, y_idx], 1.0 - original[:, y_idx])
    assert torch.allclose(dataset[0].x, original)  # the stored data is untouched


def test_splits_are_balanced():
    # disjointness of the games is tested on `_split_games` in the unit tests
    train, val = _dataset("train"), _dataset("val")
    train_rate = train._data.y.mean().item()
    val_rate = val._data.y.mean().item()
    assert len(train) > len(val) > 0
    assert abs(train_rate - val_rate) < 0.1
