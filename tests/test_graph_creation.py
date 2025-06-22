import torch
from torch_geometric.data import Data

from soccerai.data.converters import BipartiteGraphConverter, create_graph_converter
from soccerai.data.dataset import WorldCup2022Dataset
from soccerai.training.trainer_config import build_cfg

CONFIG_PATH = "configs/base.yaml"


def test_augmentations():
    cfg = build_cfg(CONFIG_PATH).data
    converter = create_graph_converter("fully_connected")
    dataset = WorldCup2022Dataset(
        root="soccerai/data/resources",
        converter=converter,
        force_reload=False,
        split="train",
        cfg=cfg,
    )

    cfg.use_augmentations = True
    aug_dataset = WorldCup2022Dataset(
        root="soccerai/data/resources",
        converter=converter,
        force_reload=False,
        split="train",
        cfg=cfg,
    )

    x_col_idx = dataset.feature_names.index("x")
    assert not torch.allclose(
        dataset[0].x[:, x_col_idx], aug_dataset[0].x[:, x_col_idx]
    )


def test_bipartite_graph_creation():
    cfg = build_cfg(CONFIG_PATH).data
    converter = create_graph_converter("bipartite")
    assert isinstance(converter, BipartiteGraphConverter)
    ds = WorldCup2022Dataset(
        root="soccerai/data/resources",
        converter=converter,
        force_reload=False,
        split="train",
        cfg=cfg,
    )
    assert isinstance(ds[0], Data)
