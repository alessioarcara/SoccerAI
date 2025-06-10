import torch
from torch_geometric.data import Data

from soccerai.data.converters import BipartiteGraphConverter, create_graph_converter
from soccerai.data.dataset import WorldCup2022Dataset
from soccerai.training.trainer_config import build_cfg
from soccerai.training.transforms import RandomHorizontalFlip

CONFIG_PATH = "configs/base.yaml"


def test_augmentations():
    cfg = build_cfg(CONFIG_PATH)
    converter = create_graph_converter("fully_connected")
    dataset = WorldCup2022Dataset(
        root="soccerai/data/resources",
        converter=converter,
        force_reload=False,
        split="train",
        cfg=cfg,
    )

    aug_dataset = WorldCup2022Dataset(
        root="soccerai/data/resources",
        converter=converter,
        force_reload=False,
        split="train",
        cfg=cfg,
        transform=RandomHorizontalFlip(p=1.0),
    )

    assert not torch.allclose(dataset[0].x[:, 0], aug_dataset[0].x[:, 0])


def test_bipartite_graph_creation():
    cfg = build_cfg(CONFIG_PATH)
    converter = create_graph_converter("bipartite")
    assert isinstance(converter, BipartiteGraphConverter)
    ds = WorldCup2022Dataset(
        root="soccerai/data/resources",
        converter=converter,
        force_reload=False,
        split="train",
        cfg=cfg,
    )
    print(ds[0])
    assert isinstance(ds[0], Data)
