import torch

from soccerai.data.converters import ConnectionMode, ShotPredictionGraphConverter
from soccerai.data.dataset import WorldCup2022Dataset
from soccerai.training.transforms import RandomHorizontalFlip


def test_transforms():
    converter = ShotPredictionGraphConverter(ConnectionMode.FULLY_CONNECTED)
    dataset = WorldCup2022Dataset(
        root="soccerai/data/resources",
        converter=converter,
        force_reload=False,
        split="train",
    )

    aug_dataset = WorldCup2022Dataset(
        root="soccerai/data/resources",
        converter=converter,
        force_reload=False,
        split="train",
        transform=RandomHorizontalFlip(p=1.0),
    )

    assert not torch.allclose(dataset[0].x[:, 0], aug_dataset[0].x[:, 0])
