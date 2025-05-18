from typing import Callable, Optional, Tuple

import polars as pl
from loguru import logger
from torch_geometric.data import InMemoryDataset

from soccerai.data.converters import GraphConverter


class WorldCup2022Dataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        converter: GraphConverter,
        transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        self.converter = converter
        super().__init__(root=root, transform=transform, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["dataset.parquet"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        df = pl.read_parquet(self.raw_paths[0])
        data_list = self.converter.convert_dataframe_to_data_list(df)
        self.save(data_list, self.processed_paths[0])


def split_dataset(
    dataset: InMemoryDataset, val_ratio: float
) -> Tuple[InMemoryDataset, InMemoryDataset]:
    num_samples = len(dataset)

    train_end_idx = int((1.0 - val_ratio) * num_samples)

    train_indices_list = list(range(train_end_idx))
    val_indices_list = list(range(train_end_idx, num_samples))

    train_dataset = dataset[train_indices_list]
    val_dataset = dataset[val_indices_list]

    logger.info(
        f"Dataset split into: Train: {len(train_dataset)}, Val: {len(val_dataset)}"
    )

    return train_dataset, val_dataset
