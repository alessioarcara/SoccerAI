from typing import Callable, Optional

import polars as pl
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
        return "dataset.parquet"

    @property
    def processed_file_names(self):
        return "data.pt"

    def process(self):
        df = pl.read_parquet(self.raw_paths[0])
        data_list = self.converter.convert_dataframe_to_data_list(df)
        self.save(data_list, self.processed_paths[0])
