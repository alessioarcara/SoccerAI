from typing import Callable, Optional

import polars as pl
from loguru import logger
from torch_geometric.data import InMemoryDataset

from soccerai.data.converters import GraphConverter


class WorldCup2022Dataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        converter: GraphConverter,
        split: str,
        val_ratio: float = 0.2,
        transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        self.converter = converter
        self.split = split
        self.val_ratio = val_ratio
        super().__init__(root=root, transform=transform, force_reload=force_reload)
        data_path_idx = 0 if self.split == "train" else 1
        self.load(self.processed_paths[data_path_idx])

    @property
    def raw_file_names(self):
        return ["dataset.parquet"]

    @property
    def processed_file_names(self):
        return ["train_data.pt", "val_data.pt"]

    def _split_dataframe_by_event_keys(
        self,
        df: pl.DataFrame,
        key_cols: list[str] = ["gameEventId", "possessionEventId"],
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        event_keys_df = df.select(key_cols).unique()
        num_events = event_keys_df.height

        train_end_idx = int((1.0 - self.val_ratio) * num_events)

        train_event_keys = event_keys_df.slice(0, train_end_idx)
        val_event_keys = event_keys_df.slice(train_end_idx, num_events)

        train_df = df.join(train_event_keys, on=key_cols, how="semi")
        val_df = df.join(val_event_keys, on=key_cols, how="semi")
        return train_df, val_df

    def process(self):
        df = pl.read_parquet(self.raw_paths[0])

        train_df, val_df = self._split_dataframe_by_event_keys(df)

        logger.info(
            "DataFrame split → train: {} rows, val: {} rows",
            train_df.height,
            val_df.height,
        )

        train_data_list = self.converter.convert_dataframe_to_data_list(train_df)
        val_data_list = self.converter.convert_dataframe_to_data_list(val_df)

        self.save(train_data_list, self.processed_paths[0])
        self.save(val_data_list, self.processed_paths[1])
