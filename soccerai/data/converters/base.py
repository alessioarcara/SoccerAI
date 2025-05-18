from abc import ABC, abstractmethod
from enum import Enum
from typing import List

import polars as pl
import torch
from torch_geometric.data import Data


class ConnectionMode(Enum):
    FULLY_CONNECTED = "fully_connected"


class GraphConverter(ABC):
    def __init__(self, mode: ConnectionMode):
        self.mode = mode

    @abstractmethod
    def _create_edge_index(self) -> torch.Tensor:
        pass

    @abstractmethod
    def _preprocess_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        pass

    @abstractmethod
    def convert_dataframe_to_data_list(self, df: pl.DataFrame) -> List[Data]:
        pass
