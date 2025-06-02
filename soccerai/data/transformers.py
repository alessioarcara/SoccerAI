from typing import Optional

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin


class BaseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, output: str = "default"):
        self.output = output

    def set_output(self, transform: Optional[str] = None) -> "BaseTransformer":
        valid_outputs = {"default", "polars"}
        if transform is None:
            self.output = "default"
        elif transform in valid_outputs:
            self.output = transform
        else:
            raise ValueError(f"Unsupported output type: {transform}")
        return self


class PlayerPositionTransformer(BaseTransformer):
    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        output: str = "default",
    ):
        super().__init__(output)
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        coords = np.asarray(X, dtype=float)
        x_normed = np.clip(coords[:, 0] / self.pitch_length, 0.0, 1.0)
        y_normed = np.clip(coords[:, 1] / self.pitch_width, 0.0, 1.0)
        result = np.column_stack((x_normed, y_normed))

        if self.output == "polars":
            return pl.DataFrame({"x": result[:, 0], "y": result[:, 1]})
        return result


class GoalPositionTransformer(BaseTransformer):
    def __init__(self, output="default"):
        super().__init__(output)

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        pass
