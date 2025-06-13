from typing import Optional

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin


class BaseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, output: str = "default"):
        self.output = output

    def fit(self, X, y=None):
        return self

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

    def transform(self, X):
        coords = np.asarray(X, dtype=float)
        x_normed = np.clip(coords[:, 0] / self.pitch_length, 0.0, 1.0)
        y_normed = np.clip(coords[:, 1] / self.pitch_width, 0.0, 1.0)
        result = np.column_stack((x_normed, y_normed))

        if self.output == "polars":
            return pl.DataFrame({"x": result[:, 0], "y": result[:, 1]})
        return result


class GoalLocationTransformer(BaseTransformer):
    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        output: str = "default",
    ):
        super().__init__(output)
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

    def transform(self, X):
        coords = np.asarray(X, dtype=float)
        x = coords[:, 0]
        y = coords[:, 1]
        x_goal = coords[:, 2]
        y_goal = coords[:, 3]

        dx = x_goal - x
        dy = y_goal - y

        goal_dist = np.hypot(dx, dy) + 1e-6

        cos_theta = dx / goal_dist
        sin_theta = dy / goal_dist

        norm = np.hypot(self.pitch_length, self.pitch_width)
        goal_dist_normed = goal_dist / norm

        result = np.column_stack((goal_dist_normed, cos_theta, sin_theta))

        if self.output == "polars":
            return pl.DataFrame(
                {
                    "goal_dist": result[:, 0],
                    "goal_cos": result[:, 1],
                    "goal_sin": result[:, 2],
                }
            )

        return result


class BallLocationTransformer(BaseTransformer):
    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        output: str = "default",
    ):
        super().__init__(output)
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

    def transform(self, X):
        coords = np.asarray(X, dtype=float)
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2] / 100.0  # height_cm -> height_m
        x_ball = coords[:, 3]
        y_ball = coords[:, 4]
        z_ball = coords[:, 5]

        #  planar distance between player and ball
        ball_dist = np.hypot(x_ball - x, y_ball - y) + 1e-6
        pitch_diag = np.hypot(self.pitch_length, self.pitch_width)
        ball_dist_normed = ball_dist / pitch_diag

        # vertical offset relative to the player height
        dz = 2.0 / (1.0 + np.exp(-(z_ball - z))) - 1.0  # [-1, 1]

        result = np.column_stack((ball_dist_normed, dz))

        if self.output == "polars":
            return pl.DataFrame(
                {
                    "ball_dist": result[:, 0],
                    "dz": result[:, 1],
                }
            )

        return result
