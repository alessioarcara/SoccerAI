from typing import List, Optional, Sequence, Union

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin


class PossessionShootingMask(BaseEstimator, TransformerMixin):
    def __init__(
        self, shoot_cols: List[str], possess_col: str = "is_possession_team_1"
    ):
        self.shoot_cols = shoot_cols
        self.possess_col = possess_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pl.DataFrame):
            raise TypeError(
                "PossessionShootingMask expects a polars.DataFrame as input"
            )
        mask = X[self.possess_col]
        updates = [(X[c] * mask).alias(c) for c in self.shoot_cols]
        return X.with_columns(updates)


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

    def _maybe_polars(
        self, data: np.ndarray, columns: Sequence[str]
    ) -> Union[np.ndarray, pl.DataFrame]:
        if self.output == "polars":
            return pl.DataFrame({c: data[:, i] for i, c in enumerate(columns)})
        return data


class PlayerLocationTransformer(BaseTransformer):
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
        cos = coords[:, 2]
        sin = coords[:, 3]
        vx = coords[:, 4]
        vy = coords[:, 5]

        x_normed = np.clip(x / self.pitch_length, 0.0, 1.0)
        y_normed = np.clip(y / self.pitch_width, 0.0, 1.0)

        res = np.column_stack((x_normed, y_normed, cos, sin, vx, vy))
        return self._maybe_polars(res, ["x", "y", "cos", "sin", "vx", "vy"])


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
        self.pitch_diag = float(np.hypot(pitch_length, pitch_width))

    def transform(self, X):
        coords = np.asarray(X, dtype=float)
        x = coords[:, 0]
        y = coords[:, 1]
        x_goal = coords[:, 2]
        y_goal = coords[:, 3]

        dx = x_goal - x
        dy = y_goal - y

        goal_dist = np.hypot(dx, dy) + 1e-6
        goal_dist_normed = goal_dist / self.pitch_diag

        goal_cos = dx / goal_dist
        goal_sin = dy / goal_dist

        res = np.column_stack((goal_dist_normed, goal_cos, goal_sin))
        return self._maybe_polars(res, ["goal_dist", "goal_cos", "goal_sin"])


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
        self.pitch_diag = float(np.hypot(pitch_length, pitch_width))

    def transform(self, X):
        coords = np.asarray(X, dtype=float)
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2] / 100.0  # cm -> m
        cos = coords[:, 3]
        sin = coords[:, 4]
        vx = coords[:, 5]
        vy = coords[:, 6]
        x_ball = coords[:, 7]
        y_ball = coords[:, 8]
        z_ball = coords[:, 9]
        cos_ball = coords[:, 10]
        sin_ball = coords[:, 11]
        vx_ball = coords[:, 12]
        vy_ball = coords[:, 13]

        # planar distance between player and ball
        ball_dist = np.hypot(x_ball - x, y_ball - y) + 1e-6
        ball_dist_normed = ball_dist / self.pitch_diag

        # vertical offset relative to the player height
        dz = 2.0 / (1.0 + np.exp(-(z_ball - z))) - 1.0  # [-1, 1]

        # cosine similarity between ball direction and players directions
        ball_direction_sim = cos_ball * cos + sin_ball * sin

        # difference between each player speed and the ball speed
        dvx = vx_ball - vx
        dvy = vy_ball - vy

        res = np.column_stack((ball_dist_normed, dz, ball_direction_sim, dvx, dvy))
        return self._maybe_polars(
            res, ["ball_dist", "dz", "ball_direction_sim", "dvx", "dvy"]
        )
