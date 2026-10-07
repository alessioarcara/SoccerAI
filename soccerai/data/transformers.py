from typing import Dict, List, Sequence

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin


def _column_names(X, fallback: Sequence[str]) -> List[str]:
    """Return the column names of a DataFrame-like input, or `fallback` for arrays."""
    cols = getattr(X, "columns", None)
    if cols is None:
        return list(fallback)
    return [str(c) for c in cols]


class BaseTransformer(TransformerMixin, BaseEstimator):
    """
    Stateless feature transformer that reads its inputs *by column name*.

    Subclasses declare `input_cols` (required columns, any order in the input)
    and `output_cols`, and implement `_transform(cols)` where `cols` maps each
    input name to a float64 column. Reading by name instead of by position makes
    the transformer independent from the order in which `ColumnTransformer`
    hands over the columns.

    `get_feature_names_out` is defined, so scikit-learn's `set_output` wraps
    the numpy result into a pandas/polars frame when requested.
    """

    input_cols: Sequence[str] = ()
    output_cols: Sequence[str] = ()

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(
            _column_names(X, self.input_cols), dtype=object
        )
        return self

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.output_cols, dtype=object)

    def _columns(self, X) -> Dict[str, np.ndarray]:
        names = _column_names(X, self.input_cols)
        data = np.asarray(
            X.to_numpy() if isinstance(X, pl.DataFrame) else X, dtype=float
        )
        if data.ndim != 2:
            raise ValueError(f"{type(self).__name__} expects a 2D input")
        missing = [c for c in self.input_cols if c not in names]
        if missing:
            raise ValueError(
                f"{type(self).__name__} is missing input columns {missing}; got {names}"
            )
        return {c: data[:, names.index(c)] for c in self.input_cols}

    def _transform(self, cols: Dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def transform(self, X) -> np.ndarray:
        return self._transform(self._columns(X))


class PlayerLocationTransformer(BaseTransformer):
    """Scale pitch coordinates to [0, 1]; angles and velocities pass through."""

    input_cols = ("x", "y", "cos", "sin", "vx", "vy")
    output_cols = ("x", "y", "cos", "sin", "vx", "vy")

    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

    def _transform(self, c: Dict[str, np.ndarray]) -> np.ndarray:
        x_normed = np.clip(c["x"] / self.pitch_length, 0.0, 1.0)
        y_normed = np.clip(c["y"] / self.pitch_width, 0.0, 1.0)
        return np.column_stack(
            (x_normed, y_normed, c["cos"], c["sin"], c["vx"], c["vy"])
        )


class GoalLocationTransformer(BaseTransformer):
    """Distance (normalised by the pitch diagonal) and direction to the goal."""

    input_cols = ("x", "y", "x_goal", "y_goal")
    output_cols = ("goal_dist", "goal_cos", "goal_sin")

    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

    @property
    def pitch_diag(self) -> float:
        return float(np.hypot(self.pitch_length, self.pitch_width))

    def _transform(self, c: Dict[str, np.ndarray]) -> np.ndarray:
        dx = c["x_goal"] - c["x"]
        dy = c["y_goal"] - c["y"]

        goal_dist = np.hypot(dx, dy) + 1e-6
        goal_dist_normed = goal_dist / self.pitch_diag

        goal_cos = dx / goal_dist
        goal_sin = dy / goal_dist

        return np.column_stack((goal_dist_normed, goal_cos, goal_sin))


class BallLocationTransformer(BaseTransformer):
    """
    Player-ball relations: planar distance (normalised), vertical offset of the
    ball w.r.t. the player's height (in [-1, 1]), cosine similarity between
    the ball and player directions, and the ball-player velocity difference.
    """

    input_cols = (
        "x",
        "y",
        "height_cm",
        "cos",
        "sin",
        "vx",
        "vy",
        "x_ball",
        "y_ball",
        "z_ball",
        "cos_ball",
        "sin_ball",
        "vx_ball",
        "vy_ball",
    )
    output_cols = ("ball_dist", "dz", "ball_direction_sim", "dvx", "dvy")

    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

    @property
    def pitch_diag(self) -> float:
        return float(np.hypot(self.pitch_length, self.pitch_width))

    def _transform(self, c: Dict[str, np.ndarray]) -> np.ndarray:
        player_height_m = c["height_cm"] / 100.0

        # planar distance between player and ball
        ball_dist = np.hypot(c["x_ball"] - c["x"], c["y_ball"] - c["y"]) + 1e-6
        ball_dist_normed = ball_dist / self.pitch_diag

        # vertical offset of the ball relative to the player height, in [-1, 1]
        dz = 2.0 / (1.0 + np.exp(-(c["z_ball"] - player_height_m))) - 1.0

        # cosine similarity between ball direction and player direction
        ball_direction_sim = c["cos_ball"] * c["cos"] + c["sin_ball"] * c["sin"]

        # difference between the ball speed and each player speed
        dvx = c["vx_ball"] - c["vx"]
        dvy = c["vy_ball"] - c["vy"]

        return np.column_stack((ball_dist_normed, dz, ball_direction_sim, dvx, dvy))


class ClippedScaler(BaseTransformer):
    """
    Clip every column to [-max_abs, max_abs] and divide by max_abs.

    Velocities are scaled this way instead of with a fitted power transform:
    the mapping is odd (f(-v) = -f(v)), so mirroring the pitch during
    augmentation (vx -> -vx) keeps the features exactly on-distribution.
    Column names are preserved.
    """

    def __init__(self, max_abs: float = 1.0):
        self.max_abs = max_abs

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.feature_names_in_, dtype=object)

    def transform(self, X) -> np.ndarray:
        data = np.asarray(
            X.to_numpy() if isinstance(X, pl.DataFrame) else X, dtype=float
        )
        return np.clip(data, -self.max_abs, self.max_abs) / self.max_abs


class NonPossessionShootingStatsMask(BaseTransformer):
    """
    Zero the shooting statistics of players that are not in possession.

    The mask column is `is_possession_team_1` when present, otherwise the last
    input column. Output columns keep the input names.
    """

    MASK_COL = "is_possession_team_1"

    def fit(self, X, y=None):
        super().fit(X, y)
        names = list(self.feature_names_in_)
        self.mask_col_ = self.MASK_COL if self.MASK_COL in names else names[-1]
        self.feature_cols_ = [c for c in names if c != self.mask_col_]
        return self

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.feature_cols_ + [self.mask_col_], dtype=object)

    def transform(self, X) -> np.ndarray:
        names = _column_names(X, list(self.feature_names_in_))
        data = np.asarray(
            X.to_numpy() if isinstance(X, pl.DataFrame) else X, dtype=float
        )
        mask = data[:, [names.index(self.mask_col_)]]
        features = data[:, [names.index(c) for c in self.feature_cols_]]
        return np.concatenate([features * mask, mask], axis=1)
