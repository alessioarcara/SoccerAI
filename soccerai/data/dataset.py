import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Optional

import polars as pl
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from torch_geometric.data import InMemoryDataset
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal

from soccerai.data.config import X_GOAL_LEFT, X_GOAL_RIGHT, Y_GOAL
from soccerai.data.converters import GraphConverter
from soccerai.data.transformers import (
    BallLocationTransformer,
    GoalLocationTransformer,
    PlayerLocationTransformer,
)
from soccerai.data.utils import reorder_dataframe_cols
from soccerai.training.trainer_config import DataConfig


class WorldCup2022Dataset(InMemoryDataset):
    FEATURE_NAMES_FILE = "feature_names.json"

    def __init__(
        self,
        root: str,
        converter: GraphConverter,
        split: str,
        cfg: DataConfig,
        transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        self.converter = converter
        self.split = split
        self.cfg = cfg
        super().__init__(root=root, transform=transform, force_reload=force_reload)

        data_path_idx = 0 if self.split == "train" else 1
        self.load(self.processed_paths[data_path_idx])

        fp = Path(self.processed_dir) / self.FEATURE_NAMES_FILE
        self.feature_names = (
            json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else None
        )

    @property
    def raw_file_names(self) -> List[str]:
        return ["dataset.parquet"]

    @property
    def processed_file_names(self) -> List[str]:
        return ["train_data.pt", "val_data.pt"]

    @property
    def num_global_features(self) -> int:
        try:
            return self[0].u.shape[-1]
        except (IndexError, AttributeError):
            return 0

    def _split_by_worldcup_phase(
        self, df: pl.DataFrame, val_ratio: float
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Split the DataFrame into training and validation sets.

        The split accounts the two phases of a FIFA World Cup:
            * 48 group-stage games -> training set
            * 16 knock-out games -> validation set
        """
        game_ids_df = df.select(["gameId"]).unique().sort("gameId")
        n_games = game_ids_df.height

        n_train_games = int((1.0 - val_ratio) * n_games)

        train_game_ids = game_ids_df.slice(0, n_train_games)
        val_game_ids = game_ids_df.slice(n_train_games, n_games)

        train_df = df.join(train_game_ids, on="gameId", how="semi")
        val_df = df.join(val_game_ids, on="gameId", how="semi")

        return train_df, val_df

    def _prepare_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        cols_to_drop = [
            "gameEventType",
            "index",
            "startTime",
            "endTime",
            "index_right",
            "gameId_right",
            "jerseyNum",
            "visibility",
            "videoUrl",
            "homeTeamName",
            "awayTeamName",
            "Age Info",
            "Full Name",
            "Height",
            "birth_date",
            "teamName",
            "playerId",
        ]
        df = df.drop(cols_to_drop)

        df = (
            df.with_columns(
                (pl.col("direction").radians().cos().alias("cos")),
                (pl.col("direction").radians().sin().alias("sin")),
            )
            .with_columns(
                pl.col("velocity").mul(pl.col("cos")).alias("vx"),
                pl.col("velocity").mul(pl.col("sin")).alias("vy"),
            )
            .drop(["velocity", "direction"])
        )

        if self.cfg.include_ball_features:
            df = df.with_columns(
                *[
                    pl.col(c)
                    .filter(pl.col("team").is_null())
                    .first()
                    .over("gameEventId", "possessionEventId")
                    .alias(f"{c}_ball")
                    for c in ["x", "y", "z", "cos", "sin", "vx", "vy"]
                ]
            ).drop("z")

        df = (
            df.filter(pl.col("team").is_not_null())
            .filter(pl.col("possessionEventType") != "SH")
            .filter(pl.col("playerName").is_not_null())
        )

        df = df.with_columns(
            [
                # (mm:ss) → s
                (
                    (
                        pl.col("frameTime").str.split(":").list.get(0).cast(pl.UInt16)
                        * 60
                        + pl.col("frameTime").str.split(":").list.get(1).cast(pl.UInt16)
                    ).alias("frameTime")
                ),
                (
                    pl.when(pl.col("playerName") == pl.col("playerName_right"))
                    .then(1)
                    .otherwise(0)
                ).alias("is_ball_carrier"),
                (pl.col("Weight").str.replace("kg", "").cast(pl.Float64)),
                (pl.col("height_cm").cast(pl.Float64)),
                (
                    pl.when(pl.col("age") < 20)
                    .then(pl.lit("Under 20"))
                    .when(pl.col("age") < 29)
                    .then(pl.lit("20-28"))
                    .when(pl.col("age") < 35)
                    .then(pl.lit("29-35"))
                    .otherwise(pl.lit("35+"))
                    .alias("age")
                ),
            ]
        ).drop(["playerName", "playerName_right"])

        if self.cfg.use_macro_roles:
            df = df.with_columns(
                pl.when(
                    pl.col("playerRole").is_in(["DM", "CM", "AM", "LM", "RM", "MCB"])
                )
                .then(pl.lit("M"))
                .when(
                    pl.col("playerRole").is_in(["RCB", "LCB", "LB", "RB", "LWB", "RWB"])
                )
                .then(pl.lit("D"))
                .when(pl.col("playerRole").is_in(["RW", "CF", "LW"]))
                .then(pl.lit("F"))
                .otherwise(pl.col("playerRole"))
                .alias("playerRole")
            )

        df = (
            df.with_columns(
                pl.col("team")
                .filter(pl.col("is_ball_carrier") == 1)
                .first()
                .over(["gameEventId", "possessionEventId"])
                .alias("possession_team_tmp")
            )
            .with_columns(
                (pl.col("team") == pl.col("possession_team_tmp"))
                .cast(pl.Int8)
                .alias("is_possession_team")
            )
            .drop("possession_team_tmp")
        ).drop_nulls(["is_possession_team"])

        is_home_team = df["team"] == "home"
        is_second_half = df["frameTime"] > df["startPeriod2"]

        is_goal_right = (
            (is_home_team & df["homeTeamStartLeft"] & ~is_second_half)
            | (is_home_team & ~df["homeTeamStartLeft"] & is_second_half)
            | (~is_home_team & ~df["homeTeamStartLeft"] & ~is_second_half)
            | (~is_home_team & df["homeTeamStartLeft"] & is_second_half)
        )
        df = df.with_columns(
            [
                pl.when(is_goal_right)
                .then(X_GOAL_RIGHT)
                .otherwise(X_GOAL_LEFT)
                .alias("x_goal"),
                pl.lit(Y_GOAL).alias("y_goal"),
            ]
        )

        df = df.drop(["team", "homeTeamStartLeft", "startPeriod2"])

        return df

    def _create_preprocessor(self, df: pl.DataFrame) -> ColumnTransformer:
        # Column groups --------------------------------------------------- #
        cat_cols = [
            "possessionEventType",
            "playerRole",
            "is_possession_team",
            "is_ball_carrier",
            "age",
        ]
        pos_cols = ["x", "y"]
        goal_cols = ["x_goal", "y_goal"]
        angle_cols = ["cos", "sin"]
        velocity_cols = ["vx", "vy"]
        exclude_cols: set[str] = {
            *cat_cols,
            *pos_cols,
            *goal_cols,
            *angle_cols,
            *velocity_cols,
            "gameEventId",
            "possessionEventId",
            "label",
            "gameId",
            "chain_id",
        }
        if self.cfg.include_ball_features:
            ball_cols = [
                "x_ball",
                "y_ball",
                "z_ball",
                "height_cm",
                "cos_ball",
                "sin_ball",
                "vx_ball",
                "vy_ball",
            ]
            exclude_cols.update(ball_cols)
        num_cols = [c for c in df.columns if c not in exclude_cols]

        # Pipelines ------------------------------------------------------- #
        num_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]  # TODO: ?
        )
        cat_pipe = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="constant", fill_value="unknown"),
                ),  # TODO: ?
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore", drop="if_binary", sparse_output=False
                    ),
                ),
            ]
        )
        player_pipe = Pipeline(
            [
                ("imputer", KNNImputer(weights="distance")),
                ("player_loc", PlayerLocationTransformer()),
                (
                    "speed_norm",
                    ColumnTransformer(
                        [("pow", PowerTransformer(), ["vx", "vy"])],
                        remainder="passthrough",
                        verbose_feature_names_out=False,
                    ),
                ),
            ]
        )

        # ColumnTransformer setup ------------------------------------------- #
        transformers = [
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
            (
                "player_loc",
                player_pipe,
                pos_cols + angle_cols + velocity_cols + goal_cols,
            ),
        ]

        if self.cfg.include_goal_features:
            transformers.append(
                (
                    "goal_loc",
                    GoalLocationTransformer(),
                    pos_cols + goal_cols,
                )
            )
        if self.cfg.include_ball_features:
            ball_loc_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),  # TODO: ?
                    ("ball_loc", BallLocationTransformer()),
                    (
                        "diff_speed_norm",
                        ColumnTransformer(
                            [("pow", PowerTransformer(), ["dvx", "dvy"])],
                            remainder="passthrough",
                            verbose_feature_names_out=False,
                        ),
                    ),
                ]
            )

            transformers.append(
                (
                    "ball_pipe",
                    ball_loc_pipe,
                    pos_cols + angle_cols + velocity_cols + ball_cols,
                )
            )

        prep = ColumnTransformer(
            transformers,
            remainder="passthrough",
            verbose_feature_names_out=False,  # No prefixes
        )

        prep.set_output(transform="polars")
        return prep

    def process(self):
        df = self._prepare_dataframe(pl.read_parquet(self.raw_paths[0]))

        train_df, val_df = self._split_by_worldcup_phase(df, self.cfg.val_ratio)

        logger.info(
            "DataFrame split → train: {} rows, val: {} rows",
            train_df.height,
            val_df.height,
        )

        preprocessor = self._create_preprocessor(train_df)

        first = [
            "x",
            "y",
            "is_possession_team_1",
            "is_ball_carrier_1",
            "vx",
            "vy",
            "cos",
            "sin",
        ]
        if self.cfg.include_goal_features:
            first.extend(["goal_cos", "goal_sin"])
        if self.cfg.include_ball_features:
            first.extend(["dvx", "dvy"])

        train_transformed = reorder_dataframe_cols(
            preprocessor.fit_transform(train_df), first
        )
        val_transformed = reorder_dataframe_cols(preprocessor.transform(val_df), first)

        train_data_list, feature_names = self.converter.convert_dataframe_to_data_list(
            train_transformed
        )

        val_data_list, _ = self.converter.convert_dataframe_to_data_list(
            val_transformed
        )

        self.save(train_data_list, self.processed_paths[0])
        self.save(val_data_list, self.processed_paths[1])

        fp = Path(self.processed_dir) / self.FEATURE_NAMES_FILE
        fp.write_text(json.dumps(feature_names, ensure_ascii=False, indent=4))

    def to_temporal_iterators(self) -> List[DynamicGraphTemporalSignal]:
        buckets = defaultdict(list)
        for data in self:
            chain_id = int(data.chain_id.item())
            buckets[chain_id].append(data)

        chains = []
        for chain_id, frames in buckets.items():
            ordered = sorted(frames, key=lambda f: float(f.frame_time.item()))

            edge_indices = [f.edge_index.numpy() for f in ordered]
            node_features = [f.x.numpy() for f in ordered]
            global_features = [f.u.numpy() for f in ordered]
            targets = [f.y.numpy() for f in ordered]
            edge_weights = [f.edge_weight.numpy() for f in ordered]

            chains.append(
                DynamicGraphTemporalSignal(
                    edge_indices=edge_indices,
                    edge_weights=edge_weights,
                    features=node_features,
                    targets=targets,
                    u=global_features,
                )
            )

        return chains
