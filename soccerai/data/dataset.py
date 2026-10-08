import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import Compose

from soccerai.data.config import SHOOTING_STATS, X_GOAL_LEFT, X_GOAL_RIGHT, Y_GOAL
from soccerai.data.converters import GraphConverter
from soccerai.data.transformers import (
    BallLocationTransformer,
    ClippedScaler,
    GoalLocationTransformer,
    NonPossessionShootingStatsMask,
    PlayerLocationTransformer,
)
from soccerai.training.trainer_config import DataConfig
from soccerai.training.transforms import RandomHorizontalFlip, RandomVerticalFlip


def home_attacks_right_expr() -> pl.Expr:
    """
    Polars counterpart of `soccerai.data.utils.home_attacks_right`: whether the
    home team attacks towards x = pitch length in the row's game period.
    """
    start_left_extra_time = pl.col("homeTeamStartLeftExtraTime").fill_null(
        pl.col("homeTeamStartLeft")
    )
    start_left = (
        pl.when(pl.col("period").is_in([3, 4]))
        .then(start_left_extra_time)
        .otherwise(pl.col("homeTeamStartLeft"))
        .cast(pl.Boolean)
    )
    first_period_of_pair = pl.col("period").is_in([1, 3])
    return start_left == first_period_of_pair


class WorldCup2022Dataset(InMemoryDataset):
    # player height used for the ball `dz` feature when roster data is unused
    NOMINAL_HEIGHT_CM = 180.0
    # velocities are clipped at these speeds (m/s) and scaled to [-1, 1]
    MAX_PLAYER_SPEED = 12.0
    MAX_BALL_RELATIVE_SPEED = 35.0
    # Bump when the preprocessing code changes in a way that must invalidate
    # previously processed files.
    PROCESSING_VERSION = 3
    # rosters were scraped in spring 2025, the tournament was played in Nov 2022
    AGE_SCRAPE_TO_TOURNAMENT_YEARS = 2.5

    def __init__(
        self,
        root: str,
        converter: GraphConverter,
        split: str,
        cfg: DataConfig,
        force_reload: bool = False,
        random_state: int = 0,
    ):
        self.converter = converter
        self.split = split
        self.cfg = cfg
        self.random_state = random_state
        super().__init__(root=root, transform=None, force_reload=force_reload)

        data_path_idx = 0 if self.split == "train" else 1
        self.load(self.processed_paths[data_path_idx])

        fp = Path(self.processed_paths[2])
        self.feature_names: Sequence[str] = json.loads(fp.read_text(encoding="utf-8"))

        self.transform = (
            Compose(self._build_augmentations())
            if split == "train" and self.cfg.use_augmentations
            else None
        )

    def _build_augmentations(self) -> list[Callable]:
        augmentations: list[Callable] = [RandomVerticalFlip(self.feature_names, 0.5)]
        # mirroring the pitch along its length would reverse the attacking
        # direction, which is fixed once the frames are normalised
        if not self.cfg.normalize_attack_direction:
            augmentations.append(RandomHorizontalFlip(self.feature_names, 0.5))
        return augmentations

    @property
    def raw_file_names(self) -> list[str]:
        return ["dataset.parquet"]

    @property
    def config_tag(self) -> str:
        """
        Short hash of everything that determines the processed files, so that
        a change of data configuration (or of the preprocessing code) can never
        silently reuse stale caches. Options applied only at load time
        (augmentations) are left out, so toggling them reuses the cache.
        """
        payload = json.dumps(
            {
                "data": self.cfg.model_dump(exclude={"use_augmentations"}),
                "converter": type(self.converter).__name__,
                "random_state": self.random_state,
                "version": self.PROCESSING_VERSION,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]

    @property
    def processed_file_names(self) -> list[str]:
        tag = self.config_tag
        return [f"train_{tag}.pt", f"val_{tag}.pt", f"feature_names_{tag}.json"]

    @property
    def num_global_features(self) -> int:
        try:
            return self[0].u.shape[-1]
        except (IndexError, AttributeError):
            return 0

    def _split_games(
        self, df: pl.DataFrame, all_game_ids: Sequence[int]
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Split the frames into training and validation sets by game, so that
        no chain (and no frame) of a game can appear in both.

        - "chronological": the last `val_ratio` of the games (ids grow with
          time, so with 0.25 the 48 group-stage games train and the 16
          knock-out games validate), decided on `all_game_ids` before any
          game is dropped.
        - "random": a seeded random `val_ratio` subset of the games present
          in `df` (after any game is dropped).
        """
        present = sorted(df.select("gameId").unique()["gameId"].to_list())

        if self.cfg.split_mode == "chronological":
            n_val = max(1, round(self.cfg.val_ratio * len(all_game_ids)))
            val_games = set(sorted(all_game_ids)[len(all_game_ids) - n_val :])
        elif self.cfg.split_mode == "random":
            if len(present) < 2:
                raise ValueError(f"Cannot split {len(present)} game(s) in two")
            n_val = min(
                max(1, round(self.cfg.val_ratio * len(present))), len(present) - 1
            )
            rng = np.random.default_rng(self.random_state)
            val_games = set(rng.permutation(present)[:n_val].tolist())
        else:
            raise ValueError(f"Unknown split mode: {self.cfg.split_mode}")

        train_df = df.filter(~pl.col("gameId").is_in(list(val_games)))
        val_df = df.filter(pl.col("gameId").is_in(list(val_games)))
        return train_df, val_df

    @staticmethod
    def _drop_games_without_negatives(df: pl.DataFrame) -> pl.DataFrame:
        """
        Games whose chains are all positive would only shift the class prior
        of their split (this happened to the extra-time matches, for which
        the negative-chain selection used to fail).
        """
        has_negatives = (pl.col("label") == 0).any().over("gameId")
        dropped = df.filter(~has_negatives).select("gameId").unique()["gameId"]
        if dropped.len() > 0:
            logger.warning(
                "Dropping {} game(s) without negative chains: {}",
                dropped.len(),
                sorted(dropped.to_list()),
            )
        return df.filter(has_negatives)

    def _filter_positive_chains(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply to the positive chains the same selection used for the
        negatives: the ball carrier of the last frame (the action before the
        shot) must be within `goal_window_for_positives` metres of the
        attacked goal line. Without it "last action far from goal" is a
        shortcut for the positive class.
        """
        window = self.cfg.goal_window_for_positives
        if window is None:
            return df

        last_carrier = (
            df.filter((pl.col("label") == 1) & (pl.col("is_ball_carrier") == 1))
            .sort("event_index")
            .group_by("chain_id")
            .last()
        )
        kept = last_carrier.filter((pl.col("x_goal") - pl.col("x")).abs() <= window)[
            "chain_id"
        ]
        n_positive = df.filter(pl.col("label") == 1)["chain_id"].n_unique()
        no_carrier = n_positive - last_carrier.height
        if no_carrier > 0:
            logger.warning(
                "Dropping {} positive chain(s) with no ball carrier in any frame",
                no_carrier,
            )
        logger.info(
            "Positive chains ending within {} m of the goal line: {} kept, {} dropped",
            window,
            kept.len(),
            n_positive - kept.len(),
        )
        return df.filter((pl.col("label") == 0) | pl.col("chain_id").is_in(kept))

    @staticmethod
    def _log_split(name: str, df: pl.DataFrame) -> None:
        chains = df.group_by("chain_id").agg(pl.col("label").first())
        logger.info(
            "{} split: {} games, {} frames, {} chains ({:.1%} positive)",
            name,
            df.select("gameId").n_unique(),
            df.select(["gameEventId", "possessionEventId"]).n_unique(),
            chains.height,
            chains["label"].mean() if chains.height else float("nan"),
        )

    def _prepare_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        if "period" not in df.columns:
            raise ValueError(
                "The dataset has no `period` column: run "
                "`scripts/patch_dataset_period.py` (or rebuild it) first"
            )
        valid_period = pl.col("period").is_in([1, 2, 3, 4])
        n_invalid = df.filter(~valid_period.fill_null(False)).height
        if n_invalid:
            # no attacking side (penalty shoot-out, missing period)
            logger.warning("Dropping {} row(s) with no valid period", n_invalid)
            df = df.filter(valid_period.fill_null(False))
        if "homeTeamStartLeftExtraTime" not in df.columns:
            df = df.with_columns(
                pl.lit(None, dtype=pl.Boolean).alias("homeTeamStartLeftExtraTime")
            )

        df = self._disambiguate_chains(df)
        # the raw event index orders the frames of a chain (the game clock
        # has a 1 s resolution and is often tied within a chain)
        df = df.rename({"index": "event_index"})

        cols_to_drop = [
            "gameEventType",
            "startTime",
            "endTime",
            "index_right",
            "gameId_right",
            "visibility",
            "videoUrl",
            "homeTeamName",
            "awayTeamName",
            "Full Name",
            "Height",
            "birth_date",
            "teamName",
            "playerId",
        ]
        df = df.drop(cols_to_drop)

        # `age` is only available for the players found on Transfermarkt; the
        # FBref "Age Info" string ("(Age: 27-172d)") covers most of the others.
        # Both are ages at scraping time, so they are shifted back to the
        # tournament date before bucketing. Unknown ages get their own bucket
        # instead of silently falling into the oldest one.
        age_from_info = (
            pl.col("Age Info").str.extract(r"Age:\s*(\d+)", 1).cast(pl.Float64)
        )
        df = df.with_columns(
            (
                pl.col("age").fill_null(age_from_info)
                - self.AGE_SCRAPE_TO_TOURNAMENT_YEARS
            ).alias("age")
        ).drop("Age Info")

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
            )
        # the players' own height above the pitch is (near) constant noise
        df = df.drop("z", strict=False)

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
                    pl.when(pl.col("age").is_null())
                    .then(pl.lit("unknown"))
                    .when(pl.col("age") < 20)
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

        if not self.cfg.use_match_clock:
            df = df.drop("frameTime")

        if not self.cfg.use_roster_features:
            # per-player constants (weight, market value, shooting record, age)
            # identify the player: without them the model has to rely on what
            # happens on the pitch. The height only feeds the ball `dz`
            # feature, so a nominal height is kept.
            df = df.drop(["Weight", "Market Value", "age", *SHOOTING_STATS])
            df = df.with_columns(pl.lit(self.NOMINAL_HEIGHT_CM).alias("height_cm"))

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

        # The goal that matters for a shot is the one attacked by the team in
        # possession: every node gets that goal (defenders included). A team
        # attacks to the right iff it is the home team and the home team
        # attacks to the right in the current period, or vice versa.
        frame_key = ["gameEventId", "possessionEventId"]
        attacks_right = (pl.col("team") == "home") == home_attacks_right_expr()
        possession_attacks_right = (
            pl.when(pl.col("is_possession_team") == 1)
            .then(attacks_right)
            .max()
            .over(frame_key)
            .alias("possession_attacks_right")
        )
        df = df.with_columns(possession_attacks_right)

        if self.cfg.normalize_attack_direction:
            df = self._normalize_attack_direction(df)

        df = df.with_columns(
            [
                pl.when(pl.col("possession_attacks_right"))
                .then(X_GOAL_RIGHT)
                .otherwise(X_GOAL_LEFT)
                .alias("x_goal"),
                pl.lit(Y_GOAL).alias("y_goal"),
            ]
        ).drop("possession_attacks_right")

        df = self._filter_positive_chains(df)

        df = df.drop(
            [
                c
                for c in [
                    "team",
                    "homeTeamStartLeft",
                    "homeTeamStartLeftExtraTime",
                    "startPeriod2",
                    "period",
                ]
                if c in df.columns
            ]
        )

        return df

    @staticmethod
    def _normalize_attack_direction(df: pl.DataFrame) -> pl.DataFrame:
        """
        Mirror the frames in which the possession team attacks to the left so
        that the attack always goes towards x = pitch length: positions,
        headings and velocities of players and ball are flipped together.
        """
        flip = ~pl.col("possession_attacks_right")
        mirrored = [
            pl.when(flip).then(X_GOAL_RIGHT - pl.col(c)).otherwise(pl.col(c)).alias(c)
            for c in ["x", "x_ball"]
            if c in df.columns
        ]
        negated = [
            pl.when(flip).then(-pl.col(c)).otherwise(pl.col(c)).alias(c)
            for c in ["cos", "vx", "cos_ball", "vx_ball"]
            if c in df.columns
        ]
        return df.with_columns(mirrored + negated).with_columns(
            pl.lit(True).alias("possession_attacks_right")
        )

    @staticmethod
    def _disambiguate_chains(df: pl.DataFrame) -> pl.DataFrame:
        """
        Make every frame belong to exactly one chain and every negative chain
        shot-free.

        Two positive chains overlap when a possession contains two shots (the
        second chain extends back past the first shot); the shared frames are
        kept in the chain whose shot comes first. Negative chains that contain
        a shot (possible when the shot's own chain was too short to be kept)
        are dropped as label noise.
        """
        frame_key = ["gameEventId", "possessionEventId"]

        chain_end = pl.col("index").max().over("chain_id")
        df = df.with_columns(chain_end.alias("_chain_end"))
        df = df.filter(
            pl.col("_chain_end") == pl.col("_chain_end").min().over(frame_key)
        ).drop("_chain_end")

        chain_has_shot = (pl.col("possessionEventType") == "SH").any().over("chain_id")
        return df.filter(~((pl.col("label") == 0) & chain_has_shot))

    def _create_preprocessor(self, df: pl.DataFrame) -> ColumnTransformer | Pipeline:
        # Column groups --------------------------------------------------- #
        cat_cols = [
            c
            for c in [
                "possessionEventType",
                "playerRole",
                "is_possession_team",
                "is_ball_carrier",
                "age",
            ]
            if c in df.columns
        ]
        # Identifier columns are passed through untouched; the converter uses
        # them to group rows into graphs and then drops them.
        id_cols = [
            "gameEventId",
            "possessionEventId",
            "event_index",
            "label",
            "gameId",
            "chain_id",
            "jerseyNum",
        ]
        pos_cols = ["x", "y"]
        goal_cols = ["x_goal", "y_goal"]
        angle_cols = ["cos", "sin"]
        velocity_cols = ["vx", "vy"]
        ball_cols = [
            "x_ball",
            "y_ball",
            "z_ball",
            "cos_ball",
            "sin_ball",
            "vx_ball",
            "vy_ball",
        ]
        exclude_cols: set[str] = {
            *cat_cols,
            *id_cols,
            *pos_cols,
            *goal_cols,
            *angle_cols,
            *velocity_cols,
        }
        if self.cfg.include_ball_features:
            # the player height is consumed by the ball pipeline (`dz`)
            exclude_cols.update(ball_cols + ["height_cm"])
        elif not self.cfg.use_roster_features:
            # the nominal height is a constant, not a feature
            exclude_cols.add("height_cm")
        num_cols = [c for c in df.columns if c not in exclude_cols]

        # Pipelines ------------------------------------------------------- #
        if self.cfg.use_regression_imputing:
            et_est = ExtraTreesRegressor(
                n_estimators=100,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=self.random_state,
            )
            imputer = IterativeImputer(
                estimator=et_est,
                max_iter=10,
                initial_strategy="median",
                random_state=self.random_state,
            )
        else:
            imputer = SimpleImputer(strategy="median")

        numeric_steps = [
            ("imputer", imputer),
            (
                "scaler",
                QuantileTransformer(
                    output_distribution="normal", random_state=self.random_state
                ),
            ),
        ]

        use_shooting_stats = self.cfg.use_roster_features
        if self.cfg.use_pca_on_roster_cols and use_shooting_stats:
            numeric_steps.append(
                (
                    "shooting_stats_pca",
                    ColumnTransformer(
                        [
                            (
                                "subset",
                                PCA(n_components=0.99, random_state=self.random_state),
                                SHOOTING_STATS,
                            )
                        ],
                        remainder="passthrough",
                        verbose_feature_names_out=False,
                    ),
                ),
            )
        num_pipe = Pipeline(steps=numeric_steps)
        cat_pipe = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="constant", fill_value="unknown"),
                ),
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
                        [("clip", ClippedScaler(self.MAX_PLAYER_SPEED), ["vx", "vy"])],
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
            ("player_loc", player_pipe, pos_cols + angle_cols + velocity_cols),
            ("ids", "passthrough", id_cols),
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
                    ("imputer", SimpleImputer(strategy="median")),
                    ("ball_loc", BallLocationTransformer()),
                    (
                        "diff_speed_norm",
                        ColumnTransformer(
                            [
                                (
                                    "clip",
                                    ClippedScaler(self.MAX_BALL_RELATIVE_SPEED),
                                    ["dvx", "dvy"],
                                )
                            ],
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
                    pos_cols + ["height_cm"] + angle_cols + velocity_cols + ball_cols,
                )
            )

        # Every column that reaches the model must be produced by one of the
        # transformers above: unlisted columns (e.g. raw goal coordinates when
        # goal features are disabled) are dropped instead of leaking through.
        prep = ColumnTransformer(
            transformers,
            remainder="drop",
            verbose_feature_names_out=False,  # No prefixes
        )

        if self.cfg.mask_non_possession_shooting_stats and use_shooting_stats:
            if self.cfg.use_pca_on_roster_cols:

                def cols_to_mask(df: pl.DataFrame) -> list[str]:
                    pca_cols = [c for c in df.columns if c.startswith("pca")]
                    return pca_cols + ["is_possession_team_1"]
            else:
                cols_to_mask = SHOOTING_STATS + ["is_possession_team_1"]  # type: ignore

            prep = Pipeline(
                [
                    ("prep", prep),
                    (
                        "shooting_stats_mask",
                        ColumnTransformer(
                            [
                                (
                                    "mask",
                                    NonPossessionShootingStatsMask(),
                                    cols_to_mask,
                                )
                            ],
                            remainder="passthrough",
                            verbose_feature_names_out=False,
                        ),
                    ),
                ]
            )

        prep.set_output(transform="polars")
        return prep

    def process(self):
        raw_df = pl.read_parquet(self.raw_paths[0])
        all_game_ids = sorted(raw_df.select("gameId").unique()["gameId"].to_list())

        df = self._prepare_dataframe(raw_df)
        if self.cfg.drop_games_without_negatives:
            df = self._drop_games_without_negatives(df)

        train_df, val_df = self._split_games(df, all_game_ids)
        self._log_split("train", train_df)
        self._log_split("val", val_df)

        preprocessor = self._create_preprocessor(train_df)

        train_transformed = preprocessor.fit_transform(train_df)
        val_transformed = preprocessor.transform(val_df)

        train_data_list, feature_names = self.converter.convert_dataframe_to_data_list(
            train_transformed
        )

        val_data_list, _ = self.converter.convert_dataframe_to_data_list(
            val_transformed
        )

        self.save(train_data_list, self.processed_paths[0])
        self.save(val_data_list, self.processed_paths[1])

        fp = Path(self.processed_paths[2])
        fp.write_text(json.dumps(feature_names, ensure_ascii=False, indent=4))
