"""
Synthetic data factories shared by the unit tests.

`make_raw_df` builds a polars DataFrame with the same schema as
`soccerai/data/resources/raw/dataset.parquet` (one row per player plus one
row for the ball, for every frame), so that the dataset pipeline can be
exercised end to end without the real data.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import polars as pl

PLAYERS_PER_TEAM = 11
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

SHOOTING_STATS = [
    "goals",
    "shots",
    "shots_on_target",
    "shots_on_target_pct",
    "shots_per90",
    "shots_on_target_per90",
    "goals_per_shot",
    "goals_per_shot_on_target",
    "average_shot_distance",
    "pens_made",
    "pens_att",
]

ROLES = ["GK", "RCB", "LCB", "RB", "LB", "DM", "CM", "AM", "RW", "LW", "CF"]


def _seconds_to_clock(seconds: int) -> str:
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def make_frame_rows(
    *,
    game_id: int,
    game_event_id: int,
    possession_event_id: float,
    event_index: int,
    chain_id: int,
    label: int,
    period: int = 1,
    home_start_left: bool = True,
    home_start_left_et: Optional[bool] = None,
    possession: str = "home",
    carrier_idx: int = 9,
    event_type: str = "PA",
    clock_seconds: int = 600,
    rng: Optional[np.random.Generator] = None,
    positions: Optional[Dict[str, np.ndarray]] = None,
    ball: Optional[Sequence[float]] = None,
) -> List[dict]:
    """Rows (22 players + ball) of one frame, in the raw parquet schema."""
    rng = rng if rng is not None else np.random.default_rng(0)
    home_name, away_name = f"Home{game_id}", f"Away{game_id}"
    start_period2 = 3000.0 if period <= 2 else None

    rows: List[dict] = []
    for team, team_name in (("home", home_name), ("away", away_name)):
        for i in range(PLAYERS_PER_TEAM):
            if positions is not None and team in positions:
                x, y = positions[team][i]
            else:
                x = float(rng.uniform(0.0, PITCH_LENGTH))
                y = float(rng.uniform(0.0, PITCH_WIDTH))
            is_carrier = team == possession and i == carrier_idx
            player_name = f"{team_name}_{i}"
            age = float(20 + i) if i % 3 else None  # some null ages, like the data
            rows.append(
                {
                    "index": event_index,
                    "gameId": game_id,
                    "gameEventId": game_event_id,
                    "possessionEventId": possession_event_id,
                    "startTime": 100.0 + event_index,
                    "endTime": 101.0 + event_index,
                    "duration": 1.0,
                    "gameEventType": "OTB",
                    "possessionEventType": event_type,
                    "teamName": team_name,
                    "playerName": None,  # set below
                    "videoUrl": "",
                    "frameTime": _seconds_to_clock(clock_seconds),
                    "period": period,
                    "chain_id": chain_id,
                    "label": label,
                    "index_right": event_index,
                    "gameId_right": game_id,
                    "team": team,
                    "x": x,
                    "y": y,
                    "z": 0.0,
                    "jerseyNum": str(i + 1),
                    "visibility": None,
                    "velocity": float(rng.uniform(0.0, 8.0)),
                    "direction": float(rng.uniform(-180.0, 180.0)),
                    "homeTeamName": home_name,
                    "awayTeamName": away_name,
                    "homeTeamStartLeft": home_start_left,
                    "homeTeamStartLeftExtraTime": home_start_left_et,
                    "startPeriod2": start_period2,
                    "playerId": float(1000 * (team == "away") + i),
                    "playerName_right": player_name,
                    "playerRole": ROLES[i],
                    "Full Name": player_name,
                    "Height": "180cm",
                    "Weight": f"{70 + i}kg",
                    "Age Info": f"(Age: {25 + i}-100d)",
                    **{s: float(i) for s in SHOOTING_STATS},
                    "Market Value": float(1e6 * (i + 1)),
                    "birth_date": "1995-01-01",
                    "age": age,
                    "height_cm": float(170 + i) if i % 2 else None,
                }
            )
            # the raw data stores the event player name in `playerName`; it is
            # equal to the roster name only for the ball carrier
            carrier_team = home_name if possession == "home" else away_name
            rows[-1]["playerName"] = (
                player_name if is_carrier else f"{carrier_team}_{carrier_idx}"
            )

    bx, by, bz = (
        ball
        if ball is not None
        else (
            float(rng.uniform(0.0, PITCH_LENGTH)),
            float(rng.uniform(0.0, PITCH_WIDTH)),
            0.5,
        )
    )
    rows.append(
        {
            **{k: rows[0][k] for k in rows[0]},
            "team": None,
            "x": bx,
            "y": by,
            "z": bz,
            "jerseyNum": None,
            "visibility": "VISIBLE",
            "velocity": 6.0,
            "direction": 45.0,
            "playerId": None,
            "playerName_right": None,
            "playerRole": None,
            "Full Name": None,
            "Height": None,
            "Weight": None,
            "Age Info": None,
            **{s: None for s in SHOOTING_STATS},
            "Market Value": None,
            "birth_date": None,
            "age": None,
            "height_cm": None,
        }
    )
    return rows


def make_raw_df(
    chains: Sequence[dict],
    seed: int = 0,
) -> pl.DataFrame:
    """
    Build a raw-schema DataFrame from chain specs.

    Each spec is a dict with keys: `game_id`, `chain_id`, `label`, `n_frames`
    and optionally `period`, `home_start_left`, `home_start_left_et`,
    `possession`, `event_types` (list, one per frame), `clock_start`.
    """
    rng = np.random.default_rng(seed)
    rows: List[dict] = []
    event_index = 0
    for spec in chains:
        n_frames = spec["n_frames"]
        event_types = spec.get("event_types") or ["PA"] * n_frames
        clock = spec.get("clock_start", 600)
        for t in range(n_frames):
            rows.extend(
                make_frame_rows(
                    game_id=spec["game_id"],
                    game_event_id=event_index,
                    possession_event_id=float(event_index),
                    event_index=event_index,
                    chain_id=spec["chain_id"],
                    label=spec["label"],
                    period=spec.get("period", 1),
                    home_start_left=spec.get("home_start_left", True),
                    home_start_left_et=spec.get("home_start_left_et"),
                    possession=spec.get("possession", "home"),
                    event_type=event_types[t],
                    clock_seconds=clock + 3 * t,
                    rng=rng,
                )
            )
            event_index += 1

    df = pl.DataFrame(rows, infer_schema_length=None)
    return df.with_columns(
        pl.col("index").cast(pl.UInt32),
        pl.col("index_right").cast(pl.UInt32),
        pl.col("label").cast(pl.Int32),
    )
