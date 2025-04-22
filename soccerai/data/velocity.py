import json
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import polars as pl
from numpy.typing import NDArray

from soccerai.data.utils import read_lines_backwards_from_offset


def add_velocity_per_player(tracking_dir_path: str, players_df: pl.DataFrame):
    gameIds = np.unique(players_df.select(pl.col("gameId")).to_numpy())
    gameEventIds = np.unique(players_df.select(pl.col("gameEventId")).to_numpy())

    velocities: List[np.floating | None] = []
    directions: List[np.floating | None] = []

    for gameId in gameIds:
        tracking_file = f"{tracking_dir_path}/{gameId}.jsonl"
        event_byte_map = create_event_byte_map(tracking_file)

        for gameEventId in gameEventIds:
            byte_pos = event_byte_map.get(gameEventId)

            players_per_event = players_df.filter(pl.col("gameEventId") == gameEventId)

            if byte_pos is None:
                velocities.extend([None] * players_per_event.height)
                directions.extend([None] * players_per_event.height)
                continue

            time_elapsed, ball_delta, home_players_deltas, away_players_deltas = (
                extract_tracking_data(tracking_file, byte_pos)
            )

            if (
                time_elapsed is None
                or home_players_deltas is None
                or away_players_deltas is None
                or ball_delta is None
            ):
                velocities.extend([None] * players_per_event.height)
                directions.extend([None] * players_per_event.height)
                continue

            for row in players_per_event.iter_rows(named=True):
                delta = None
                team = row["team"]
                if team == "home":
                    delta = home_players_deltas[row["jerseyNum"]]
                elif team == "away":
                    delta = away_players_deltas[row["jerseyNum"]]
                else:
                    delta = ball_delta
                velocity, direction = compute_velocity(delta, time_elapsed)
                velocities.append(velocity)
                directions.append(direction)

    return players_df.with_columns(
        pl.Series("velocity", velocities),
        pl.Series("direction", directions),
    )


def compute_velocity(
    positions_delta: NDArray[np.float64], time_elapsed: np.floating
) -> Tuple[np.floating, np.floating]:
    """
    space_delta: Array of position differences [x, y] or [x, y, z]
    """
    velocity_vector = positions_delta / time_elapsed
    velocity = np.linalg.norm(velocity_vector)
    direction = np.rad2deg(np.arctan2(velocity_vector[1], velocity_vector[0]))
    return velocity, direction


def create_event_byte_map(tracking_file: str) -> Dict[int, int]:
    event_byte_map: Dict[int, int] = {}
    pending_events: Dict[int, int] = {}

    with open(tracking_file, "r") as tracking_data:
        byte_pos = tracking_data.tell()

        while True:
            frame = tracking_data.readline()
            if not frame:
                break

            frame_info = json.loads(frame)
            frame_num = frame_info["frameNum"]
            game_event_id = frame_info["game_event_id"]

            if game_event_id not in event_byte_map:
                # pending events
                if frame_num in pending_events.values():
                    for pending_id, pending_frame in list(pending_events.items()):
                        if frame_num == pending_frame:
                            event_byte_map[pending_id] = byte_pos
                            del pending_events[pending_id]

                # current event
                elif game_event_id is not None:
                    game_event_id = int(game_event_id)

                    if game_event_id not in pending_events:
                        # new event
                        end_frame = frame_info["game_event"]["end_frame"]
                        if frame_num == end_frame:
                            event_byte_map[game_event_id] = byte_pos
                        else:
                            pending_events[game_event_id] = end_frame
                    elif (
                        frame_info["possession_event_id"] is not None
                        and frame_info["ballsSmoothed"] is not None
                    ):
                        event_byte_map[game_event_id] = byte_pos
                        del pending_events[game_event_id]

            byte_pos = tracking_data.tell()

    return event_byte_map


def extract_tracking_data(
    tracking_file: str, byte_pos: int
) -> Union[
    Tuple[
        np.floating,
        NDArray[np.float64],
        Dict[str, NDArray[np.float64]],
        Dict[str, NDArray[np.float64]],
    ],
    Tuple[None, ...],
]:
    time_deltas: List[List[float]] = []
    ball_deltas: List[List[float]] = []
    home_players_deltas: Dict[str, List[List[float]]] = {}
    away_players_deltas: Dict[str, List[List[float]]] = {}
    previous_frame = -1

    frames = read_lines_backwards_from_offset(tracking_file, byte_pos, max_lines=60)
    if len(frames) > 0:
        last_frame_info = json.loads(frames[-1])
    if len(frames) >= 2 and (
        last_frame_info["game_event"]["game_event_type"]
        not in {"SUB", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF"}
    ):
        for frame in frames:
            frame_info = json.loads(frame)
            if is_valid_frame(frame_info, previous_frame):
                ball_deltas, home_players_deltas, away_players_deltas, time_deltas = (
                    extract_frame_info(
                        ball_deltas,
                        home_players_deltas,
                        away_players_deltas,
                        time_deltas,
                        frame_info,
                    )
                )
            previous_frame = frame_info["frameNum"]

    if len(ball_deltas) < 2:
        if len(ball_deltas) == 1:
            ball_deltas = []
            home_players_deltas = {}
            away_players_deltas = {}
            time_deltas = []
        previous_frame = -1
        with open(tracking_file, "r") as file:
            file.seek(byte_pos)
            for _ in range(4):
                line = file.readline()
                if not line:
                    break
                frame_info = json.loads(line)
                if is_valid_frame(frame_info, previous_frame):
                    (
                        ball_deltas,
                        home_players_deltas,
                        away_players_deltas,
                        time_deltas,
                    ) = extract_frame_info(
                        ball_deltas,
                        home_players_deltas,
                        away_players_deltas,
                        time_deltas,
                        frame_info,
                    )
                previous_frame = frame_info["frameNum"]

    if len(ball_deltas) < 2:
        return None, None, None, None

    ball_delta = np.median(compute_deltas(np.array(ball_deltas)), axis=0)
    home_players_delta: Dict[str, NDArray[np.float64]] = {}
    for jersey_num, player_frames in list(home_players_deltas.items()):
        home_players_delta[jersey_num] = np.median(
            compute_deltas(np.array(player_frames)), axis=0
        )
    away_players_delta: Dict[str, NDArray[np.float64]] = {}
    for jersey_num, player_frames in list(away_players_deltas.items()):
        away_players_delta[jersey_num] = np.median(
            compute_deltas(np.array(player_frames)), axis=0
        )
    time_delta = np.median(compute_deltas(np.array(time_deltas)))

    return time_delta, ball_delta, home_players_delta, away_players_delta


def compute_deltas(x: NDArray[np.float64]) -> NDArray[np.float64]:
    deltas = []
    for i in range(0, len(x) - 1, 2):
        deltas.append(x[i + 1] - x[i])
    return np.array(deltas)


def extract_frame_info(
    ball_delta: List[List[float]],
    home_players_deltas: Dict[str, List[List[float]]],
    away_players_deltas: Dict[str, List[List[float]]],
    time_delta: List[List[float]],
    frame_info: Dict[str, Any],
) -> Tuple[
    List[List[float]],
    Dict[str, List[List[float]]],
    Dict[str, List[List[float]]],
    List[List[float]],
]:
    ball_delta.append(
        [
            frame_info["ballsSmoothed"]["x"],
            frame_info["ballsSmoothed"]["y"],
            frame_info["ballsSmoothed"]["z"],
        ]
    )
    for player in frame_info["homePlayersSmoothed"]:
        home_list = home_players_deltas.get(player["jerseyNum"], [])
        home_list.append(
            [
                player["x"],
                player["y"],
            ]
        )
        home_players_deltas[player["jerseyNum"]] = home_list
    for player in frame_info["awayPlayersSmoothed"]:
        away_list = away_players_deltas.get(player["jerseyNum"], [])
        away_list.append(
            [
                player["x"],
                player["y"],
            ]
        )
        away_players_deltas[player["jerseyNum"]] = away_list

    time_delta.append([np.round(frame_info["videoTimeMs"] / 1000, 3)])

    return ball_delta, home_players_deltas, away_players_deltas, time_delta


def is_valid_frame(frame_info: Dict[str, Any], previous_frame: int) -> bool:
    return (
        frame_info["frameNum"] != previous_frame
        and frame_info["ballsSmoothed"] is not None
        and frame_info["ballsSmoothed"]["x"] is not None
        and frame_info["ballsSmoothed"]["y"] is not None
        and frame_info["homePlayersSmoothed"] is not None
        and frame_info["awayPlayersSmoothed"] is not None
    )
