import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
from numpy.typing import NDArray


@dataclass
class FrameData:
    ball_positions: List[List[float]]
    home_players_positions: Dict[str, List[List[float]]]
    away_players_positions: Dict[str, List[List[float]]]
    timestamps: List[List[float]]

    def has_sufficient_data(self) -> bool:
        return len(self.ball_positions) >= 2

    def reset(self) -> None:
        self.ball_positions = []
        self.home_players_positions = {}
        self.away_players_positions = {}
        self.timestamps = []


class PlayerVelocityEnricher:
    """
    Class to process soccer tracking data and calculate player velocities
    """

    def __init__(self, tracking_dir_path: str):
        self.tracking_dir_path = tracking_dir_path

    def add_velocity_per_player(self, players_df: pl.DataFrame) -> pl.DataFrame:
        """
        Add velocity and direction columns to the players dataframe.
        """
        gameIds = np.unique(players_df.select(pl.col("gameId")).to_numpy())
        gameEventIds = np.unique(players_df.select(pl.col("gameEventId")).to_numpy())

        velocities: List[Optional[np.floating]] = []
        directions: List[Optional[np.floating]] = []

        for gameId in gameIds:
            tracking_file = f"{self.tracking_dir_path}/{gameId}.jsonl"
            event_byte_map = self._create_event_byte_map(tracking_file)

            for gameEventId in gameEventIds:
                byte_pos = event_byte_map.get(gameEventId)
                players_per_event = players_df.filter(
                    pl.col("gameEventId") == gameEventId
                )

                if byte_pos is None:
                    velocities.extend([None] * players_per_event.height)
                    directions.extend([None] * players_per_event.height)
                    continue

                time_elapsed, ball_delta, home_players_deltas, away_players_deltas = (
                    self._extract_tracking_data(tracking_file, byte_pos)
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
                    velocity, direction = self._compute_velocity(delta, time_elapsed)
                    velocities.append(velocity)
                    directions.append(direction)

        return players_df.with_columns(
            pl.Series("velocity", velocities),
            pl.Series("direction", directions),
        )

    def _compute_velocity(
        self, positions_delta: NDArray[np.float64], time_elapsed: np.floating
    ) -> Tuple[np.floating, np.floating]:
        velocity_vector = positions_delta / time_elapsed
        velocity = np.linalg.norm(velocity_vector)
        direction = np.rad2deg(np.arctan2(velocity_vector[1], velocity_vector[0]))
        return velocity, direction

    def _create_event_byte_map(self, tracking_file: str) -> Dict[int, int]:
        """
        Create a mapping of game event IDs to byte positions in the tracking file.;w
        """
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

    def _extract_tracking_data(
        self, tracking_file: str, byte_pos: int
    ) -> Union[
        Tuple[
            np.floating,
            NDArray[np.float64],
            Dict[str, NDArray[np.float64]],
            Dict[str, NDArray[np.float64]],
        ],
        Tuple[None, ...],
    ]:
        """
        Extract tracking data from a file at a specific byte position
        """
        frame_data = FrameData([], {}, {}, [])

        self._process_frames(
            self._read_frames_backward(tracking_file, byte_pos), frame_data
        )

        if not frame_data.has_sufficient_data():
            frame_data.reset()
            self._process_frames(
                self._read_frames_forward(tracking_file, byte_pos),
                frame_data,
                check_game_event=False,
            )

        if not frame_data.has_sufficient_data():
            return (None,) * 4

        return self._compute_deltas_avoiding_outliers(frame_data)

    def _process_frames(
        self, frames: List[str], frame_data: FrameData, check_game_event: bool = True
    ):
        previous_frame = -1

        if not frames:
            return

        if check_game_event and len(frames) >= 2:
            last_frame_info = json.loads(frames[-1])
            if last_frame_info["game_event"]["game_event_type"] in {
                "SUB",
                "SECONDKICKOFF",
                "THIRDKICKOFF",
                "FOURTHKICKOFF",
            }:
                return

        for frame in frames:
            frame_info = json.loads(frame)
            if self._is_valid_frame(frame_info, previous_frame):
                self._extract_frame_info(frame_data, frame_info)
            previous_frame = frame_info["frameNum"]

    def _is_valid_frame(self, frame_info: Dict[str, Any], previous_frame: int) -> bool:
        return (
            frame_info["frameNum"] != previous_frame
            and frame_info["ballsSmoothed"] is not None
            and frame_info["ballsSmoothed"]["x"] is not None
            and frame_info["ballsSmoothed"]["y"] is not None
            and frame_info["homePlayersSmoothed"] is not None
            and frame_info["awayPlayersSmoothed"] is not None
        )

    def _extract_players_data(
        self, players: List[Dict[str, Any]], players_dict: Dict[str, List[List[float]]]
    ) -> None:
        for player in players:
            jersey_num = player["jerseyNum"]
            if jersey_num not in players_dict:
                players_dict[jersey_num] = []
            players_dict[jersey_num].append([player["x"], player["y"]])

    def _extract_frame_info(
        self,
        frame_data: FrameData,
        frame_info: Dict[str, Any],
    ) -> None:
        ball = frame_info["ballsSmoothed"]
        frame_data.ball_positions.append([ball["x"], ball["y"], ball["z"]])

        self._extract_players_data(
            frame_info["homePlayersSmoothed"], frame_data.home_players_positions
        )
        self._extract_players_data(
            frame_info["awayPlayersSmoothed"], frame_data.away_players_positions
        )

        frame_data.timestamps.append([np.round(frame_info["videoTimeMs"] / 1000, 3)])

    def _compute_deltas_avoiding_outliers(
        self, frame_data: FrameData
    ) -> Tuple[
        np.floating,
        NDArray[np.float64],
        Dict[str, NDArray[np.float64]],
        Dict[str, NDArray[np.float64]],
    ]:
        def compute_pairwise_differences(x):
            return np.array([x[i + 1] - x[i] for i in range(0, len(x) - 1, 2)])

        ball_delta = np.median(
            compute_pairwise_differences(np.array(frame_data.ball_positions)), axis=0
        )

        home_players_delta = {
            jersey_num: np.median(
                compute_pairwise_differences(np.array(positions)), axis=0
            )
            for jersey_num, positions in frame_data.home_players_positions.items()
        }

        away_players_delta = {
            jersey_num: np.median(
                compute_pairwise_differences(np.array(positions)), axis=0
            )
            for jersey_num, positions in frame_data.away_players_positions.items()
        }

        time_delta = np.median(
            compute_pairwise_differences(np.array(frame_data.timestamps))
        )
        return time_delta, ball_delta, home_players_delta, away_players_delta

    def _read_frames_backward(
        self, filename: str, start_pos: int, max_lines: int = 60, block_size: int = 4096
    ) -> List[str]:
        lines: List[bytes] = []
        with open(filename, "rb") as f:
            f.seek(start_pos)
            f.readline()
            buffer = b""
            position = f.tell()

            while position > 0 and len(lines) < max_lines:
                read_size = min(block_size, position)
                position -= read_size
                f.seek(position)
                data = f.read(read_size)
                buffer = data + buffer
                lines_found = buffer.split(b"\n")

                buffer = lines_found[0]
                lines_from_chunk = lines_found[1:]

                for line in reversed(lines_from_chunk):
                    if len(lines) >= max_lines:
                        break
                    if line != b"":
                        lines.append(line)

            if buffer and len(lines) < max_lines:
                lines.append(buffer)

        return [
            line.decode(encoding="utf-8", errors="replace") for line in reversed(lines)
        ]

    def _read_frames_forward(
        self, filename: str, start_pos: int, max_lines: int = 4
    ) -> List[str]:
        frames = []
        with open(filename, "r") as f:
            f.seek(start_pos)
            for _ in range(max_lines):
                line = f.readline()
                if not line:
                    break
                frames.append(line)
        return frames
