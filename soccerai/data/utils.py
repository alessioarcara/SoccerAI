import json
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt


def offset_x(x: int) -> float:
    return (x or 0.0) + 52.5


def offset_y(y: int) -> float:
    return (y or 0.0) + 34.0


def compute_velocity(
    space_delta: npt.NDArray[np.float64], time_delta: np.floating
) -> Tuple[np.floating, np.floating]:
    velocity_x = space_delta[0] / time_delta
    velocity_y = space_delta[1] / time_delta
    if len(space_delta) > 2:
        velocity_z = space_delta[2] / time_delta
        velocity = np.linalg.norm([velocity_x, velocity_y, velocity_z])
    else:
        velocity = np.linalg.norm([velocity_x, velocity_y])
    direction = np.arctan2(velocity_y, velocity_x)
    direction = np.rad2deg(direction)
    return velocity, direction


def download_video_frame(
    frame_index, event_dict, output_dir="./frames"
) -> Optional[str]:
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{output_dir}/frame_{frame_index}.jpeg"

    if os.path.exists(output_filename):
        return frame_index, output_filename

    video_url = event_dict.get("videoUrl")
    if not video_url:
        return frame_index, None

    parts = video_url.split("/")
    if len(parts) < 7:
        return frame_index, None

    match_id = parts[5]
    try:
        video_seconds = float(parts[6])
    except Exception:
        video_seconds = 0.0

    playlist_url = f"https://d293djmf54wuo5.cloudfront.net/{match_id}/playlist.m3u8"

    ffmpeg_command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-ss",
        str(video_seconds),
        "-copyts",
        "-start_at_zero",
        "-i",
        playlist_url,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        output_filename,
        "-y",
    ]

    try:
        subprocess.run(
            ffmpeg_command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return frame_index, output_filename
    except subprocess.CalledProcessError:
        return frame_index, None


def create_event_byte_map(game_id: int) -> Dict[int, int]:
    event_byte_map: Dict[int, int] = {}
    pending_events: Dict[int, int] = {}
    tracking_file = (
        f"/home/soccerdata/FIFA_WorldCup_2022/Event Data/Tracking Data/{game_id}.jsonl"
    )
    with open(tracking_file, "r") as tracking_data:
        current_byte_pos = tracking_data.tell()
        while True:
            frame = tracking_data.readline()
            if not frame:
                break
            frame_info = json.loads(frame)
            frame_num = frame_info["frameNum"]
            current_event_id = frame_info["game_event_id"]
            if current_event_id not in event_byte_map:
                if frame_num in pending_events.values():
                    for pending_id, pending_frame in list(pending_events.items()):
                        if frame_num == pending_frame:
                            event_byte_map[pending_id] = current_byte_pos
                            del pending_events[pending_id]
                elif current_event_id is not None:
                    current_event_id = int(current_event_id)
                    if current_event_id not in pending_events:
                        end_frame = frame_info["game_event"]["end_frame"]
                        if frame_num != end_frame:
                            pending_events[current_event_id] = end_frame
                        else:
                            event_byte_map[current_event_id] = current_byte_pos
                    elif (
                        frame_info["possession_event_id"] is not None
                        and frame_info["ballsSmoothed"] is not None
                    ):
                        event_byte_map[current_event_id] = current_byte_pos
                        del pending_events[current_event_id]

            current_byte_pos = tracking_data.tell()
        return event_byte_map


def read_last_n_lines(
    filename: str, start_pos: int, max_lines: int, block_size: int = 4096
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

    return [line.decode(encoding="utf-8", errors="replace") for line in reversed(lines)]


def compute_deltas(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    deltas = []
    for i in range(0, len(values) - 1, 2):
        deltas.append(values[i + 1] - values[i])
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
