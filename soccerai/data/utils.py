import json
import os
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from numpy.typing import NDArray


def offset_x(x: int) -> float:
    return (x or 0.0) + 52.5


def offset_y(y: int) -> float:
    return (y or 0.0) + 34.0


def compute_velocity(
    positions_delta: NDArray[np.float64], time_elapsed: np.floating
) -> Tuple[np.floating, np.floating]:
    """
    space_delta: Array of position differences [x, y] or [x, y, z]
    """
    velocity_vector = positions_delta / time_elapsed
    velocity = np.linalg.norm(velocity_vector)
    direction = np.rad2deg(np.arctan2(*velocity_vector))
    return velocity, direction


def download_video_frame(
    frame_index: int, event_dict: Dict[str, Any], output_dir: str
) -> Tuple[int, Optional[str]]:
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


def download_video_frames(
    frames: List[int],
    event_df: pl.DataFrame,
    output_dir: str = "./frames",
    max_workers: int = 8,
) -> Dict[int, str]:
    video_files = {}
    event_dicts = event_df.to_dicts()

    os.makedirs(output_dir, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: Dict[Future, int] = {
            executor.submit(
                download_video_frame, f_idx, event_dicts[f_idx], output_dir
            ): f_idx
            for f_idx in frames
        }
        for future in as_completed(futures):
            res: Tuple[int, Optional[str]] = future.result()
            frame_idx, filename = res
            if filename is not None:
                video_files[frame_idx] = filename
    return video_files


def save_accepted_chains(
    accepted_chains: List[List[int]], dst_dir: str, are_positive: bool
) -> None:
    output_file = os.path.join(
        dst_dir, f"accepted_{'pos' if are_positive else 'neg'}_chains.json"
    )
    all_accepted = []

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            all_accepted = json.load(f)

    all_accepted.extend(accepted_chains)

    with open(output_file, "w") as f:
        json.dump(all_accepted, f)


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


def read_lines_backwards_from_offset(
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
