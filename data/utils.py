from typing import Tuple, Union, Optional, Dict, Any
import os
import subprocess
import numpy as np
import bz2
import json


def offset_x(x: int):
    return (x or 0) + 52.5


def offset_y(y: int):
    return (y or 0) + 34


def compute_velocity(
    start_pos: Tuple[float], end_pos: Tuple[float], start_time: float, end_time: float
) -> float:

    delta_t = end_time - start_time
    delta_x = end_pos[0] - start_pos[0]
    delta_y = end_pos[1] - start_pos[1]
    velocity_y = delta_y / delta_t
    velocity_x = delta_x / delta_t
    velocity = np.linalg.norm([velocity_x, velocity_y])
    return velocity


def compute_direction(start_pos: Tuple[float], end_pos: Tuple[float]) -> float:
    delta_x = end_pos[0] - start_pos[0]
    delta_y = end_pos[1] - start_pos[1]

    direction = np.arctan2(delta_y, delta_x)
    return direction


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


def create_event_byte_map(game_id: int) -> Dict[int, Any]:
    event_byte_map = {}
    tracking_file = f"/home/soccerdata/FIFA_WorldCup_2022/Tracking Data/{game_id}.jsonl"
    with open(tracking_file, "r") as tracking_data:
        event_id = -1
        end_frame = -1
        current_byte_pos = tracking_data.tell()
        while True:
            frame = tracking_data.readline()
            if not frame:
                break
            frame_info = json.loads(frame)
            if frame_info["game_event_id"] is not None and end_frame == -1:
                event_id = int(frame_info["game_event_id"])
                event_byte_map[event_id] = {"ball_pos": -1, "players_pos": -1}
                end_frame = frame_info["game_event"]["end_frame"]

            if frame_info["frameNum"] == end_frame:
                event_byte_map[event_id]["players_pos"] = current_byte_pos
            elif frame_info["frameNum"] > end_frame and end_frame != -1:
                event_byte_map[event_id]["ball_pos"] = current_byte_pos
                end_frame = -1

                if frame_info["game_event_id"] is not None and end_frame == -1:
                    event_id = int(frame_info["game_event_id"])
                    event_byte_map[event_id] = {"ball_pos": -1, "players_pos": -1}
                    end_frame = frame_info["game_event"]["end_frame"]
                if frame_info["frameNum"] == end_frame:
                    event_byte_map[event_id]["players_pos"] = current_byte_pos

            current_byte_pos = tracking_data.tell()
        return event_byte_map


def decompress_tracking_file(filepath: str) -> None:
    command = ["bunzip2", filepath]
    subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def compress_tracking_file(filepath: str) -> None:
    command = ["bzip2", filepath]
    subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
