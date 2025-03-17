from typing import Tuple, Union, Optional
import os
import subprocess
import numpy as np


def offset_x_by_60(x: int):
    return (x or 0) + 60


def offset_y_by_40(y: int):
    return (y or 0) + 40


def compute_velocity(
    start_pos: Tuple[float],
    end_pos: Tuple[float],
    start_time: float,
    end_time: float,
    return_direction=False,
) -> Union[Tuple[float], float]:

    delta_t = end_time - start_time
    delta_x = end_pos[0] - start_pos[0]
    delta_y = end_pos[1] - start_pos[1]
    velocity_y = delta_y / delta_t
    velocity_x = delta_x / delta_t
    velocity = np.linalg.norm([velocity_x, velocity_y])

    if return_direction:
        direction = np.arctan2(velocity_y / velocity_x)
        return velocity, direction

    return velocity


def compute_direction():
    pass


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
