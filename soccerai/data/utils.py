import json
import os
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import polars as pl


def offset_x(x: int) -> float:
    return (x or 0.0) + 52.5


def offset_y(y: int) -> float:
    return (y or 0.0) + 34.0


# def compute_velocity(
#     start_pos: Tuple[float],
#     end_pos: Tuple[float],
#     start_time: float,
#     end_time: float,
#     return_direction=False,
# ) -> Union[Tuple[float], float]:
#     delta_t = end_time - start_time
#     delta_x = end_pos[0] - start_pos[0]
#     delta_y = end_pos[1] - start_pos[1]
#     velocity_y = delta_y / delta_t
#     velocity_x = delta_x / delta_t
#     velocity = np.linalg.norm([velocity_x, velocity_y])

#     if return_direction:
#         direction = np.arctan2(velocity_y / velocity_x)
#         return velocity, direction

#     return velocity


# def compute_direction():
#     pass


def download_video_frame(
    frame_index: int, event_dict: Dict[str, Any], output_dir: str
) -> Tuple[int, Optional[str]]:
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


def download_video_frames(
    frames: List[int],
    event_df: pl.DataFrame,
    output_dir: str = "./frames",
    max_workers: int = 8,
) -> Dict[int, str]:
    video_files = {}
    event_dicts = event_df.to_dicts()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: Dict[Future, int] = {
            executor.submit(
                download_video_frame, f_idx, event_dicts[f_idx], output_dir
            ): f_idx
            for f_idx in frames
        }
        for future in as_completed(futures):
            res: Optional[Tuple[int, str]] = future.result()
            if res:
                frame_idx, filename = res
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
