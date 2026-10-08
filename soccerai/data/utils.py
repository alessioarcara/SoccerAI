import json
import os
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any

import polars as pl


def offset_x(x: float) -> float:
    return (x or 0.0) + 52.5


def home_attacks_right(
    period: int,
    home_team_start_left: bool,
    home_team_start_left_extra_time: bool | None = None,
) -> bool:
    """
    Whether the home team attacks towards x = pitch length in the given period.

    PFF metadata gives the side the home team *starts* on (`homeTeamStartLeft`,
    and `homeTeamStartLeftExtraTime` for the extra-time periods 3 and 4).
    Teams swap ends between the two periods of each pair, so the home team
    attacks to the right in periods 1/3 when it starts on the left, and in
    periods 2/4 when it starts on the right.
    """
    if period in (3, 4) and home_team_start_left_extra_time is not None:
        start_left = home_team_start_left_extra_time
    else:
        start_left = home_team_start_left
    first_period_of_pair = period in (1, 3)
    return bool(start_left) == first_period_of_pair


def offset_y(y: float) -> float:
    return (y or 0.0) + 34.0


def download_video_frame(
    frame_index: int, event_dict: dict[str, Any], output_dir: str
) -> tuple[int, str | None]:
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
    frames: list[int],
    event_df: pl.DataFrame,
    output_dir: str = "./frames",
    max_workers: int = 8,
) -> dict[int, str]:
    video_files = {}
    event_dicts = event_df.to_dicts()

    os.makedirs(output_dir, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[Future, int] = {
            executor.submit(
                download_video_frame, f_idx, event_dicts[f_idx], output_dir
            ): f_idx
            for f_idx in frames
        }
        for future in as_completed(futures):
            res: tuple[int, str | None] = future.result()
            frame_idx, filename = res
            if filename is not None:
                video_files[frame_idx] = filename
    return video_files


def save_accepted_chains(
    accepted_chains: list[list[int]], dst_dir: str, are_positive: bool
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
