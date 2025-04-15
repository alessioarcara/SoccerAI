from typing import Any, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from IPython.display import Image, clear_output, display
from ipywidgets import widgets
from mplsoccer import Pitch
from numpy.typing import NDArray

from soccerai.data import config
from soccerai.data.utils import download_video_frames

mpl.rcParams["animation.embed_limit"] = 50


def plot_players_with_numbers(
    ax: plt.Axes,
    x_coords: NDArray[np.float64],
    y_coords: NDArray[np.float64],
    jersey_numbers: NDArray[np.str_],
    color: str,
    alpha: float,
    s: int,
    zorder: int,
    label_color: str = "white",
) -> List[Any]:
    scat = ax.scatter(x_coords, y_coords, color=color, alpha=alpha, s=s, zorder=zorder)
    texts = []

    for x, y, number in zip(x_coords, y_coords, jersey_numbers):
        text = ax.text(
            x,
            y,
            str(number),
            ha="center",
            va="center",
            color=label_color,
            fontsize=10,
            fontweight="bold",
            zorder=zorder + 1,
        )
        texts.append(text)

    return [scat, *texts]


def plot_pitch_info(
    ax: plt.Axes,
    home_team_name: str,
    away_team_name: str,
    home_team_color: str,
    away_team_color: str,
    home_text_color: str,
    away_text_color: str,
    time_str: str = "",
):
    ax.text(
        0.01,
        1.03,
        home_team_name,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        color=home_text_color,
        ha="left",
        va="center",
        bbox=dict(facecolor=home_team_color, alpha=0.8, boxstyle="round,pad=0.3"),
    )

    ax.text(
        0.99,
        1.03,
        away_team_name,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        color=away_text_color,
        ha="right",
        va="center",
        bbox=dict(facecolor=away_team_color, alpha=0.8, boxstyle="round,pad=0.3"),
    )

    ax.text(
        0.5,
        1.02,
        time_str,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color="white",
        ha="center",
        va="bottom",
        bbox=dict(facecolor="black", alpha=0.75, boxstyle="round,pad=0.5"),
    )


def get_contrasting_text_color(team_color: str) -> str:
    color = team_color.strip().lower()
    return "black" if color in ["#ffffff", "white"] else "white"


def visualize_frame(
    frame_index: int,
    event_df: pl.DataFrame,
    frames_df: pl.DataFrame,
    metadata_df: pl.DataFrame,
    ball_trajectory: List[Tuple[float]],
) -> None:
    frame_df = frames_df.filter(pl.col("index") == frame_index)
    match_id = event_df[frame_index]["gameId"].item()
    metadata_game = metadata_df.filter(pl.col("gameId") == str(match_id)).row(
        0, named=True
    )

    home_team_color = metadata_game["homeTeamColor"]
    away_team_color = metadata_game["awayTeamColor"]
    home_text_color = get_contrasting_text_color(home_team_color)
    away_text_color = get_contrasting_text_color(away_team_color)

    pitch = Pitch(**config.PITCH_SETTINGS)
    _, ax = pitch.draw(figsize=config.PITCH_FIGSIZE)

    home_coords = (
        frame_df.filter(pl.col("team") == "home")
        .select(pl.col("x"), pl.col("y"), pl.col("jerseyNum"))
        .to_numpy()
    )

    away_coords = (
        frame_df.filter(pl.col("team") == "away")
        .select(pl.col("x"), pl.col("y"), pl.col("jerseyNum"))
        .to_numpy()
    )

    plot_players_with_numbers(
        ax=ax,
        x_coords=home_coords[:, 0],
        y_coords=home_coords[:, 1],
        jersey_numbers=home_coords[:, 2],
        color=home_team_color,
        alpha=0.9,
        s=300,
        label_color=home_text_color,
        zorder=3,
    )

    plot_players_with_numbers(
        ax,
        x_coords=away_coords[:, 0],
        y_coords=away_coords[:, 1],
        jersey_numbers=away_coords[:, 2],
        color=away_team_color,
        alpha=0.5,
        s=300,
        label_color=away_text_color,
        zorder=4,
    )

    # Add pitch info (team names and time)
    plot_pitch_info(
        ax,
        metadata_game["homeTeamName"],
        metadata_game["awayTeamName"],
        home_team_color,
        away_team_color,
        home_text_color,
        away_text_color,
        event_df.row(frame_index, named=True)["frameTime"],
    )

    # Draw the ball using the ball sprite
    # ball_df = frame_df.filter(pl.col("team").is_null())
    # ball_row = ball_df.row(0, named=True)
    # ball_img = mpimg.imread(config.BALL_IMAGE_PATH)
    # ball_image = OffsetImage(ball_img, zoom=config.BALL_ZOOM)
    # ball_ab = AnnotationBbox(
    #    ball_image,
    #    (ball_row["x"] + config.BALL_OFFSET_X, ball_row["y"] + config.BALL_OFFSET_Y),
    #    xycoords="data",
    #    frameon=False,
    #    zorder=5,
    # )
    # ax.add_artist(ball_ab)

    # Draw ball trajectory
    if len(ball_trajectory) > 1:
        xs, ys = zip(*ball_trajectory)
        ax.plot(xs, ys, color="yellow", linewidth=2, alpha=0.7, zorder=4)
        ax.annotate(
            "",
            xy=(xs[-1], ys[-1]),
            xytext=(xs[-2], ys[-2]),
            arrowprops=dict(arrowstyle="->", color="yellow", lw=5),
            zorder=5,
        )

    plt.tight_layout()
    plt.show()


def shot_frames_navigator(
    frames: List[int],
    event_df: pl.DataFrame,
    players_df: pl.DataFrame,
    metadata_df: pl.DataFrame,
    output_dir: str,
    show_video: bool = True,
    interval: str = "1000",
):
    play = widgets.Play(
        value=0,
        min=0,
        max=len(frames) - 1,
        step=1,
        interval=int(interval),
        description="Play",
        disabled=False,
    )

    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(frames) - 1,
        description="Frame:",
        continuous_update=False,
        layout=widgets.Layout(width="300px"),
    )
    widgets.jslink((play, "value"), (slider, "value"))

    pitch_output = widgets.Output(layout=widgets.Layout(**config.PITCH_OUTPUT_LAYOUT))
    video_output = widgets.Output(layout=widgets.Layout(**config.VIDEO_OUTPUT_LAYOUT))

    # Pre-download video files if show_video is True
    video_files = {}
    if show_video:
        video_files = download_video_frames(frames, event_df, output_dir)

    # Collect ball coordinates to draw trajectory
    frames_df = event_df.filter(pl.col("index").is_in(frames)).join(
        players_df, on=["gameEventId", "possessionEventId"]
    )

    ball_coordinates = []
    for f_idx in frames:
        ball_data = frames_df.filter(
            (pl.col("team").is_null()) & (pl.col("index") == f_idx)
        ).row(0, named=True)
        ball_coordinates.append((ball_data["x"], ball_data["y"]))

    # Updates pitch and video outputs.
    def on_slider_value_change(change):
        new_val = change["new"]
        frame_index = frames[new_val]
        ball_trajectory = ball_coordinates[: new_val + 1]

        with pitch_output:
            clear_output(wait=True)
            visualize_frame(
                frame_index, event_df, frames_df, metadata_df, ball_trajectory
            )

        if show_video:
            with video_output:
                clear_output(wait=True)
                video_file = video_files.get(frame_index)
                if video_file:
                    display(Image(filename=video_file, width=400))
                else:
                    print("Frame not available.")
        else:
            with video_output:
                clear_output(wait=True)
                print("Video disabled.")

    slider.observe(on_slider_value_change, names="value")
    on_slider_value_change({"new": 0})

    outputs_container = widgets.HBox(
        [pitch_output, video_output], layout=widgets.Layout(**config.TOP_BOX_LAYOUT)
    )

    controls_container = widgets.HBox(
        [play, slider], layout=widgets.Layout(**config.CONTROLS_BOX_LAYOUT)
    )

    ui = widgets.VBox(
        [outputs_container, controls_container],
        layout=widgets.Layout(width="100%", align_items="center"),
    )
    display(ui)
