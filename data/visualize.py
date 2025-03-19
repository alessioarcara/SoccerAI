import matplotlib.pyplot as plt
import polars as pl
from mplsoccer import Pitch
from typing import List, Tuple, Any
from ipywidgets import widgets, Button, Layout
from IPython.display import Image, display, clear_output
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib as mpl
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import config
from utils import download_video_frame

mpl.rcParams["animation.embed_limit"] = 50


def plot_players_with_numbers(
    ax: plt.Axes,
    x_coords: List[float],
    y_coords: List[float],
    jersey_numbers: List[int],
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


def visualize_frame(
    frame_index: int,
    event_df: pl.DataFrame,
    players_df: pl.DataFrame,
    metadata_df: pl.DataFrame,
    ball_trajectory: List[Tuple[float]],
) -> None:
    frame_df = pl.from_dict(event_df.row(frame_index, named=True))
    join_table = frame_df.join(players_df, on="gameEventId")

    match_id = event_df[frame_index]["gameId"].item()
    metadata_game = metadata_df.filter(pl.col("gameId") == str(match_id)).row(
        0, named=True
    )

    def get_contrasting_text_color(team_color: str) -> str:
        color = team_color.strip().lower()
        return "black" if color in ["#ffffff", "white"] else "white"

    home_team_color = metadata_game["homeTeamColor"]
    away_team_color = metadata_game["awayTeamColor"]
    home_text_color = get_contrasting_text_color(home_team_color)
    away_text_color = get_contrasting_text_color(away_team_color)

    pitch = Pitch(**config.PITCH_SETTINGS)
    _, ax = pitch.draw(figsize=config.PITCH_FIGSIZE)

    home_coords = (
        join_table.filter(pl.col("team") == "home")
        .select(pl.col("x"), pl.col("y"), pl.col("jerseyNum"))
        .to_numpy()
    )

    away_coords = (
        join_table.filter(pl.col("team") == "away")
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
        ax=ax,
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
    ball_df = join_table.filter(pl.col("team").is_null())
    ball_row = ball_df.row(0, named=True)
    ball_img = mpimg.imread(config.BALL_IMAGE_PATH)
    ball_image = OffsetImage(ball_img, zoom=config.BALL_ZOOM)
    ball_ab = AnnotationBbox(
        ball_image,
        (ball_row["x"] + config.BALL_OFFSET_X, ball_row["y"] + config.BALL_OFFSET_Y),
        xycoords="data",
        frameon=False,
        zorder=5,
    )
    ax.add_artist(ball_ab)

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
        event_dicts = event_df.to_dicts()
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(
                    download_video_frame, f_idx, event_dicts[f_idx], output_dir
                ): f_idx
                for f_idx in frames
            }
            for future in as_completed(futures):
                res = future.result()
                if res:
                    frame_idx, filename = res
                    video_files[frame_idx] = filename

    # Collect ball coordinates to draw trajectory
    frame_df = event_df.filter(pl.col("index").is_in(frames))
    join_table = frame_df.join(players_df, on="gameEventId")
    ball_coordinates = []
    for f_idx in frames:
        ball_x = join_table.filter(
            (pl.col("team").is_null()) & (pl.col("index") == f_idx)
        ).row(0, named=True)["x"]
        ball_y = join_table.filter(
            (pl.col("team").is_null()) & (pl.col("index") == f_idx)
        ).row(0, named=True)["y"]
        ball_coordinates.append((ball_x, ball_y))

    # Updates pitch and video outputs.
    def on_slider_value_change(change):
        new_val = change["new"]
        frame_index = frames[new_val]
        ball_traj = ball_coordinates[: new_val + 1]

        with pitch_output:
            clear_output(wait=True)
            visualize_frame(frame_index, event_df, players_df, metadata_df, ball_traj)

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


def label_shot_chains(
    chains: List[List[int]],
    chains_range: Tuple[int, int],
    event_df: pl.DataFrame,
    players_df: pl.DataFrame,
    metadata_df: pl.DataFrame,
    output_dir: str,
    show_video: bool = True,
    interval: int = 1000,
) -> List[List[int]]:
    accepted_chains = []
    current_chain_index = chains_range[0]
    selection_widget = None

    main_output = widgets.Output()
    extra_output = widgets.Output()

    accept_button = Button(
        description="Accept Chain",
        button_style="success",
        layout=Layout(**config.BUTTON_STYLE),
    )
    discard_button = Button(
        description="Discard Chain",
        button_style="danger",
        layout=Layout(**config.BUTTON_STYLE),
    )

    controls = widgets.HBox([accept_button, discard_button])

    def update_ui():
        with main_output:
            clear_output(wait=True)
            shot_frames_navigator(
                chains[current_chain_index],
                event_df,
                players_df,
                metadata_df,
                output_dir,
                show_video=show_video,
                interval=interval,
            )

    def update_selection_widget():
        nonlocal selection_widget
        with extra_output:
            clear_output(wait=True)
            current_chain = chains[current_chain_index]
            selection_widget = widgets.SelectMultiple(
                options=current_chain,
                value=current_chain,
                description="Keep frames:",
                disabled=False,
            )
            display(selection_widget)

    def next_chain():
        nonlocal current_chain_index
        current_chain_index += 1
        if current_chain_index < chains_range[1]:
            update_ui()
            update_selection_widget()
        else:
            with main_output:
                clear_output(wait=True)
                print("Labeling complete!")

    def on_accept(_):
        nonlocal selection_widget
        selected_frames = list(selection_widget.value)
        accepted_chains.append(selected_frames)
        next_chain()

    def on_discard(_):
        next_chain()

    accept_button.on_click(on_accept)
    discard_button.on_click(on_discard)

    update_selection_widget()
    update_ui()

    ui = widgets.VBox([main_output, extra_output, controls])
    display(ui)

    return accepted_chains
