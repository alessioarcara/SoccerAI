from mplsoccer import Pitch
import matplotlib.pyplot as plt
from typing import List, Tuple, Any
import polars as pl
import ipywidgets as widgets
from IPython.display import Image, display, clear_output
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib as mpl
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.animation import FuncAnimation
from utils import download_video_frame
import numpy as np

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


def visualize_frame(
    row_index: int,
    event_df: pl.DataFrame,
    players_df: pl.DataFrame,
    ball_trajectory: List[Tuple[float]],
) -> None:
    frame_df = pl.from_dict(event_df.row(row_index, named=True))
    join_table = frame_df.join(players_df, on="gameEventId")

    pitch = Pitch(
        pitch_type="custom",
        pitch_width=68,
        pitch_length=105,
        pitch_color="grass",
        line_color="#c7d5cc",
        stripe=True,
    )
    _, ax = pitch.draw(figsize=(10, 8))

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
        color="red",
        alpha=0.9,
        s=300,
        jersey_numbers=home_coords[:, 2],
        label_color="white",
        zorder=3,
    )
    plot_players_with_numbers(
        ax=ax,
        x_coords=away_coords[:, 0],
        y_coords=away_coords[:, 1],
        color="blue",
        s=300,
        alpha=0.5,
        jersey_numbers=away_coords[:, 2],
        label_color="white",
        zorder=4,
    )

    ball_df = join_table.filter(pl.col("team").is_null())
    ball_row = ball_df.row(0, named=True)
    ax.scatter(
        ball_row["x"], ball_row["y"], color="white", edgecolors="black", s=100, zorder=5
    )
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
    output_dir: str,
    show_video: bool = True,
) -> None:
    """ """
    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(frames) - 1,
        description="Frame:",
        continuous_update=False,
    )
    pitch_output = widgets.Output()
    video_output = widgets.Output()

    if show_video:
        event_dicts = event_df.to_dicts()
        video_files = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(
                    download_video_frame, frame_index, event_dicts[frame_index], output_dir
                ): frame_index
                for frame_index in frames
            }

            for future in as_completed(futures):
                result = future.result()
                if result:
                    frame_index, filename = result
                    video_files[frame_index] = filename

    frame_df = event_df.filter(pl.col("index").is_in(frames))
    join_table = frame_df.join(players_df, on="gameEventId")

    ball_coordinates = []
    for frame_idx in frames:
        ball_xco = join_table.filter(
            (pl.col("team").is_null()) & (pl.col("index") == frame_idx)
        ).row(0, named=True)["x"]
        ball_yco = join_table.filter(
            (pl.col("team").is_null()) & (pl.col("index") == frame_idx)
        ).row(0, named=True)["y"]
        ball_coordinates.append((ball_xco, ball_yco))

    def on_value_change(change):
        new_val = change["new"]
        frame_index = frames[new_val]

        ball_trajectory = ball_coordinates[: new_val + 1]

        with pitch_output:
            clear_output(wait=True)
            visualize_frame(frame_index, event_df, players_df, ball_trajectory)

        if show_video:
            with video_output:
                clear_output(wait=True)
                video_file = video_files.get(frame_index)
                if video_file:
                    display(Image(filename=video_file))
                else:
                    print("Frame not available.")

        else:
            with video_output:
                clear_output(wait=True)
                print("Video disabled")

    slider.observe(on_value_change, names="value")
    on_value_change({"new": 0})

    if show_video:
        ui = widgets.VBox([widgets.HBox([pitch_output, video_output]), slider])
    else:
        ui = widgets.VBox([pitch_output, slider])

    display(ui)


def animate_pitch(
    frames: List[int],
    event_df: pl.DataFrame,
    players_df: pl.DataFrame,
    metadata_df: pl.DataFrame,
    interval=250,
):
    # Create the pitch with custom settings
    pitch = Pitch(
        pitch_type="custom",
        pitch_width=68,
        pitch_length=105,
        pitch_color="grass",
        line_color="#c7d5cc",
        stripe=True,
    )
    fig, ax = pitch.draw(figsize=(10, 8))

    # Load the ball image (ensure you have a PNG with transparency)
    ball_img = mpimg.imread("soccer_ball_transparent.png")
    ball_image = OffsetImage(ball_img, zoom=0.05)

    # Create the initial AnnotationBbox for the ball and store it in a mutable container (a list)
    ball_container = [
        AnnotationBbox(
            ball_image,
            (52.5, 52.5),  # Initial position (will be updated)
            xycoords="data",  # Use data coordinates from the pitch
            frameon=False,
            zorder=5,
        )
    ]
    ax.add_artist(ball_container[0])

    # Extract match metadata
    match_id = event_df[frames[0]]["gameId"].item()
    metadata_game = metadata_df.filter(pl.col("gameId") == str(match_id)).row(
        0, named=True
    )
    home_team_name = metadata_game["homeTeamName"]
    home_team_color = metadata_game["homeTeamColor"]
    away_team_name = metadata_game["awayTeamName"]
    away_team_color = metadata_game["awayTeamColor"]

    # Create scatter plots for home and away players
    home_scatter = ax.scatter([], [], color=home_team_color, s=300, alpha=0.9, zorder=3)
    away_scatter = ax.scatter([], [], color=away_team_color, s=300, alpha=0.5, zorder=4)
    (traj_line,) = ax.plot([], [], color="yellow", alpha=0.7, lw=2, zorder=4)
    (last_line,) = ax.plot([], [], color="red", alpha=0.7, lw=2, zorder=4)

    traj_x, traj_y = [], []
    jersey_texts = []

    # Function to choose a contrasting text color based on the team color
    def get_contrasting_text_color(team_color):
        color = team_color.strip().lower()
        # Check for common white representations
        if color in ["#ffffff", "white"]:
            return "black"
        else:
            return "white"

    home_text_color = get_contrasting_text_color(home_team_color)
    away_text_color = get_contrasting_text_color(away_team_color)
    # Add team names on the pitch
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

    # Add time text on the pitch
    time_text = ax.text(
        0.5,
        1.02,
        "",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color="white",
        ha="center",
        va="bottom",
        bbox=dict(facecolor="black", alpha=0.75, boxstyle="round,pad=0.5"),
    )

    # Define a constant offset for the ball (adjust these values as needed)
    ball_offset_x = 1  # offset in x-direction
    ball_offset_y = 1  # offset in y-direction
    last_index = frames[-1]

    def init():
        # Initialize player scatter plots and trajectory line
        home_scatter.set_offsets(np.empty((0, 2)))
        away_scatter.set_offsets(np.empty((0, 2)))
        traj_line.set_data([], [])
        for text in jersey_texts:
            text.remove()
        jersey_texts.clear()
        # Return the artists to be updated
        return home_scatter, away_scatter, ball_container[0], traj_line, time_text

    def update(i):
        frame_idx = frames[i]
        frame_df = event_df.filter(pl.col("index") == frame_idx)
        join_table = frame_df.join(players_df, on="gameEventId")

        home = join_table.filter(pl.col("team") == "home")
        away = join_table.filter(pl.col("team") == "away")
        ball = join_table.filter(pl.col("team").is_null())

        home_coords = home.select(["x", "y"]).to_numpy()
        away_coords = away.select(["x", "y"]).to_numpy()

        home_scatter.set_offsets(home_coords if home.height else np.empty((0, 2)))
        away_scatter.set_offsets(away_coords if away.height else np.empty((0, 2)))

        # Get the ball coordinates from data
        bx, by = ball.row(0, named=True)["x"], ball.row(0, named=True)["y"]
        # Apply the offset so the ball doesn't overlap with the player
        new_bx = bx + ball_offset_x
        new_by = by + ball_offset_y

        # Remove the previous AnnotationBbox and create a new one with updated position
        ball_container[0].remove()
        new_ball_ab = AnnotationBbox(
            ball_image, (new_bx, new_by), xycoords="data", frameon=False, zorder=5
        )
        ax.add_artist(new_ball_ab)
        ball_container[0] = new_ball_ab

        traj_x.append(bx)
        traj_y.append(by)
        traj_line.set_data(traj_x, traj_y)

        if frame_idx != last_index:
            traj_line.set_data(traj_x, traj_y)
        else:
            traj_line.set_data(traj_x[:-1], traj_y[:-1])
            last_line.set_data(traj_x[-2:], traj_y[-2:])

        for text in jersey_texts:
            text.remove()
        jersey_texts.clear()

        # Add jersey numbers for home players
        for player in home.iter_rows(named=True):
            text = ax.text(
                player["x"],
                player["y"],
                str(player["jerseyNum"]),
                ha="center",
                va="center",
                color=home_text_color,
                fontsize=10,
                fontweight="bold",
                zorder=4,
            )
            jersey_texts.append(text)

        # Add jersey numbers for away players
        for player in away.iter_rows(named=True):
            text = ax.text(
                player["x"],
                player["y"],
                str(player["jerseyNum"]),
                ha="center",
                va="center",
                color=away_text_color,
                fontsize=10,
                fontweight="bold",
                zorder=5,
            )
            jersey_texts.append(text)

        current_time = frame_df.row(0, named=True)["startTime"]
        minutes = int(current_time // 60)
        seconds = int(current_time % 60)
        time_text.set_text(f"Time: {minutes:02d}:{seconds:02d}")

        return (
            home_scatter,
            away_scatter,
            ball_container[0],
            traj_line,
            *jersey_texts,
            time_text,
        )

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        init_func=init,
        interval=interval,
        blit=False,  # Disable blitting to force full redraw
        repeat=True,
    )

    plt.close(fig)
    return anim


from IPython.display import HTML, clear_output, display
import ipywidgets as widgets


def manual_labeling(
    chain_range,
    animate_chain,
    pos_chains,
    event_df,
    players_df,
    metadata_df,
    interval=250,
):
    # Dictionary to store labels for each chain
    labels = {}
    current_chain = chain_range[0]

    # Create buttons for selecting or discarding a chain
    select_button = widgets.Button(
        description="Select Key Pass", button_style="success"
    )
    discard_button = widgets.Button(description="Discard Chain", button_style="danger")
    button_box = widgets.HBox([select_button, discard_button])
    output_area = widgets.Output()

    def next_chain():
        nonlocal current_chain
        current_chain += 1
        with output_area:
            clear_output()
            if current_chain >= chain_range[1]:
                print("Labeling complete!")
                print("Labels:", labels)
            else:
                anim = animate_chain(
                    current_chain,
                    pos_chains,
                    event_df,
                    players_df,
                    metadata_df,
                    interval=interval,
                )
                display(HTML(anim.to_jshtml()))

    def on_select_button_clicked(b):
        labels[current_chain] = "key_pass"
        next_chain()

    def on_discard_button_clicked(b):
        labels[current_chain] = "discarded"
        next_chain()

    select_button.on_click(on_select_button_clicked)
    discard_button.on_click(on_discard_button_clicked)

    with output_area:
        clear_output()
        anim = animate_chain(
            current_chain,
            pos_chains,
            event_df,
            players_df,
            metadata_df,
            interval=interval,
        )
        display(HTML(anim.to_jshtml()))

    display(button_box, output_area)
    return labels


def animate_chain(
    chain_id, pos_chains, event_df, players_df, metadata_df, interval=250
):
    # Extract frames for the given chain and create the animation
    frames_for_chain = pos_chains[chain_id]
    anim = animate_pitch(
        frames_for_chain, event_df, players_df, metadata_df, interval=interval
    )
    return anim
