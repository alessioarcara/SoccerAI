from typing import List

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mplsoccer import Pitch


def animate_pitch(
    frames: List[int],
    event_df: pl.DataFrame,
    players_df: pl.DataFrame,
    metadata_df: pl.DataFrame,
    interval=250,
):
    """
    Create an animation showing a sequence of frames on a soccer pitch.
    Shows home/away players, the ball, and the ball trajectory.

    Parameters:
        frames (List[int]): Frame indices to animate.
        event_df (pl.DataFrame): Dataframe containing event/frame data.
        players_df (pl.DataFrame): Dataframe with player info for joining.
        metadata_df (pl.DataFrame): Dataframe with match metadata (team colors/names, etc.).
        interval (int): Delay between frames in milliseconds.

    Returns:
        FuncAnimation: A matplotlib.animation.FuncAnimation object containing the animation.
    """
    # Configure the pitch
    pitch = Pitch(
        pitch_type="custom",
        pitch_width=68,
        pitch_length=105,
        pitch_color="grass",
        line_color="#c7d5cc",
        stripe=True,
    )
    fig, ax = pitch.draw(figsize=(10, 8))

    # Load the transparent ball image (PNG recommended)
    ball_img = mpimg.imread("soccer_ball_transparent.png")
    ball_image = OffsetImage(ball_img, zoom=0.05)

    # Create an AnnotationBbox for the ball and store it in a container so we can update it
    ball_container = [
        AnnotationBbox(
            ball_image,
            (52.5, 52.5),  # Temporary initial position
            xycoords="data",
            frameon=False,
            zorder=5,
        )
    ]
    ax.add_artist(ball_container[0])

    # Extract relevant metadata (teams, colors, etc.)
    match_id = event_df[frames[0]]["gameId"].item()
    metadata_game = metadata_df.filter(pl.col("gameId") == str(match_id)).row(
        0, named=True
    )
    home_team_name = metadata_game["homeTeamName"]
    home_team_color = metadata_game["homeTeamColor"]
    away_team_name = metadata_game["awayTeamName"]
    away_team_color = metadata_game["awayTeamColor"]

    # Create scatter plots for home and away teams; we'll update these in the animation
    home_scatter = ax.scatter([], [], color=home_team_color, s=300, alpha=0.9, zorder=3)
    away_scatter = ax.scatter([], [], color=away_team_color, s=300, alpha=0.5, zorder=4)

    # Prepare lines for the ball trajectory
    (traj_line,) = ax.plot([], [], color="yellow", alpha=0.7, lw=2, zorder=4)
    (last_line,) = ax.plot([], [], color="red", alpha=0.7, lw=2, zorder=4)

    traj_x, traj_y = [], []
    jersey_texts = []

    def get_contrasting_text_color(team_color: str) -> str:
        """
        Return a suitable text color ('black' or 'white') for visibility against team_color.
        """
        color = team_color.strip().lower()
        return "black" if color in ["#ffffff", "white"] else "white"

    home_text_color = get_contrasting_text_color(home_team_color)
    away_text_color = get_contrasting_text_color(away_team_color)

    # Add team labels at the top of the pitch
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

    # Time text label
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

    # Offsets for the ball so it doesn't overlap with players at the same (x,y)
    ball_offset_x = 1
    ball_offset_y = 1
    last_index = frames[-1]

    def init():
        """
        Initialization function for FuncAnimation. Sets up empty plots.
        """
        home_scatter.set_offsets(np.empty((0, 2)))
        away_scatter.set_offsets(np.empty((0, 2)))
        traj_line.set_data([], [])
        for text in jersey_texts:
            text.remove()
        jersey_texts.clear()
        return home_scatter, away_scatter, ball_container[0], traj_line, time_text

    def update(i):
        """
        Update function for each animation frame. Fetches data for the current index
        and updates player positions, ball location, trajectory, and displayed time.
        """
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

        # Retrieve ball coordinates, then apply offset
        bx, by = ball.row(0, named=True)["x"], ball.row(0, named=True)["y"]
        new_bx = bx + ball_offset_x
        new_by = by + ball_offset_y

        # Remove previous ball AnnotationBbox and add an updated one
        ball_container[0].remove()
        new_ball_ab = AnnotationBbox(
            ball_image, (new_bx, new_by), xycoords="data", frameon=False, zorder=5
        )
        ax.add_artist(new_ball_ab)
        ball_container[0] = new_ball_ab

        # Update trajectory
        traj_x.append(bx)
        traj_y.append(by)
        # If this isn't the last frame, display full trajectory; otherwise, highlight the last segment in red
        if frame_idx != last_index:
            traj_line.set_data(traj_x, traj_y)
        else:
            traj_line.set_data(traj_x[:-1], traj_y[:-1])
            last_line.set_data(traj_x[-2:], traj_y[-2:])

        # Remove and re-draw jersey number texts
        for text in jersey_texts:
            text.remove()
        jersey_texts.clear()

        # Label home players
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

        # Label away players
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

        # Update time text
        current_time = frame_df.row(0, named=True)["frameTime"]
        time_text.set_text(f"Time: {current_time}")

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
        blit=False,
        repeat=True,
    )
    plt.close(fig)
    return anim
