import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from mplsoccer import Pitch
from typing import List, Tuple, Any
from ipywidgets import widgets, Button, Layout
from IPython.display import Image, display, clear_output
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib as mpl
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.animation import FuncAnimation
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
    """
    Plot players (represented as scatter points) on a pitch and display their jersey numbers.

    Parameters:
        ax (plt.Axes): The matplotlib Axes object where the players will be plotted.
        x_coords (List[float]): List of X coordinates for each player.
        y_coords (List[float]): List of Y coordinates for each player.
        jersey_numbers (List[int]): Jersey numbers for the players in the same order as x_coords and y_coords.
        color (str): Color used to fill the scatter points.
        alpha (float): Alpha (opacity) for the scatter points.
        s (int): Size of the scatter points.
        zorder (int): Drawing order for the points (higher zorder means drawn on top).
        label_color (str): Color for the jersey number text.

    Returns:
        List[Any]: A list containing the scatter plot object and the text objects for the jersey numbers.
    """
    # Create a scatter plot for all players
    scat = ax.scatter(x_coords, y_coords, color=color, alpha=alpha, s=s, zorder=zorder)
    texts = []
    # Overlay text (jersey numbers) at each player's position
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


def add_pitch_info(
    ax,
    home_team_name: str,
    away_team_name: str,
    home_team_color: str,
    away_team_color: str,
    home_text_color: str,
    away_text_color: str,
    time_str: str = ""
):
    """
    Display team labels (names) and time information on the pitch.

    Parameters:
        ax (plt.Axes): The Matplotlib Axes object on which to place the labels.
        home_team_name (str): Name of the home team, displayed on the left.
        away_team_name (str): Name of the away team, displayed on the right.
        home_team_color (str): Background color for the home team label box.
        away_team_color (str): Background color for the away team label box.
        home_text_color (str): Text color for the home team name.
        away_text_color (str): Text color for the away team name.
        time_str (str): Initial time string to display at the center-top of the pitch (optional).

    Returns:
        None
    """
    # Home team label (top left)
    ax.text(
        0.01, 1.03,
        home_team_name,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        color=home_text_color,
        ha="left",
        va="center",
        bbox=dict(facecolor=home_team_color, alpha=0.8, boxstyle="round,pad=0.3")
    )

    # Away team label (top right)
    ax.text(
        0.99, 1.03,
        away_team_name,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        color=away_text_color,
        ha="right",
        va="center",
        bbox=dict(facecolor=away_team_color, alpha=0.8, boxstyle="round,pad=0.3")
    )

    # Time label (centered above the pitch)
    ax.text(
        0.5, 1.02,
        time_str,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color="white",
        ha="center",
        va="bottom",
        bbox=dict(facecolor="black", alpha=0.75, boxstyle="round,pad=0.5")
    )


def visualize_frame(
    frame_index: int,
    event_df: pl.DataFrame,
    players_df: pl.DataFrame,
    metadata_df: pl.DataFrame,
    ball_trajectory: List[Tuple[float]],
) -> None:
    """
    Visualize a single frame on the pitch, including home/away players, the ball, and its trajectory.

    Parameters:
        frame_index (int): Index of the frame to visualize.
        event_df (pl.DataFrame): Dataframe containing event information (including frame data).
        players_df (pl.DataFrame): Dataframe containing player information to join with events.
        metadata_df (pl.DataFrame): Dataframe with match metadata (team colors, names, etc.).
        ball_trajectory (List[Tuple[float]]): List of (x, y) coordinates for the ball, forming a trajectory up to this frame.

    Returns:
        None
    """
    # Retrieve data for the specified frame and merge with player information
    frame_df = pl.from_dict(event_df.row(frame_index, named=True))
    join_table = frame_df.join(players_df, on="gameEventId")
    
    # Extract metadata (team names, colors, etc.)
    match_id = event_df[frame_index]["gameId"].item()
    metadata_game = metadata_df.filter(pl.col("gameId") == str(match_id)).row(0, named=True)
    home_team_name = metadata_game["homeTeamName"]
    home_team_color = metadata_game["homeTeamColor"]
    away_team_name = metadata_game["awayTeamName"]
    away_team_color = metadata_game["awayTeamColor"]
    
    def get_contrasting_text_color(team_color: str) -> str:
        """
        Choose black or white text to ensure contrast with the background (team_color).
        """
        color = team_color.strip().lower()
        return "black" if color in ["#ffffff", "white"] else "white"
    
    home_text_color = get_contrasting_text_color(home_team_color)
    away_text_color = get_contrasting_text_color(away_team_color)
    
    # Create the pitch using settings from config
    pitch = Pitch(**config.PITCH_SETTINGS)
    fig, ax = pitch.draw(figsize=config.PITCH_FIGSIZE)
    
    # Separate and plot home players
    home_coords = (
        join_table.filter(pl.col("team") == "home")
        .select(pl.col("x"), pl.col("y"), pl.col("jerseyNum"))
        .to_numpy()
    )
    # Separate and plot away players
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
    current_time = event_df.row(frame_index, named=True)['frameTime']
    add_pitch_info(
        ax,
        home_team_name,
        away_team_name,
        home_team_color,
        away_team_color,
        home_text_color,
        away_text_color,
        current_time
    )
    
    # Draw the ball using the configured image
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
    
    # Draw ball trajectory if available
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
    """
    Create an interactive widget to navigate through shot frames.
    Displays:
      - A pitch visualization (left) for the selected frame.
      - The corresponding video frame (right), if available.
      - Playback/slider controls at the bottom.

    Parameters:
        frames (List[int]): List of frame indices to navigate through.
        event_df (pl.DataFrame): Event dataframe containing relevant frames.
        players_df (pl.DataFrame): Dataframe with player information.
        metadata_df (pl.DataFrame): Dataframe with match metadata.
        output_dir (str): Directory to store and retrieve downloaded video frames.
        show_video (bool): If True, display video frames alongside the pitch.
        interval (str): Milliseconds between frames when "Play" is pressed.

    Returns:
        None
    """
    # Play widget
    play = widgets.Play(
        value=0,
        min=0,
        max=len(frames) - 1,
        step=1,
        interval=int(interval),
        description="Play",
        disabled=False
    )
    
    # Slider widget
    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(frames) - 1,
        description="Frame:",
        continuous_update=False,
        layout=widgets.Layout(width="300px")
    )
    widgets.jslink((play, 'value'), (slider, 'value'))

    pitch_output = widgets.Output(layout=widgets.Layout(**config.PITCH_OUTPUT_LAYOUT))
    video_output = widgets.Output(layout=widgets.Layout(**config.VIDEO_OUTPUT_LAYOUT))

    # Pre-download video files if show_video is True
    video_files = {}
    if show_video:
        event_dicts = event_df.to_dicts()
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(download_video_frame, f_idx, event_dicts[f_idx], output_dir): f_idx
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

    def on_value_change(change):
        """
        Callback for slider value changes. Updates pitch and video outputs.
        """
        new_val = change["new"]
        frame_index = frames[new_val]
        ball_traj = ball_coordinates[: new_val + 1]

        with pitch_output:
            clear_output(wait=True)
            visualize_frame(frame_index, event_df, players_df, metadata_df, ball_traj)

        if show_video:
            with video_output:
                clear_output(wait=True)
                vfile = video_files.get(frame_index)
                if vfile:
                    display(Image(filename=vfile, width=400))
                else:
                    print("Frame not available.")
        else:
            with video_output:
                clear_output(wait=True)
                print("Video disabled.")

    slider.observe(on_value_change, names="value")
    on_value_change({"new": 0})

    top_box = widgets.HBox(
        [pitch_output, video_output],
        layout=widgets.Layout(**config.TOP_BOX_LAYOUT)
    )
    
    controls_box = widgets.HBox(
        [play, slider],
        layout=widgets.Layout(**config.CONTROLS_BOX_LAYOUT)
    )
    
    # Combine everything in a vertical layout
    ui = widgets.VBox(
        [top_box, controls_box],
        layout=widgets.Layout(width="100%", align_items="center")
    )
    display(ui)


def label_shot_chains_with_options(
    chains: List[List[int]],
    chain_range: Tuple[int, int],
    event_df: pl.DataFrame,
    players_df: pl.DataFrame,
    metadata_df: pl.DataFrame,
    output_dir: str,
    show_video: bool = True,
    interval: int = 1000
) -> List[List[int]]:
    """
    Provide an interactive UI to label a subset of shot chains.

    For each chain within the specified range:
      1) Display a navigation widget showing each frame on the pitch (and video if available).
      2) Offer three choices:
         - Accept Entire Chain: Keep the chain as is.
         - Accept & Customize: Keep only selected frames from the chain.
         - Discard Chain: Ignore the chain entirely.
         
    Parameters:
        chains (List[List[int]]): A list of chains, each chain is a list of frame indices.
        chain_range (Tuple[int, int]): (start, end) indices defining which chains to label.
        event_df (pl.DataFrame): Dataframe containing event information.
        players_df (pl.DataFrame): Dataframe with player details for joining.
        metadata_df (pl.DataFrame): Dataframe containing match metadata.
        output_dir (str): Directory for video frames.
        show_video (bool): If True, displays video frames for each event.
        interval (int): Milliseconds between frames in the Play widget.

    Returns:
        List[List[int]]: A list of lists, each sub-list containing the accepted frame indices for a chain.
    """
    accepted_chains = []
    current_chain_index = chain_range[0]
    
    main_output = widgets.Output()
    extra_output = widgets.Output()
    
    # Create action buttons
    accept_entire_button = Button(
        description="Accept Entire Chain",
        button_style="success",
        layout=Layout(**config.BUTTON_STYLE)
    )
    customize_button = Button(
        description="Accept & Customize",
        button_style="info",
        layout=Layout(**config.BUTTON_STYLE)
    )
    discard_button = Button(
        description="Discard Chain",
        button_style="danger",
        layout=Layout(**config.BUTTON_STYLE)
    )

    buttons_box = widgets.HBox([accept_entire_button, customize_button, discard_button])
    
    def update_ui():
        """
        Show the navigation widget (pitch + video) for the current chain.
        """
        with main_output:
            clear_output(wait=True)
            shot_frames_navigator(
                chains[current_chain_index],
                event_df,
                players_df,
                metadata_df,
                output_dir,
                show_video=show_video,
                interval=interval
            )
    
    def next_chain():
        """
        Proceed to the next chain if available, or indicate completion.
        """
        nonlocal current_chain_index
        current_chain_index += 1
        with extra_output:
            clear_output(wait=True)
        if current_chain_index < chain_range[1]:
            update_ui()
        else:
            with main_output:
                clear_output(wait=True)
                print("Labeling complete!")

    def on_accept_entire(_):
        """
        Accept the entire chain without modifications.
        """
        accepted_chains.append(chains[current_chain_index])
        print(f"Chain {current_chain_index} accepted entirely.")
        next_chain()
    
    def on_discard(_):
        """
        Discard the chain entirely.
        """
        print(f"Chain {current_chain_index} discarded.")
        next_chain()
    
    def on_customize(_):
        """
        Accept only a subset of frames from the current chain, chosen by the user.
        """
        current_chain = chains[current_chain_index]
        selection_widget = widgets.SelectMultiple(
            options=current_chain,
            value=tuple(current_chain),  # default to all frames selected
            description="Keep frames:",
            disabled=False
        )
        confirm_button = widgets.Button(
            description="Confirm Custom Selection",
            button_style="info"
        )
        
        def on_confirm(_):
            selected_frames = list(selection_widget.value)
            accepted_chains.append(selected_frames)
            print(f"Chain {current_chain_index} accepted with custom selection: {selected_frames}")
            next_chain()
        
        confirm_button.on_click(on_confirm)
        with extra_output:
            clear_output(wait=True)
            display(selection_widget, confirm_button)
    
    # Attach callbacks to buttons
    accept_entire_button.on_click(on_accept_entire)
    customize_button.on_click(on_customize)
    discard_button.on_click(on_discard)
    
    # Display the first chain
    update_ui()
    
    ui = widgets.VBox([main_output, extra_output, buttons_box])
    display(ui)
    
    return accepted_chains


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
    metadata_game = metadata_df.filter(pl.col("gameId") == str(match_id)).row(0, named=True)
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
        current_time = frame_df.row(0, named=True)['frameTime']
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
        repeat=True
    )
    plt.close(fig)
    return anim
