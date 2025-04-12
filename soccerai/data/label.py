from typing import List, Optional, Tuple

import polars as pl
from data import config
from data.visualize import shot_frames_navigator
from IPython.display import clear_output, display
from ipywidgets import Button, Layout, widgets


def pos_labeling(
    event_df: pl.DataFrame, chain_len: Optional[int] = None
) -> List[List[int]]:
    shots_df = event_df.filter(event_df["possessionEventType"] == "SH")
    pos_chains = []

    for shot in shots_df.iter_rows(named=True):
        shot_idx = shot["index"]
        team_name = shot["teamName"]

        pos_chain = [shot_idx]
        prev_idx = shot_idx - 1

        while (
            prev_idx >= 0
            and event_df.row(prev_idx, named=True)["teamName"] == team_name
        ):
            # Skip challenge events
            if event_df.row(prev_idx, named=True)["possessionEventType"] == "CH":
                prev_idx -= 1
                continue

            pos_chain.append(prev_idx)
            prev_idx -= 1

        pos_chain = pos_chain[::-1]

        if chain_len is None or len(pos_chain) >= chain_len:
            pos_chains.append(pos_chain)

    return pos_chains


def is_within_range(
    event_df: pl.DataFrame,
    players_df: pl.DataFrame,
    metadata_df,
    last_action_idx: int,
    team_name: str,
    outer_distance: float,
    inner_distance: float,
) -> bool:
    chain_last_action_event_df = event_df.filter(pl.col("index") == last_action_idx)
    game_id = chain_last_action_event_df.select("gameId").item()

    try:
        metadata_event = metadata_df.filter(pl.col("gameId").cast(int) == game_id).row(
            0, named=True
        )

        ball_last_action = (
            chain_last_action_event_df.join(
                players_df, on=["gameEventId", "possessionEventId"]
            )
            .filter(pl.col("team").is_null())
            .row(0, named=True)
        )
    except Exception:
        return False

    home_team_name = metadata_event["homeTeamName"]
    home_team_start_left = metadata_event["homeTeamStartLeft"]
    second_half_start = metadata_event["startPeriod2"]
    frame_time = ball_last_action["frameTime"]
    x_position = ball_last_action["x"]

    try:
        minutes, seconds = map(int, frame_time.split(":"))
        current_time_seconds = minutes * 60 + seconds
        is_second_half = current_time_seconds >= second_half_start
    except Exception:
        return False

    is_within_left_range = inner_distance <= x_position <= outer_distance
    is_within_right_range = (
        (105 - outer_distance) <= x_position <= (105 - inner_distance)
    )
    is_home_team = team_name == home_team_name

    if is_home_team:
        if home_team_start_left:
            if not is_second_half:
                result = is_within_right_range
            else:
                result = is_within_left_range
        else:
            if not is_second_half:
                result = is_within_left_range
            else:
                result = is_within_right_range
    else:
        if home_team_start_left:
            if not is_second_half:
                result = is_within_left_range
            else:
                result = is_within_right_range
        else:
            if not is_second_half:
                result = is_within_right_range
            else:
                result = is_within_left_range

    return result


def neg_labeling(
    event_df: pl.DataFrame,
    players_df: pl.DataFrame,
    metadata_df: pl.DataFrame,
    pos_chains: List[List[int]],
    chain_len: int,
    outer_distance: float,
    inner_distance: float = 0.0,
) -> List[List[int]]:
    pos_indices = [idx for chain in pos_chains for idx in chain]
    negatives_df = event_df.filter(~pl.col("index").is_in(pos_indices))

    neg_chains = []
    neg_chain = []
    curr_team_name = None

    for row in negatives_df.iter_rows(named=True):
        idx = row["index"]
        team_name = row["teamName"]

        if curr_team_name == team_name:
            neg_chain.append(idx)
        else:
            if len(neg_chain) >= chain_len and is_within_range(
                event_df,
                players_df,
                metadata_df,
                neg_chain[-1],
                curr_team_name,
                outer_distance,
                inner_distance,
            ):
                neg_chains.append(neg_chain)

            neg_chain = [idx]
            curr_team_name = team_name

    return neg_chains


def filter_shot_chains(
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
