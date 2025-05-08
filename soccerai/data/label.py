from typing import Dict, List, Tuple

import numpy as np
import polars as pl
from IPython.display import clear_output, display
from ipywidgets import Button, Layout, widgets
from loguru import logger
from tqdm.notebook import tqdm

from soccerai.data import config
from soccerai.data.data import _flatten_chains
from soccerai.data.visualize import shot_frames_navigator


def get_chains(
    event_df: pl.DataFrame,
    players_df: pl.DataFrame,
    metadata_df: pl.DataFrame,
    chain_len: int = 6,
    outer_distance: float = 25.0,
    inner_distance: float = 0.0,
    skip_challenge_events: bool = True,
) -> Dict[str, List[List[int]]]:
    """
    Categorizes event sequences in soccer matches into chains. Extracts
    positive chains (those leading to shots) and negative chains (those not
    leading to shots), and further classifies them as long or short based on a
    defined chain_len threshold.
    Returns a dictionary containing all categorized chains.
    """
    all_pos_chains = _pos_labeling(event_df, 2, skip_challenge_events)
    pos_long_chains, pos_short_chains = _split_into_long_short_chains(
        all_pos_chains, chain_len
    )
    logger.success(
        "Positive chains: total={}, long={}, short={}",
        len(all_pos_chains),
        len(pos_long_chains),
        len(pos_short_chains),
    )

    all_neg_chains = _neg_labeling(
        event_df,
        players_df,
        metadata_df,
        all_pos_chains,
        2,
        outer_distance,
        inner_distance,
    )
    neg_long_chains, neg_short_chains = _split_into_long_short_chains(
        all_neg_chains, chain_len
    )
    logger.success(
        "Negative chains: total={}, long={}, short={}",
        len(all_neg_chains),
        len(neg_long_chains),
        len(neg_short_chains),
    )

    total_positive_events = np.sum([len(chain) for chain in all_pos_chains])
    logger.info(f"Original positive events: {total_positive_events}")
    logger.info(f"Augmented positive events: {total_positive_events * 4}")

    return {
        "all_pos_chains": all_pos_chains,
        "pos_long_chains": pos_long_chains,
        "pos_short_chains": pos_short_chains,
        "all_neg_chains": all_neg_chains,
        "neg_long_chains": neg_long_chains,
        "neg_short_chains": neg_short_chains,
    }


def _split_into_long_short_chains(
    all_chains: List[List[int]], chain_len: int
) -> Tuple[List[List[int]], ...]:
    long_chains: List[List[int]] = []
    short_chains: List[List[int]] = []

    for chain in all_chains:
        (long_chains if len(chain) >= chain_len else short_chains).append(chain)

    return long_chains, short_chains


def _pos_labeling(
    event_df: pl.DataFrame, chain_len: int, skip_challenge_events: bool
) -> List[List[int]]:
    shots_df = event_df.filter(event_df["possessionEventType"] == "SH")
    pos_chains = []

    for shot in tqdm(
        shots_df.iter_rows(named=True),
        total=shots_df.height,
        desc="Computing positive chains",
        colour="green",
    ):
        shot_idx = shot["index"]
        team_name = shot["teamName"]

        pos_chain = [shot_idx]
        prev_idx = shot_idx - 1

        while (
            prev_idx >= 0
            and event_df.row(prev_idx, named=True)["teamName"] == team_name
        ):
            if (
                skip_challenge_events
                and event_df.row(prev_idx, named=True)["possessionEventType"] == "CH"
            ):
                prev_idx -= 1
                continue

            pos_chain.append(prev_idx)
            prev_idx -= 1

        pos_chain = pos_chain[::-1]

        if len(pos_chain) >= chain_len:
            pos_chains.append(pos_chain)

    return pos_chains


def _is_within_range(
    event_df: pl.DataFrame,
    players_df: pl.DataFrame,
    metadata_df: pl.DataFrame,
    last_action_idx: int,
    team_name: str,
    outer_distance: float,
    inner_distance: float,
) -> bool:
    last_action_event_df = event_df.filter(pl.col("index") == last_action_idx)
    game_id = last_action_event_df.select("gameId").item()

    try:
        metadata_event = metadata_df.filter(pl.col("gameId").cast(int) == game_id).row(
            0, named=True
        )

        ball = (
            last_action_event_df.join(
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
    frame_time = ball["frameTime"]
    x_position = ball["x"]

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


def _neg_labeling(
    event_df: pl.DataFrame,
    players_df: pl.DataFrame,
    metadata_df: pl.DataFrame,
    pos_chains: List[List[int]],
    chain_len: int,
    outer_distance: float,
    inner_distance: float = 0.0,
) -> List[List[int]]:
    pos_indices = _flatten_chains(pos_chains)
    negatives_df = event_df.filter((~pl.col("index").is_in(pos_indices)))

    neg_chains = []
    neg_chain = []
    curr_team_name = negatives_df[0, "teamName"]

    for row in tqdm(
        negatives_df.iter_rows(named=True),
        total=negatives_df.height,
        desc="Computing negative chains",
        colour="red",
    ):
        idx = row["index"]
        team_name = row["teamName"]

        if curr_team_name == team_name:
            neg_chain.append(idx)
        else:
            if len(neg_chain) >= chain_len and _is_within_range(
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
