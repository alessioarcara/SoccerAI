import numpy as np
import polars as pl
from IPython.display import clear_output, display
from ipywidgets import Button, Layout, widgets
from loguru import logger
from tqdm.notebook import tqdm

from soccerai.data import config
from soccerai.data.annotations import FRAME_KEYS
from soccerai.data.data import _flatten_chains
from soccerai.data.utils import home_attacks_right
from soccerai.data.visualize import shot_frames_navigator


def get_chains(
    event_df: pl.DataFrame,
    players_df: pl.DataFrame,
    metadata_df: pl.DataFrame,
    rosters_df: pl.DataFrame,
    chain_len: int = 6,
    outer_distance: float = 25.0,
    inner_distance: float = 0.0,
    skip_challenge_events: bool = True,
    use_player_pos: bool = False,
) -> dict[str, list[list[int]]]:
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
        rosters_df,
        all_pos_chains,
        2,
        outer_distance,
        inner_distance,
        use_player_pos,
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
    all_chains: list[list[int]], chain_len: int
) -> tuple[list[list[int]], ...]:
    long_chains: list[list[int]] = []
    short_chains: list[list[int]] = []

    for chain in all_chains:
        (long_chains if len(chain) >= chain_len else short_chains).append(chain)

    return long_chains, short_chains


def _pos_labeling(
    event_df: pl.DataFrame, chain_len: int, skip_challenge_events: bool
) -> list[list[int]]:
    rows = event_df.sort("index").to_dicts()
    shot_positions = [
        i for i, row in enumerate(rows) if row["possessionEventType"] == "SH"
    ]
    pos_chains = []

    for shot_pos in tqdm(
        shot_positions,
        total=len(shot_positions),
        desc="Computing positive chains",
        colour="green",
    ):
        shot = rows[shot_pos]
        shot_idx = shot["index"]
        team_name = shot["teamName"]
        if team_name is None or shot["period"] not in (1, 2, 3, 4):
            continue

        pos_chain = [shot_idx]
        prev_pos = shot_pos - 1
        while prev_pos >= 0:
            previous = rows[prev_pos]
            if (
                previous["teamName"] != team_name
                or previous["gameId"] != shot["gameId"]
                or previous["period"] != shot["period"]
                or previous["possessionEventType"] == "SH"
                or previous["possessionEventType"] is None
            ):
                break
            if not (skip_challenge_events and previous["possessionEventType"] == "CH"):
                pos_chain.append(previous["index"])
            prev_pos -= 1

        pos_chain = pos_chain[::-1]

        if len(pos_chain) >= chain_len:
            pos_chains.append(pos_chain)

    return pos_chains


def _is_within_range(
    event_df: pl.DataFrame,
    players_df: pl.DataFrame,
    metadata_df: pl.DataFrame,
    rosters_df: pl.DataFrame,
    last_action_idx: int,
    team_name: str,
    outer_distance: float,
    inner_distance: float,
    use_player_pos: bool,
) -> bool:
    """
    Whether the last action of a chain happens between `inner_distance` and
    `outer_distance` metres from the goal line attacked by `team_name`.
    """
    last_action_event_df = event_df.filter(pl.col("index") == last_action_idx)
    if last_action_event_df.height != 1:
        logger.warning(
            "{} events with index {}: chain discarded",
            last_action_event_df.height,
            last_action_idx,
        )
        return False
    game_id = last_action_event_df.select("gameId").item()
    period = last_action_event_df.select("period").item()
    if period not in (1, 2, 3, 4):
        logger.debug(
            "Event {} has period {!r}: chain discarded", last_action_idx, period
        )
        return False
    period = int(period)

    metadata_rows = metadata_df.filter(pl.col("gameId").cast(int) == game_id)
    if metadata_rows.height == 0:
        logger.warning("No metadata for game {}: chain discarded", game_id)
        return False
    metadata_event = metadata_rows.row(0, named=True)
    home_team_name = metadata_event["homeTeamName"]
    away_team_name = metadata_event["awayTeamName"]

    identity = last_action_event_df.row(0, named=True)
    frame_players = players_df.filter(
        pl.all_horizontal(
            [pl.col(c).eq_missing(pl.lit(identity[c])) for c in FRAME_KEYS]
        )
    )
    joined_df = last_action_event_df.join(
        frame_players, on=FRAME_KEYS, nulls_equal=True
    )

    if use_player_pos:
        candidates = (
            joined_df.with_columns(
                pl.when(pl.col("team") == "home")
                .then(pl.lit(home_team_name))
                .when(pl.col("team") == "away")
                .then(pl.lit(away_team_name))
                .otherwise(None)
                .alias("team_name_mapped")
            )
            .join(
                rosters_df,
                left_on=["team_name_mapped", "jerseyNum"],
                right_on=["playerTeam", "shirtNumber"],
                how="left",
            )
            .filter(pl.col("playerName") == pl.col("playerName_right"))
        )
    else:
        candidates = joined_df.filter(pl.col("team").is_null())

    if candidates.height == 0:
        logger.debug(
            "No {} found for event {}: chain discarded",
            "ball carrier" if use_player_pos else "ball",
            last_action_idx,
        )
        return False

    x_position = candidates.row(0, named=True)["x"]
    if x_position is None or not np.isfinite(x_position):
        return False

    is_within_left_range = inner_distance <= x_position <= outer_distance
    is_within_right_range = (
        (105 - outer_distance) <= x_position <= (105 - inner_distance)
    )

    is_home_team = team_name == home_team_name
    attacks_right = (
        home_attacks_right(
            period,
            metadata_event["homeTeamStartLeft"],
            metadata_event.get("homeTeamStartLeftExtraTime"),
        )
        == is_home_team
    )

    return is_within_right_range if attacks_right else is_within_left_range


def _neg_labeling(
    event_df: pl.DataFrame,
    players_df: pl.DataFrame,
    metadata_df: pl.DataFrame,
    rosters_df: pl.DataFrame,
    pos_chains: list[list[int]],
    chain_len: int,
    outer_distance: float,
    inner_distance: float = 0.0,
    use_player_pos: bool = False,
) -> list[list[int]]:
    pos_indices = set(_flatten_chains(pos_chains))
    neg_chains: list[list[int]] = []
    run: list[dict] = []
    current_key = None

    def finish_run():
        if not run:
            return
        # A possession containing any shot is never a negative example,
        # including shots whose positive chain is below the length threshold.
        if any(
            r["possessionEventType"] in ("SH", None) or r["index"] in pos_indices
            for r in run
        ):
            return
        indices = [r["index"] for r in run]
        if len(indices) >= chain_len and _is_within_range(
            event_df,
            players_df,
            metadata_df,
            rosters_df,
            indices[-1],
            run[0]["teamName"],
            outer_distance,
            inner_distance,
            use_player_pos,
        ):
            neg_chains.append(indices)

    for row in tqdm(
        event_df.sort("index").iter_rows(named=True),
        total=event_df.height,
        desc="Computing negative chains",
        colour="red",
    ):
        key = (row["gameId"], row["period"], row["teamName"])
        valid = row["teamName"] is not None and row["period"] in (1, 2, 3, 4)
        if not valid or key != current_key:
            finish_run()
            run = []
        current_key = key if valid else None
        if valid:
            run.append(row)
    finish_run()
    return neg_chains


def filter_shot_chains(
    chains: list[list[int]],
    chains_range: tuple[int, int],
    event_df: pl.DataFrame,
    players_df: pl.DataFrame,
    metadata_df: pl.DataFrame,
    output_dir: str,
    show_video: bool = True,
    interval: int = 1000,
) -> list[list[int]]:
    if not 0 <= chains_range[0] < chains_range[1] <= len(chains):
        raise ValueError("chains_range must select a nonempty range of existing chains")
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
            accept_button.disabled = discard_button.disabled = True
            selection_widget.disabled = True
            with main_output:
                clear_output(wait=True)
                print("Labeling complete!")

    def on_accept(_):
        nonlocal selection_widget
        selected_frames = list(selection_widget.value)
        if not selected_frames:
            return
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
