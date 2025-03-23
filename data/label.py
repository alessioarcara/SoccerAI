from typing import List, Optional, Tuple

import config
import polars as pl
from IPython.display import clear_output, display
from ipywidgets import Button, Layout, widgets
from visualize import shot_frames_navigator


def pos_labeling(
    event_df: pl.DataFrame, chain_len: Optional[int] = None
) -> List[List[int]]:
    """ """
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


def neg_labeling(event_df: pl.DataFrame) -> List[List[int]]:
    pass

    # all_indices = list(range(event_data.height))
    # negative_indices = [index for index in all_indices if index not in indices]
    # tmp_indices = []
    # negative_indices_copy = negative_indices.copy()

    # for idx in range(len(negative_indices_copy) - 1):
    #     print(tmp_indices)
    #     tmp_indices.append(negative_indices_copy[idx])
    #     if not (negative_indices_copy[idx + 1] - negative_indices_copy[idx] == 1
    #         and event_data.row(negative_indices_copy[idx + 1], named=True)['teamName'] ==  event_data.row(negative_indices_copy[idx], named=True)['teamName']):
    #         if len(tmp_indices) < 5:
    #             negative_indices = [index for index in negative_indices if index not in tmp_indices]

    #         print(tmp_indices)
    #         tmp_indices = []
    #         print(tmp_indices)

    # event_data = event_data.with_columns(
    #     pl.when(pl.col("idx").is_in(indices)).then(1)
    #     .when(pl.col("idx").is_in(negative_indices)).then(0)
    #     .alias("label")
    # )
    # event_data = event_data.filter(pl.col("label").is_not_null())


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
