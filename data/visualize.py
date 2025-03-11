from mplsoccer import Pitch
import matplotlib.pyplot as plt
from typing import List
import polars as pl
import ipywidgets as widgets
from IPython.display import display, clear_output


def visualize_frame(row_index: int, event_df: pl.DataFrame, players_df: pl.DataFrame):
    frame_df = pl.from_dict(event_df.row(row_index, named=True))
    join_table = frame_df.join(players_df, on="gameEventId")

    pitch = Pitch(pitch_type="statsbomb", pitch_color="grass", line_color="#c7d5cc")
    _, ax = pitch.draw()

    home_players = join_table.filter(pl.col("team") == "home")
    away_players = join_table.filter(pl.col("team") == "away")

    home_coords = home_players.select(["x", "y"]).to_numpy()
    away_coords = away_players.select(["x", "y"]).to_numpy()

    _ = pitch.scatter(home_coords[:, 0], home_coords[:, 1], color="red", ax=ax)
    _ = pitch.scatter(away_coords[:, 0], away_coords[:, 1], color="blue", ax=ax)

    plt.tight_layout()
    plt.show()


def shot_frames_navigator(
    frames: List[int], event_df: pl.DataFrame, players_df: pl.DataFrame
):
    w = widgets.IntSlider(value=0, min=0, max=len(frames) - 1)
    output = widgets.Output()
    display(w, output)

    def on_value_change(change):
        with output:
            clear_output(wait=True)
            visualize_frame(frames[change["new"]], event_df, players_df)

    w.observe(on_value_change, names="value")

    with output:
        visualize_frame(frames[w.value], event_df, players_df)




