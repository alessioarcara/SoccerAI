import numpy as np
import polars as pl

from soccerai.data.data import load_and_process_soccer_events
from soccerai.data.velocity import (
    add_velocity_per_player,
    compute_velocity,
    create_event_byte_map,
)


def test_compute_velocity():
    velocity, direction = compute_velocity(np.ones((2,)), 1.0)
    assert np.isclose(velocity, np.sqrt(2))
    assert np.isclose(direction, 45.0)


def test_build_event_byte_map():
    tracking_file = "/home/soccerdata/FIFA_WorldCup_2022/Tracking Data/3839.jsonl"
    event_byte_map = create_event_byte_map(tracking_file)
    assert len(event_byte_map) != 0


def test_add_velocity_two_players():
    _, players_df = load_and_process_soccer_events(
        "/home/soccerdata/FIFA_WorldCup_2022/Event Data"
    )
    players = players_df.filter(pl.col("index").is_in([0, 1]))
    augmented_players = add_velocity_per_player(
        "/home/soccerdata/FIFA_WorldCup_2022/Tracking Data", players
    )

    assert "velocity" in augmented_players.columns, "Velocity column missing"
    assert "direction" in augmented_players.columns, "Direction column missing"
