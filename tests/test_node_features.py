import polars as pl

from soccerai.data.data import load_and_process_soccer_events
from soccerai.data.enrichers.player_velocity import PlayerVelocityEnricher


def test_add_velocity_two_players():
    _, players_df = load_and_process_soccer_events(
        "/home/soccerdata/FIFA_WorldCup_2022/Event Data"
    )
    players = players_df.filter(pl.col("index").is_in([0, 1]))

    enricher = PlayerVelocityEnricher(
        "/home/soccerdata/FIFA_WorldCup_2022/Tracking Data"
    )
    augmented_players = enricher.add_velocity_per_player(players)

    assert "velocity" in augmented_players.columns, "Velocity column missing"
    assert "direction" in augmented_players.columns, "Direction column missing"
