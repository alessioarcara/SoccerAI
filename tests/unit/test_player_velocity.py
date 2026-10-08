import numpy as np
import polars as pl
import pytest

from soccerai.data.enrichers.player_velocity import PlayerVelocityEnricher


def test_velocity_assignment_preserves_interleaved_input_rows(monkeypatch):
    df = pl.DataFrame(
        {
            "gameId": [1, 2, 1, 1],
            "gameEventId": [10, 10, 20, 10],
            "team": ["home"] * 4,
            "jerseyNum": ["1"] * 4,
        }
    )
    enricher = PlayerVelocityEnricher("/tmp")
    monkeypatch.setattr(enricher, "_create_event_byte_map", lambda _: {10: 10, 20: 20})

    def tracking(path, pos):
        speed = pos + (100 if path.endswith("/2.jsonl") else 0)
        return np.float64(1), np.zeros(3), {"1": np.array([speed, 0.0])}, {}

    monkeypatch.setattr(enricher, "_extract_tracking_data", tracking)
    out = enricher.add_velocity_per_player(df)
    assert out["velocity"].to_list() == [10.0, 110.0, 20.0, 10.0]
    assert out.drop("velocity", "direction").equals(df)


@pytest.mark.parametrize("elapsed", [0.0, -1.0, float("nan")])
def test_velocity_rejects_invalid_time_deltas(monkeypatch, elapsed):
    enricher = PlayerVelocityEnricher("/tmp")
    df = pl.DataFrame(
        {"gameId": [1], "gameEventId": [10], "team": ["home"], "jerseyNum": ["1"]}
    )
    monkeypatch.setattr(enricher, "_create_event_byte_map", lambda _: {10: 0})
    monkeypatch.setattr(
        enricher, "_extract_tracking_data", lambda *_: (elapsed, np.zeros(3), {}, {})
    )
    assert enricher.add_velocity_per_player(df)["velocity"][0] is None


def test_vertical_ball_motion_does_not_create_horizontal_speed():
    velocity, _ = PlayerVelocityEnricher("/tmp")._compute_velocity(
        np.array([0.0, 0.0, 10.0]), np.float64(1)
    )
    assert velocity == 0.0
