import numpy as np

from soccerai.data.utils import compute_velocity, create_event_byte_map


def test_compute_velocity():
    velocity, direction = compute_velocity(np.ones((2,)), 1.0)
    assert np.isclose(velocity, np.sqrt(2))
    assert np.isclose(direction, 45.0)


def test_build_event_byte_map():
    tracking_file = "/home/soccerdata/FIFA_WorldCup_2022/Tracking Data/3839.jsonl"
    event_byte_map = create_event_byte_map(tracking_file)
    assert len(event_byte_map) != 0
