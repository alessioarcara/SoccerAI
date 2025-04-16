import json

from soccerai.data.data import extract_players
from soccerai.data.utils import create_event_byte_map


def test_extract_players():
    with open("tests/sample_event.json", "r") as f:
        event = json.load(f)

    tracking_file = "/home/soccerdata/FIFA_WorldCup_2022/Tracking Data/3812.jsonl"
    game_event_byte_map = create_event_byte_map(tracking_file)
    game_event_id = list(game_event_byte_map.keys())[0]
    byte_pos = game_event_byte_map[game_event_id]

    players = extract_players(
        event,
        byte_pos,
    )
    assert len(players) > 0
