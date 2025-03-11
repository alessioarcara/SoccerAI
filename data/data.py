import os
from typing import Dict, Any, List, Tuple
import polars as pl
import json
from utils import offset_x_by_60, offset_y_by_40


def extract_event(event: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "gameId": event["gameId"],
        "gameEventId": event["gameEventId"],
        "possessionEventId": event["possessionEventId"],
        "startTime": event["startTime"],
        "endTime": event["endTime"],
        "duration": event["duration"],
        "gameEventType": event["gameEvents"]["gameEventType"],
        "possessionEventType": event["possessionEvents"]["possessionEventType"],
        "teamName": event["gameEvents"]["teamName"],
        "playerName": event["gameEvents"]["playerName"],
        "videoUrl": event["gameEvents"]["videoUrl"],
        "label": None,
    }


def extract_players(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    players = []
    for player in event["homePlayers"]:
        players.append(
            {
                "gameId": event["gameId"],
                "gameEventId": event["gameEventId"],
                "jerseyNum": player["jerseyNum"],
                "x": offset_x_by_60(player["x"]),
                "y": offset_y_by_40(player["y"]),
                "z": 0.0,
                "team": "home",
            }
        )
    for player in event["awayPlayers"]:
        players.append(
            {
                "gameId": event["gameId"],
                "gameEventId": event["gameEventId"],
                "jerseyNum": player["jerseyNum"],
                "x": offset_x_by_60(player["x"]),
                "y": offset_y_by_40(player["y"]),
                "z": 0.0,
                "team": "away",
            }
        )
    ball = event["ball"]
    players.append(
        {
            "gameId": event["gameId"],
            "gameEventId": event["gameEventId"],
            "jerseyNum": None,
            "x": offset_x_by_60(ball["x"]),
            "y": offset_y_by_40(ball["y"]),
            "z": ball["z"],
            "team": None,
        }
    )
    return players


def load_and_process_soccer_events(
    event_dir_path: str,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    event_files = [f for f in os.listdir(event_dir_path) if f.endswith(".json")]

    all_events = []
    all_players = []
    for event_file in event_files:
        with open(os.path.join(event_dir_path, event_file), "r") as f:
            data = json.load(f)

        for e in data:
            if (
                e["homePlayers"] is None
                or e["awayPlayers"] is None
                or e["ball"] is None
                or e["gameEvents"]["gameEventType"] == "OUT"
            ):
                continue

            all_events.append(extract_event(e))
            all_players.extend(extract_players(e))

    event_df = pl.DataFrame(all_events).with_row_index()
    players_df = pl.DataFrame(all_players).with_row_index()

    return event_df, players_df
