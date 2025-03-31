import os
from typing import Dict, Any, List, Tuple, Optional
import polars as pl
import json
from utils import (
    offset_x,
    offset_y,
    compute_velocity,
    compute_direction,
    create_event_byte_map,
)
import numpy as np


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


def extract_players(
    event: Dict[str, Any],
    byte_pos: Optional[int] = None,
) -> List[Dict[str, Any]]:

    players = []
    game_id = event["gameId"]
    event_id = event["gameEventId"]
    if byte_pos is not None and byte_pos != -1:
        time, ball_pos, players_pos = extract_tracking_data(game_id, byte_pos)
    else:
        ball_pos = None
        time = None
        players_pos = {"home": None, "away": None}

    for player in event["homePlayers"]:
        if (
            players_pos["home"] is not None
            and player["jerseyNum"] in players_pos["home"]["start"]
            and player["jerseyNum"] in players_pos["home"]["end"]
        ):
            player_velocity = compute_velocity(
                players_pos["home"]["start"][player["jerseyNum"]],
                players_pos["home"]["end"][player["jerseyNum"]],
                time["start"],
                time["end"],
            )
            player_direction = compute_direction(
                players_pos["home"]["start"][player["jerseyNum"]],
                players_pos["home"]["end"][player["jerseyNum"]],
            )
        else:
            player_velocity = None
            player_direction = None
        players.append(
            {
                "gameId": game_id,
                "gameEventId": event_id,
                "jerseyNum": player["jerseyNum"],
                "x": offset_x(player["x"]),
                "y": offset_y(player["y"]),
                "z": 0.0,
                "velocity": player_velocity,
                "direction": player_direction,
                "team": "home",
            }
        )

    for player in event["awayPlayers"]:
        if (
            players_pos["away"] is not None
            and player["jerseyNum"] in players_pos["away"]["start"]
            and player["jerseyNum"] in players_pos["away"]["end"]
        ):
            player_velocity = compute_velocity(
                players_pos["away"]["start"][player["jerseyNum"]],
                players_pos["away"]["end"][player["jerseyNum"]],
                time["start"],
                time["end"],
            )
            player_direction = compute_direction(
                players_pos["away"]["start"][player["jerseyNum"]],
                players_pos["away"]["end"][player["jerseyNum"]],
            )
        else:
            player_velocity = None
            player_direction = None
        players.append(
            {
                "gameId": game_id,
                "gameEventId": event_id,
                "jerseyNum": player["jerseyNum"],
                "x": offset_x(player["x"]),
                "y": offset_y(player["y"]),
                "z": 0.0,
                "velocity": player_velocity,
                "direction": player_direction,
                "team": "away",
            }
        )
    ball = event["ball"]
    if ball_pos is not None:
        ball_velocity = compute_velocity(
            ball_pos["start"],
            ball_pos["end"],
            time["start"],
            time["end"],
        )
        ball_direction = compute_direction(ball_pos["start"], ball_pos["end"])
    else:
        ball_velocity = None
        ball_direction = None
    players.append(
        {
            "gameId": game_id,
            "gameEventId": event_id,
            "jerseyNum": None,
            "x": offset_x(ball["x"]),
            "y": offset_y(ball["y"]),
            "z": ball["z"],
            "velocity": ball_velocity,
            "direction": ball_direction,
            "team": None,
            "visibility": ball["visibility"],
        }
    )
    return players


def extract_metadata(game_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "gameId": game_metadata[0]["id"],
        "awayTeamName": game_metadata[0]["awayTeam"]["name"],
        "awayTeamColor": game_metadata[0]["awayTeamKit"]["primaryColor"],
        "homeTeamName": game_metadata[0]["homeTeam"]["name"],
        "homeTeamColor": game_metadata[0]["homeTeamKit"]["primaryColor"],
    }


def extract_tracking_data(
    game_id: int, byte_pos: int
) -> Tuple[Dict[str, Any], float, Tuple[float]]:

    tracking_file = f"/home/soccerdata/FIFA_WorldCup_2022/Tracking Data/{game_id}.jsonl"
    players_pos = {
        "home": {"start": {}, "end": {}},
        "away": {"start": {}, "end": {}},
    }
    frame_key = "start"
    time = {}
    ball_pos = {}
    frames_pair_found = False
    with open(tracking_file, "r") as tracking_data:
        while not frames_pair_found:
            if tracking_data.tell() == 0:
                tracking_data.seek(byte_pos)
            line = tracking_data.readline()
            if line:
                frame_info = json.loads(line)
                if (
                    frame_info["ballsSmoothed"] is None
                    or frame_info["ballsSmoothed"]["x"] is None
                    or frame_info["ballsSmoothed"]["y"] is None
                ):
                    continue
                else:
                    ball_pos[frame_key] = (
                        frame_info["ballsSmoothed"]["x"],
                        frame_info["ballsSmoothed"]["y"],
                    )

                time[frame_key] = np.round(frame_info["videoTimeMs"] / 1000, 3)
                if frame_info["homePlayersSmoothed"] is not None:
                    for player in frame_info["homePlayersSmoothed"]:
                        players_pos["home"][frame_key][player["jerseyNum"]] = (
                            player["x"],
                            player["y"],
                        )
                else:
                    continue
                if frame_info["awayPlayersSmoothed"] is not None:
                    for player in frame_info["awayPlayersSmoothed"]:
                        players_pos["away"][frame_key][player["jerseyNum"]] = (
                            player["x"],
                            player["y"],
                        )
                else:
                    continue
                if frame_key == "start":
                    frame_key = "end"
                else:
                    frames_pair_found = True
            else:
                ball_pos = None
                players_pos["home"] = None
                players_pos["away"] = None
                time["end"] = None
                break

    return time, ball_pos, players_pos


def load_and_process_soccer_events(
    event_dir_path: str,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    event_files = [f for f in os.listdir(event_dir_path) if f.endswith(".json")]

    all_events = []
    all_players = []
    for event_file in event_files:
        with open(os.path.join(event_dir_path, event_file), "r") as f:
            data = json.load(f)
        event_byte_map = create_event_byte_map(data[0]["gameId"])
        for e in data:
            if (
                e["homePlayers"] is None
                or e["awayPlayers"] is None
                or e["ball"] is None
            ):
                continue

            all_events.append(extract_event(e))
            if e["gameEventId"] in event_byte_map:
                all_players.extend(extract_players(e, event_byte_map[e["gameEventId"]]))
            else:
                all_players.extend(extract_players(e))
    event_df = pl.DataFrame(all_events).with_row_index()
    players_df = pl.DataFrame(all_players).with_row_index()

    return event_df, players_df


def load_and_process_metadata(
    metadata_dir_path: str,
) -> pl.DataFrame:
    metadata_files = [f for f in os.listdir(metadata_dir_path) if f.endswith(".json")]

    metadata_matches = []

    for metadata_file in metadata_files:
        with open(os.path.join(metadata_dir_path, metadata_file), "r") as f:
            data = json.load(f)

        metadata_matches.append(extract_metadata(data))

    metadata_df = pl.DataFrame(metadata_matches).with_row_index()

    return metadata_df
