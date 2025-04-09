import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from utils import (
    compute_deltas,
    compute_velocity,
    create_event_byte_map,
    offset_x,
    offset_y,
    read_last_n_lines,
)


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
        time_delta, ball_delta, home_players_deltas, away_players_deltas = (
            extract_tracking_data(game_id, byte_pos)
        )
    else:
        ball_delta = None
        time_delta = None
        home_players_deltas = None
        away_players_deltas = None

    for player in event["homePlayers"]:
        if (
            home_players_deltas is not None
            and player["jerseyNum"] in home_players_deltas
        ):
            player_velocity, player_direction = compute_velocity(
                home_players_deltas[player["jerseyNum"]],
                time_delta,
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
                "possessionEventId": event["possessionEventId"],
            }
        )

    for player in event["awayPlayers"]:
        if (
            away_players_deltas is not None
            and player["jerseyNum"] in away_players_deltas
        ):
            player_velocity, player_direction = compute_velocity(
                away_players_deltas[player["jerseyNum"]],
                time_delta,
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
    if ball_delta is not None:
        ball_velocity, ball_direction = compute_velocity(ball_delta, time_delta)
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
    time_delta = []
    ball_delta = []
    home_players_deltas = {}
    away_players_deltas = {}
    previous_frame = -1

    frames = read_last_n_lines(tracking_file, byte_pos, max_lines=60)
    if len(frames) > 0:
        last_frame_info = json.loads(frames[-1])
    if len(frames) >= 2 and (
        last_frame_info["game_event"] is None
        or last_frame_info["game_event"]["game_event_type"] != "SECONDKICKOFF"
    ):
        for frame in frames:
            frame_info = json.loads(frame)
            if (
                frame_info["frameNum"] == previous_frame
                or frame_info["ballsSmoothed"] is None
                or frame_info["ballsSmoothed"]["x"] is None
                or frame_info["ballsSmoothed"]["y"] is None
                or frame_info["homePlayersSmoothed"] is None
                or frame_info["awayPlayersSmoothed"] is None
            ):
                previous_frame = frame_info["frameNum"]
                continue
            else:
                ball_delta.append(
                    [
                        frame_info["ballsSmoothed"]["x"],
                        frame_info["ballsSmoothed"]["y"],
                        frame_info["ballsSmoothed"]["z"],
                    ]
                )
                for player in frame_info["homePlayersSmoothed"]:
                    home_list = home_players_deltas.get(player["jerseyNum"], [])
                    home_list.append(
                        [
                            player["x"],
                            player["y"],
                        ]
                    )
                    home_players_deltas[player["jerseyNum"]] = home_list
                for player in frame_info["awayPlayersSmoothed"]:
                    away_list = away_players_deltas.get(player["jerseyNum"], [])
                    away_list.append(
                        [
                            player["x"],
                            player["y"],
                        ]
                    )
                    away_players_deltas[player["jerseyNum"]] = away_list

                time_delta.append([np.round(frame_info["videoTimeMs"] / 1000, 3)])
                previous_frame = frame_info["frameNum"]

    if ball_delta is not None and len(ball_delta) < 2:
        if len(ball_delta) == 1:
            ball_delta = []
            home_players_deltas = {}
            away_players_deltas = {}
            time_delta = []
        previous_frame = -1
        with open(tracking_file, "r") as file:
            file.seek(byte_pos)
            for _ in range(4):
                line = file.readline()
                if line:
                    frame_info = json.loads(line)
                    if (
                        frame_info["frameNum"] == previous_frame
                        or frame_info["ballsSmoothed"] is None
                        or frame_info["ballsSmoothed"]["x"] is None
                        or frame_info["ballsSmoothed"]["y"] is None
                        or frame_info["homePlayersSmoothed"] is None
                        or frame_info["awayPlayersSmoothed"] is None
                    ):
                        previous_frame = frame_info["frameNum"]
                        continue
                    ball_delta.append(
                        [
                            frame_info["ballsSmoothed"]["x"],
                            frame_info["ballsSmoothed"]["y"],
                            frame_info["ballsSmoothed"]["z"],
                        ]
                    )
                    for player in frame_info["homePlayersSmoothed"]:
                        home_list = home_players_deltas.get(player["jerseyNum"], [])
                        home_list.append(
                            [
                                player["x"],
                                player["y"],
                            ]
                        )
                        home_players_deltas[player["jerseyNum"]] = home_list
                    for player in frame_info["awayPlayersSmoothed"]:
                        away_list = away_players_deltas.get(player["jerseyNum"], [])
                        away_list.append(
                            [
                                player["x"],
                                player["y"],
                            ]
                        )
                        away_players_deltas[player["jerseyNum"]] = away_list

                    time_delta.append([np.round(frame_info["videoTimeMs"] / 1000, 3)])
                    previous_frame = frame_info["frameNum"]
                else:
                    break
    if ball_delta is not None and len(ball_delta) < 2:
        ball_delta = None
        time_delta = None
        home_players_deltas = None
        away_players_deltas = None

    if ball_delta is not None:
        if len(ball_delta) % 2 == 1:
            ball_delta.pop(-1)
        ball_delta = np.median(compute_deltas(ball_delta), axis=0)
        for jersey_num, frames in list(home_players_deltas.items()):
            home_players_deltas[jersey_num] = np.median(compute_deltas(frames), axis=0)
        for jersey_num, frames in list(away_players_deltas.items()):
            away_players_deltas[jersey_num] = np.median(compute_deltas(frames), axis=0)
        time_delta = np.median(compute_deltas(time_delta))
    return time_delta, ball_delta, home_players_deltas, away_players_deltas


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
