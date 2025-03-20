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
    decompress_tracking_file,
    compress_tracking_file,
)
import bz2
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
    byte_pos_ball: Optional[int] = None,
    byte_pos_players: Optional[int] = None,
) -> List[Dict[str, Any]]:

    players = []
    game_id = event["gameId"]
    event_id = event["gameEventId"]
    if byte_pos_ball is not None:
        players_velocities, ball_end_time, ball_end_pos, players_end_pos = (
            extract_tracking_data(game_id, byte_pos_ball, byte_pos_players)
        )
    else:
        players_velocities = None
        ball_end_time = None
        ball_end_pos = None
        players_end_pos = None

    for player in event["homePlayers"]:
        if players_velocities is not None:
            player_velocity = players_velocities["home"][player["jerseyNum"]]
            player_direction = compute_direction(
                (player["x"], player["y"]), players_end_pos["home"][player["jerseyNum"]]
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
        if players_velocities is not None:
            player_velocity = players_velocities["away"][player["jerseyNum"]]
            player_direction = compute_direction(
                (player["x"], player["y"]), players_end_pos["away"][player["jerseyNum"]]
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
    ball_start_time = (
        event["endTime"] if event["endTime"] is not None else event["startTime"]
    )
    if ball_end_pos is not None and ball["x"] is not None and ball["y"] is not None:
        ball_velocity = compute_velocity(
            (ball["x"], ball["y"]), ball_end_pos, ball_start_time, ball_end_time
        )
        ball_direction = compute_direction((ball["x"], ball["y"]), ball_end_pos)
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
    game_id: int, byte_pos_ball: int, byte_pos_player: int
) -> Tuple[Dict[str, Any], float, Tuple[float]]:

    tracking_file = f"/home/soccerdata/FIFA_WorldCup_2022/Tracking Data/{game_id}.jsonl"
    velocities = {"home": {}, "away": {}}
    players_pos = {"home": {}, "away": {}}
    with open(tracking_file, "r") as tracking_data:
        try:
            tracking_data.seek(byte_pos_player)
        except ValueError:
            print(byte_pos_player)
        frame_info = json.loads(tracking_data.readline())
        if frame_info["homePlayersSmoothed"] is not None:
            for player in frame_info["homePlayersSmoothed"]:
                velocities["home"][player["jerseyNum"]] = player.get("speed", 0)
        else:
            velocities["home"] = None
        if frame_info["awayPlayersSmoothed"] is not None:
            for player in frame_info["awayPlayersSmoothed"]:
                velocities["away"][player["jerseyNum"]] = player.get("speed", 0)
        else:
            velocities["away"] = None
        try:
            tracking_data.seek(byte_pos_ball)
        except ValueError:
            print(byte_pos_ball)
        frame_info = json.loads(tracking_data.readline())
        if (
            frame_info["ballsSmoothed"] is None
            or frame_info["ballsSmoothed"]["x"] is None
            or frame_info["ballsSmoothed"]["y"] is None
        ):
            if len(frame_info["balls"]) != 0:
                ball_pos = (frame_info["balls"][0]["x"], frame_info["balls"][0]["y"])
            else:
                ball_pos = None
        else:
            ball_pos = (
                frame_info["ballsSmoothed"]["x"],
                frame_info["ballsSmoothed"]["y"],
            )

        end_time = np.round(frame_info["videoTimeMs"] / 1000, 3)
        for player in frame_info["homePlayersSmoothed"]:
            players_pos["home"][player["jerseyNum"]] = (
                player["x"],
                player["y"],
            )
        for player in frame_info["awayPlayersSmoothed"]:
            players_pos["away"][player["jerseyNum"]] = (
                player["x"],
                player["y"],
            )

    return velocities, end_time, ball_pos, players_pos


def load_and_process_soccer_events(
    event_dir_path: str,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    event_files = [f for f in os.listdir(event_dir_path) if f.endswith(".json")]

    all_events = []
    all_players = []
    count_game = 0
    for event_file in event_files:
        with open(os.path.join(event_dir_path, event_file), "r") as f:
            data = json.load(f)
        tracking_filepath = f"/home/soccerdata/FIFA_WorldCup_2022/Tracking Data/{data[0]['gameId']}.jsonl.bz2"
        decompress_tracking_file(tracking_filepath)
        event_byte_map = create_event_byte_map(data[0]["gameId"])
        count_game += 1
        print(f"Game number {count_game} in progress...")
        count_event = 0
        for e in data:
            count_event += 1
            if (
                e["homePlayers"] is None
                or e["awayPlayers"] is None
                or e["ball"] is None
            ):
                continue

            all_events.append(extract_event(e))
            print(f"Event number {e["gameEventId"]} in progress...")
            if e["gameEventId"] in event_byte_map:
                all_players.extend(
                    extract_players(
                        e,
                        event_byte_map[e["gameEventId"]]["ball_pos"],
                        event_byte_map[e["gameEventId"]]["players_pos"],
                    )
                )
            else:
                all_players.extend(extract_players(e))
        compress_tracking_file(tracking_filepath.removesuffix(".bz2"))
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
