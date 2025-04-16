import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
from numpy.typing import NDArray

from soccerai.data.utils import (
    compute_deltas,
    compute_velocity,
    create_event_byte_map,
    extract_frame_info,
    is_valid_frame,
    offset_x,
    offset_y,
    read_lines_backwards_from_offset,
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
        "frameTime": event["possessionEvents"]["formattedGameClock"],
    }


def extract_players(
    event: Dict[str, Any],
    byte_pos: Optional[int] = None,
) -> List[Dict[str, Any]]:
    players = []
    game_id = event["gameId"]
    game_event_id = event["gameEventId"]
    possession_event_id = event["possessionEventId"]
    if byte_pos is not None and byte_pos != -1:
        time_elapsed, ball_delta, home_players_deltas, away_players_deltas = (
            extract_tracking_data(game_id, byte_pos)
        )
    else:
        ball_delta = None
        time_elapsed = None
        home_players_deltas = None
        away_players_deltas = None

    def extract_entity(
        positions_delta: NDArray[np.float64],
        team: str,
        z: float,
        jerseyNum: str,
        visibility: str,
    ) -> Dict[str, Any]:
        if (
            positions_delta is not None
            and time_elapsed is not None
            and player["jerseyNum"] in positions_delta
        ):
            velocity, direction = compute_velocity(
                positions_delta[player["jerseyNum"]],
                time_elapsed,
            )
        else:
            velocity = None
            direction = None
        return {
            "gameId": game_id,
            "gameEventId": game_event_id,
            "possessionEventId": possession_event_id,
            "jerseyNum": jerseyNum,
            "x": offset_x(player["x"]),
            "y": offset_y(player["y"]),
            "z": z,
            "velocity": velocity,
            "direction": direction,
            "team": team,
            "visibility": visibility,
        }

    for player in event["homePlayers"]:
        players.append(
            extract_entity(home_players_deltas, "home", 0.0, player["jerseyNum"], None)
        )
    for player in event["awayPlayers"]:
        players.append(
            extract_entity(away_players_deltas, "away", 0.0, player["jerseyNum"], None)
        )
    ball = event["ball"]
    players.append(
        extract_entity(ball_delta, None, ball["z"], None, ball["visibility"])
    )
    return players


def extract_metadata(game_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "gameId": game_metadata[0]["id"],
        "awayTeamName": game_metadata[0]["awayTeam"]["name"],
        "awayTeamColor": game_metadata[0]["awayTeamKit"]["primaryColor"],
        "homeTeamName": game_metadata[0]["homeTeam"]["name"],
        "homeTeamColor": game_metadata[0]["homeTeamKit"]["primaryColor"],
        "homeTeamStartLeft": game_metadata[0]["homeTeamStartLeft"],
        "startPeriod2": game_metadata[0]["startPeriod2"],
    }


def extract_tracking_data(
    game_id: int, byte_pos: int
) -> Union[
    Tuple[
        np.floating,
        NDArray[np.float64],
        Dict[str, NDArray[np.float64]],
        Dict[str, NDArray[np.float64]],
    ],
    Tuple[None, ...],
]:
    tracking_file = f"/home/soccerdata/FIFA_WorldCup_2022/Tracking Data/{game_id}.jsonl"
    time_deltas: List[List[float]] = []
    ball_deltas: List[List[float]] = []
    home_players_deltas: Dict[str, List[List[float]]] = {}
    away_players_deltas: Dict[str, List[List[float]]] = {}
    previous_frame = -1

    frames = read_lines_backwards_from_offset(tracking_file, byte_pos, max_lines=60)
    if len(frames) > 0:
        last_frame_info = json.loads(frames[-1])
    if len(frames) >= 2 and (
        last_frame_info["game_event"]["game_event_type"]
        not in {"SUB", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF"}
    ):
        for frame in frames:
            frame_info = json.loads(frame)
            if is_valid_frame(frame_info, previous_frame):
                ball_deltas, home_players_deltas, away_players_deltas, time_deltas = (
                    extract_frame_info(
                        ball_deltas,
                        home_players_deltas,
                        away_players_deltas,
                        time_deltas,
                        frame_info,
                    )
                )
            previous_frame = frame_info["frameNum"]

    if len(ball_deltas) < 2:
        if len(ball_deltas) == 1:
            ball_deltas = []
            home_players_deltas = {}
            away_players_deltas = {}
            time_deltas = []
        previous_frame = -1
        with open(tracking_file, "r") as file:
            file.seek(byte_pos)
            for _ in range(4):
                line = file.readline()
                if not line:
                    break
                frame_info = json.loads(line)
                if is_valid_frame(frame_info, previous_frame):
                    (
                        ball_deltas,
                        home_players_deltas,
                        away_players_deltas,
                        time_deltas,
                    ) = extract_frame_info(
                        ball_deltas,
                        home_players_deltas,
                        away_players_deltas,
                        time_deltas,
                        frame_info,
                    )
                previous_frame = frame_info["frameNum"]

    if len(ball_deltas) < 2:
        return None, None, None, None

    ball_delta = np.median(compute_deltas(np.array(ball_deltas)), axis=0)
    home_players_delta: Dict[str, NDArray[np.float64]] = {}
    for jersey_num, player_frames in list(home_players_deltas.items()):
        home_players_delta[jersey_num] = np.median(
            compute_deltas(np.array(player_frames)), axis=0
        )
    away_players_delta: Dict[str, NDArray[np.float64]] = {}
    for jersey_num, player_frames in list(away_players_deltas.items()):
        away_players_delta[jersey_num] = np.median(
            compute_deltas(np.array(player_frames)), axis=0
        )
    time_delta = np.median(compute_deltas(np.array(time_deltas)))

    return time_delta, ball_delta, home_players_delta, away_players_delta


def extract_player_info(player_info: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "playerId": player_info["player"]["id"],
        "playerNickname": player_info["player"]["nickname"],
        "shirtNumber": player_info["shirtNumber"],
        "playerTeam": player_info["team"]["name"],
        "playerRole": player_info["positionGroupType"],
    }


def load_and_process_soccer_events(
    base_dir_path: str,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    event_dir_path = os.path.join(base_dir_path, "Event Data")
    event_files = [f for f in os.listdir(event_dir_path) if f.endswith(".json")]

    all_events = []
    all_players = []
    for event_file in event_files:
        with open(os.path.join(event_dir_path, event_file), "r") as f:
            data = json.load(f)
        event_byte_map = create_event_byte_map(
            f"{os.path.join(base_dir_path, 'Tracking Data')}/{data[0]['gameId']}.jsonl"
        )
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


def load_and_process_roosters(
    roosters_dir_path: str,
) -> pl.DataFrame:
    roosters_files = [f for f in os.listdir(roosters_dir_path) if f.endswith(".json")]

    roosters = []
    teams = []

    for rooster_file in roosters_files:
        with open(os.path.join(roosters_dir_path, rooster_file), "r") as f:
            match_roosters_data = json.load(f)

        # first player is an home team player
        home_team_name = match_roosters_data[0]["team"]["name"]
        for i in range(len(match_roosters_data)):
            if match_roosters_data[i]["team"]["name"] != home_team_name:
                away_team_name = match_roosters_data[i]["team"]["name"]
                break

        for player_info in match_roosters_data:
            player_team = player_info["team"]["name"]
            if player_team not in teams:
                roosters.append(extract_player_info(player_info))

        if home_team_name not in teams:
            teams.append(home_team_name)
        if away_team_name not in teams:
            teams.append(away_team_name)

    roosters_df = pl.DataFrame(roosters)

    return roosters_df
