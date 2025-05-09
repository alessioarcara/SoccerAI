import json
import os
from typing import Any, Dict, List, Tuple

import polars as pl

from soccerai.data.utils import (
    offset_x,
    offset_y,
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
) -> List[Dict[str, Any]]:
    players = []
    game_id = event["gameId"]
    game_event_id = event["gameEventId"]
    possession_event_id = event["possessionEventId"]

    def extract_entity(
        team: str | None,
        x: float,
        y: float,
        z: float,
        jerseyNum: str | None,
        visibility: str | None,
    ) -> Dict[str, Any]:
        return {
            "gameId": game_id,
            "gameEventId": game_event_id,
            "possessionEventId": possession_event_id,
            "team": team,
            "x": offset_x(x),
            "y": offset_y(y),
            "z": z,
            "jerseyNum": jerseyNum,
            "visibility": visibility,
        }

    for player in event["homePlayers"]:
        players.append(
            extract_entity(
                "home", player["x"], player["y"], 0.0, player["jerseyNum"], None
            )
        )
    for player in event["awayPlayers"]:
        players.append(
            extract_entity(
                "away", player["x"], player["y"], 0.0, player["jerseyNum"], None
            )
        )
    ball = event["ball"]
    players.append(
        extract_entity(None, ball["x"], ball["y"], ball["z"], None, ball["visibility"])
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


def extract_player_info(player_info: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "playerId": player_info["player"]["id"],
        "playerName": player_info["player"]["nickname"],
        "shirtNumber": player_info["shirtNumber"],
        "playerTeam": player_info["team"]["name"],
        "playerRole": player_info["positionGroupType"],
    }


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
            ):
                continue

            all_events.append(extract_event(e))
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


def load_and_process_rosters(
    rosters_dir_path: str,
) -> pl.DataFrame:
    rosters_files = [f for f in os.listdir(rosters_dir_path) if f.endswith(".json")]

    rosters = []
    teams = []

    for roster_file in rosters_files:
        with open(os.path.join(rosters_dir_path, roster_file), "r") as f:
            match_rosters_data = json.load(f)

        # first player is an home team player
        home_team_name = match_rosters_data[0]["team"]["name"]
        for i in range(len(match_rosters_data)):
            if match_rosters_data[i]["team"]["name"] != home_team_name:
                away_team_name = match_rosters_data[i]["team"]["name"]
                break

        for player_info in match_rosters_data:
            player_team = player_info["team"]["name"]
            if player_team not in teams:
                rosters.append(extract_player_info(player_info))

        if home_team_name not in teams:
            teams.append(home_team_name)
        if away_team_name not in teams:
            teams.append(away_team_name)

    rosters_df = pl.DataFrame(rosters)

    return rosters_df
