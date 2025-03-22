import json
import os
from typing import Any, Dict, List, Tuple

import polars as pl
from utils import offset_x, offset_y


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
        "label": None,
    }


def extract_players(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    players = []
    game_id = event["gameId"]
    event_id = event["gameEventId"]
    # players_velocities, end_time, ball_end_pos, players_end_pos = extract_tracking_data(
    #     game_id, event_id
    # )
    # ball_velocity, ball_direction = compute_velocity(
    #     (event["ball"]["x"], event["ball"]["y"]),
    #     ball_end_pos,
    #     event["end_time"],
    #     end_time,
    #     True,
    # )
    for player in event["homePlayers"]:
        players.append(
            {
                "gameId": game_id,
                "gameEventId": event_id,
                "jerseyNum": player["jerseyNum"],
                "x": offset_x(player["x"]),
                "y": offset_y(player["y"]),
                "z": 0.0,
                "velocity": 0.0,
                "team": "home",
            }
        )
    for player in event["awayPlayers"]:
        players.append(
            {
                "gameId": game_id,
                "gameEventId": event_id,
                "jerseyNum": player["jerseyNum"],
                "x": offset_x(player["x"]),
                "y": offset_y(player["y"]),
                "z": 0.0,
                "team": "away",
            }
        )
    ball = event["ball"]
    players.append(
        {
            "gameId": game_id,
            "gameEventId": event_id,
            "jerseyNum": None,
            "x": offset_x(ball["x"]),
            "y": offset_y(ball["y"]),
            "z": ball["z"],
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
    game_id: int, event_id: str
) -> Tuple[Dict[str, Any], float, Tuple[float]]:
    pass
    # tracking_file = f"./FIFA_WorldCup_2022/Tracking Data/{game_id}.jsonl.bz2"
    # velocities = {"home": {}, "away": {}}
    # players_pos = {"home": {}, "away": {}}
    # end_frame = -1
    # with bz2.BZ2File(tracking_file, "r") as tracking_data:
    #     for frame in tracking_data:
    #         frame_info = json.loads(frame.decode())
    #         if frame_info["game_event_id"] == event_id and (
    #             frame_info["frameNum"] == frame_info["game_event"]["end_frame"]
    #         ):
    #             end_frame = frame_info["frameNum"]
    #             # for player in frame_info["homePlayers"]:
    #             #     velocities["home"][player["jerseyNum"]] = player["speed"]
    #             # for player in frame_info["awayPlayers"]:
    #             #     velocities["away"][player["jerseyNum"]] = player["speed"]
    #         elif frame_info["game_event_id"] == end_frame + 1:
    #             ball_pos = (frame_info["balls"][0]["x"], frame_info["balls"][0]["y"])
    #             end_time = np.round(frame_info["videoTimeMs"] / 1000, 3)
    #             for player in frame_info["homePlayers"]:
    #                 players_pos["home"][player["jerseyNum"]] = (
    #                     player["x"],
    #                     player["y"],
    #                 )
    #             for player in frame_info["awayPlayers"]:
    #                 players_pos["away"][player["jerseyNum"]] = (
    #                     player["x"],
    #                     player["y"],
    #                 )
    #             break

    # return velocities, end_time, ball_pos, players_pos


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
