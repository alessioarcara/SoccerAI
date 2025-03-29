from typing import Optional
from datetime import datetime
import requests
from utils import normalize


def search_player_id(player_name: str, nationality: str) -> Optional[str]:
    search_url = f"http://localhost:8000/players/search/{player_name}"
    response = requests.get(search_url)
    if response.status_code != 200:
        print(f"Error during player search: {response.status_code}")
        return None

    data = response.json()
    candidates = data.get("results", [])
    normalized_nationality = normalize(nationality)

    matches = [
        player for player in candidates
        if normalized_nationality in [normalize(n) for n in player.get("nationalities", [])]
    ]

    if not matches:
        print(f" No match for {player_name} ({nationality})")
        return None

    # If multiple candidates match (same full name), select the one with the highest market value.
    best_match = max(matches, key=lambda p: p.get("marketValue") or 0)
    player_id = best_match["id"]
    print(f"Found player ID for {player_name}: {player_id}")
    return player_id


def get_avg_market_value_pre2021(player_id: str, num_years_back: int = 3) -> Optional[int]:
    mv_url = f"http://localhost:8000/players/{player_id}/market_value"
    response = requests.get(mv_url)
    if response.status_code != 200:
        print(f"Error retrieving market value: {response.status_code}")
        return None

    data = response.json()
    history = data.get("marketValueHistory", [])
    if not history:
        print(f"No market value history available for player {player_id}")
        return None

    start_date = datetime(2021, 12, 31)
    end_date = datetime(2021 - num_years_back, 1, 1)
    values = []

    for entry in history:
        try:
            entry_date = datetime.strptime(entry.get("date", ""), "%Y-%m-%d")
        except ValueError:
            continue

        value = entry.get("marketValue")
        if value is not None and end_date <= entry_date < start_date:
            values.append(value)

    if values:
        avg = int(sum(values) / len(values))
        print(f"Average market value ({end_date.year}-{start_date.year - 1}): €{avg:,}")
        return avg
    else:
        print(f"No values in range {end_date.year}-{start_date.year - 1} for player {player_id}")
        return None
    

def get_player_physical_profile(player_id: str) -> dict:
    url = f"http://localhost:8000/players/{player_id}/profile"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error retrieving profile info: {response.status_code}")
        return {}

    data = response.json()
    birth_str = data.get("dateOfBirth")
    height = data.get("height")  
    age = data.get("age")

    return {
        "birth_date": birth_str,
        "age": age,
        "height_cm": height
    }
