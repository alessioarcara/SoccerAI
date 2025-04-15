import re
from datetime import datetime
from typing import Dict, List, Optional, Union
from urllib.parse import quote

import Levenshtein
import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz
from soccerai.data.config import TEAM_ABBREVS
from soccerai.data.scraping.utils import get_api_data, normalize


def search_player_id(
    player_name: str, nationality: str, base_url: str
) -> Optional[str]:
    search_url = f"{base_url}/players/search/{player_name}"
    data = get_api_data(search_url)
    if data is None:
        print(f"Error fetching data for {player_name}")
        return None
    candidates = data.get("results", [])
    normalized_nationality = normalize(nationality)
    matches = [
        player
        for player in candidates
        if normalized_nationality
        in [
            standardize_nationality(n, TEAM_ABBREVS)
            for n in player.get("nationalities", [])
        ]
    ]
    if not matches:
        print(f"No match for {player_name} ({nationality})")
        return None
    best_match = max(matches, key=lambda p: p.get("marketValue") or 0)
    player_id = best_match["id"]
    print(f"Found player ID for {player_name}: {player_id}")
    return player_id


def get_avg_market_value_pre2021(mv_url: str, num_years_back: int = 3) -> Optional[int]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
    }
    data = get_api_data(mv_url, headers=headers)
    if data is None:
        return None
    history = data.get("marketValueHistory", [])
    if not history:
        print("No market value history available.")
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
        print(f"Average market value ({end_date.year}-{start_date.year}): €{avg:,}")
        return avg
    else:
        print(f"No values in range {end_date.year}-{start_date.year}.")
        return None


def get_player_physical_profile(
    player_id: str, base_url: str
) -> Optional[Dict[str, Optional[Union[str, int]]]]:
    url = f"{base_url}/players/{player_id}/profile"
    data = get_api_data(url)
    if data is None:
        print(f"Error during player physical profile from: {url}")
        return None
    return {
        "birth_date": data.get("dateOfBirth"),
        "age": data.get("age"),
        "height_cm": data.get("height"),
    }


def search_club_id(club_name: str, base_url: str) -> Optional[str]:
    encoded_club_name = quote(club_name)
    search_url = f"{base_url}/clubs/search/{encoded_club_name}"
    data = get_api_data(search_url)
    if data is None:
        print(f"Error during club search: could not fetch data from {search_url}")
        return None
    candidates = data.get("results", [])
    norm_club_name = normalize(club_name)
    matches = [
        candidate
        for candidate in candidates
        if standardize_nationality(candidate.get("name", ""), TEAM_ABBREVS)
        == norm_club_name
        and standardize_nationality(candidate.get("country", ""), TEAM_ABBREVS)
        == norm_club_name
    ]
    if not matches:
        print(
            f"No exact match found for club '{club_name}' with country '{norm_club_name}'"
        )
        return None
    club_id = matches[0].get("id")
    print(f"Found club ID for '{club_name}' in '{norm_club_name}': {club_id}")
    return club_id


def get_players_from_rooster(rooster_url: str) -> list:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
    }
    response = requests.get(rooster_url, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching roster page: {response.status_code}")
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    players = []
    price_pattern = re.compile(r"^\d+(?:[,\.]\d+)?\s*(mln|mila)", re.IGNORECASE)
    for td in soup.find_all("td", class_="hauptlink"):
        a = td.find("a", href=True)
        if not a:
            continue
        name = a.get_text(strip=True)
        if not name:
            continue
        if price_pattern.match(name):
            continue
        href = a["href"]
        parts = href.strip("/").split("/")
        player_id = None
        for part in parts:
            if part.isdigit():
                player_id = part
                break
        if player_id:
            players.append({"id": player_id, "name": name, "url": href})
    return players


def find_best_player_match(input_name: str, players: List[Dict]) -> Optional[Dict]:
    norm_input = normalize(input_name)
    best_match = min(
        players,
        key=lambda player: Levenshtein.distance(
            norm_input, normalize(player.get("name", ""))
        ),
        default=None,
    )
    return best_match


def standardize_nationality(input_nat: str, standard_nats: Dict[str, str]) -> str:
    norm_input = normalize(input_nat)
    normalized = {nat: normalize(nat) for nat in standard_nats.keys()}
    best_nat = max(
        standard_nats,
        key=lambda nat: fuzz.token_sort_ratio(norm_input, normalized[nat]),
    )
    return normalized[best_nat]
