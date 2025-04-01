from typing import Optional, List, Dict, Union
from datetime import datetime
from bs4 import BeautifulSoup
import requests, re
from urllib.parse import quote
from utils import normalize
import Levenshtein
from rapidfuzz import fuzz
from config import TEAM_ABBREVS


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
        if normalized_nationality in [standardize_nationality(n,TEAM_ABBREVS) for n in player.get("nationalities", [])]
    ]
    
    if not matches:
        print(f" No match for {player_name} ({nationality})")
        return None

    # If multiple candidates match (same full name), select the one with the highest market value.
    best_match = max(matches, key=lambda p: p.get("marketValue") or 0)
    player_id = best_match["id"]
    print(f"Found player ID for {player_name}: {player_id}")
    return player_id


def get_avg_market_value_pre2021(mv_url: str, num_years_back: int = 3) -> Optional[int]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
    }
    response = requests.get(mv_url,headers=headers)
    if response.status_code != 200:
        print(f"Error retrieving market value: {response.status_code}")
        return None

    data = response.json()
    history = data.get("marketValueHistory", [])
    if not history:
        print(f"No market value history available.")
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
    

def get_player_physical_profile(player_id: str) -> Dict[str, Optional[Union[str, int]]]:
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


#####

def search_club_id(club_name: str) -> Optional[str]:
    encoded_club_name = quote(club_name)
    search_url = f"http://localhost:8000/clubs/search/{encoded_club_name}"
    response = requests.get(search_url)
    
    if response.status_code != 200:
        print(f"Error during club search: {response.status_code}")
        return None
    
    data = response.json()
    candidates = data.get("results", [])
    norm_club_name = normalize(club_name)
    
    matches = [
        candidate for candidate in candidates
        if standardize_nationality(candidate.get("name", ""),TEAM_ABBREVS) == norm_club_name and standardize_nationality(candidate.get("country", ""),TEAM_ABBREVS) == norm_club_name
    ]
    
    if not matches:
        print(f"No exact match found for club '{club_name}' with country '{norm_club_name}'")
        return None
    
    club_id = matches[0].get("id")
    print(f"Found club ID for '{club_name}' in '{norm_club_name}': {club_id}")
    return club_id



def get_players_from_tmrooster(rooster_url: str, season: int = 2022) -> list:    
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
    }

    response = requests.get(rooster_url,headers=headers)
    if response.status_code != 200:
        print(f"Error fetching roster page: {response.status_code}")
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    players = []
    price_pattern = re.compile(r'^\d+(?:[,\.]\d+)?\s*(mln|mila)', re.IGNORECASE)
    
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
    best_match = None
    best_distance = float('inf')
    norm_input = normalize(input_name)
    
    for player in players:
        candidate_name = player.get("name", "")
        norm_candidate = normalize(candidate_name)
        distance = Levenshtein.distance(norm_input, norm_candidate)
        if distance < best_distance:
            best_distance = distance
            best_match = player
    
    return best_match



def standardize_nationality(input_nat: str, standard_nats: List[str]) -> str:
    norm_input = normalize(input_nat)
    best_match = norm_input
    best_score = 0
    
    for nat in standard_nats:
        score = fuzz.token_sort_ratio(norm_input, normalize(nat))
        if score > best_score:
            best_score = score
            best_match = nat          

    return normalize(best_match)

