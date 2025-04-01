from typing import Optional
import time
import random
import re
import polars as pl
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException
from config import TEAM_ABBREVS, SHOOTING_STATS
from utils import normalize


def create_webdriver():
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument(
        'user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)"'
    )
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    return webdriver.Chrome(options=chrome_options)


def get_fbref_player_id(driver, player_name: str, player_team: str) -> Optional[str]:
    country_code = TEAM_ABBREVS.get(player_team)
    if not country_code:
        print(f"Country not recognized for {player_name}")
        return None

    search_url = (
        f"https://fbref.com/search/search.fcgi?search={player_name.replace(' ', '+')}"
    )
    print(f"Searching {player_name} ({country_code}) at: {search_url}")
    driver.get(search_url)
    time.sleep(random.uniform(1, 3))

    current_url = driver.current_url
    if "/en/players/" in current_url:
        parts = current_url.split("/")
        if len(parts) > 5 and parts[5]:
            print(f"Direct redirect detected: {current_url}")
            return parts[5]

    search_items = driver.find_elements(By.CSS_SELECTOR, "div.search-item")
    expected_normalized = normalize(player_name)
    name_tokens = expected_normalized.split()

    for item in search_items:
        try:
            link = item.find_element(By.CSS_SELECTOR, "div.search-item-name a")
            href = link.get_attribute("href")
            found_name = normalize(link.text)
        except Exception:
            continue

        try:
            alt_name = item.find_element(
                By.CSS_SELECTOR, "div.search-item-alt-names"
            ).text
            found_alt_name = normalize(alt_name)
        except Exception:
            found_alt_name = ""

        block_text = item.text.lower()

        year_match = re.search(r"(\d{4})[–-](\d{4})", item.text)
        if year_match:
            start_year = int(year_match.group(1))
            end_year = int(year_match.group(2))
            if end_year < 2022:
                print(f"Player retired in {end_year}, skipping...")
                continue
            if start_year > 2021:
                print(f"Player has started his career in {start_year}, skipping...")
                continue
        else:
            print("No year range found, skipping...")
            continue

        if all(token in found_name for token in name_tokens) or all(
            token in found_alt_name for token in name_tokens
        ):
            if re.search(r"\b" + re.escape(country_code) + r"\b", block_text):
                parts = href.split("/")
                if len(parts) > 5 and parts[5]:
                    return parts[5]

    print(f"No valid match for {player_name} ({player_team})")
    return None


def get_fbref_html(driver, player_name, player_id):
    url = f"https://fbref.com/en/players/{player_id}/{player_name.replace(' ', '-')}"
    print(url)
    try:
        driver.get(url)
        time.sleep(random.uniform(2, 5))
        return driver.page_source
    except WebDriverException as e:
        print(f"Error downloading page for {player_name}: {e}")
        return None


def extract_fbref_metastats(html: str) -> pl.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    meta_div = soup.find("div", id="meta")

    if not meta_div:
        return {
            "Full Name": None,
            "Height": None,
            "Weight": None,
            "Age Info": None,
        }

    data = {}

    h1_full = meta_div.find("h1")
    if h1_full:
        data["Full Name"] = h1_full.get_text(strip=True)

    p_hw = None
    for p in meta_div.find_all("p"):
        text = p.get_text()
        if "cm" in text and "kg" in text:
            p_hw = p
            break
    if p_hw:
        spans = p_hw.find_all("span")
        if len(spans) >= 2:
            data["Height"] = spans[0].get_text(strip=True)
            data["Weight"] = spans[1].get_text(strip=True)

    p_born = None
    for p in meta_div.find_all("p"):
        strong = p.find("strong")
        if strong and "Born:" in strong.get_text():
            p_born = p
            break
    if p_born:
        nobr = p_born.find("nobr")
        if nobr:
            data["Age Info"] = nobr.get_text(strip=True)

    for key, value in data.items():
        if isinstance(value, str):
            data[key] = value.replace("\xa0", " ")

    return data


def compute_shooting_stats_average(
    season_data: int, num_years_back: int = 3, min_minutes_90s: float = 5.0
):
    if not season_data:
        return {}

    indices_2021 = [
        i for i, (season, _) in enumerate(season_data) if season in "2021-2022"
    ]
    if indices_2021:
        reference_index = indices_2021[-1]
    else:
        reference_index = len(season_data) - 1

    selected_seasons = season_data[
        max(0, reference_index - num_years_back) : reference_index + 1
    ]
    stats_sum = {}
    stats_count = {}
    skip_keys = {"age", "team", "country", "comp_level", "lg_finish", "matches"}

    for _, stats in selected_seasons:
        minutes = stats.get("minutes_90s", 0.0)
        if minutes is None or minutes < min_minutes_90s:
            continue

        has_useful_data = any(
            key not in skip_keys and value is not None for key, value in stats.items()
        )
        if not has_useful_data:
            continue

        for key, value in stats.items():
            if key in skip_keys or value is None:
                continue
            stats_count[key] = stats_count.get(key, 0) + 1
            stats_sum[key] = stats_sum.get(key, 0) + value

    return {
        key: round(stats_sum[key] / stats_count[key], 3)
        for key in stats_sum
        if stats_count[key] > 0
    }


def extract_fbref_shoot_stats(
    html_file: str, num_years_back: int = 3, min_minute_90s: float = 5.0
) -> dict:
    soup = BeautifulSoup(html_file, "html.parser")
    div_all_stats = soup.find("div", id="all_stats_shooting")

    if not div_all_stats:
        return {}

    table = div_all_stats.find("table")
    if not table:
        return {}

    tbody = table.find("tbody")
    if not tbody:
        return {}

    rows = tbody.find_all("tr")

    season_data = []
    for row in rows:
        season_th = row.find("th", {"data-stat": "year_id"})
        if not season_th:
            continue

        season = season_th.get_text(strip=True)
        data_cells = row.find_all("td")
        if not data_cells:
            continue

        stats = {}
        for cell in data_cells:
            stat_name = cell.get("data-stat")
            stat_value = cell.get_text(strip=True)
            try:
                value = float(stat_value)
            except ValueError:
                value = None

            stats[stat_name] = value

        season_data.append((season, stats))

    averaged_stats = compute_shooting_stats_average(
        season_data, num_years_back=num_years_back, min_minutes_90s=min_minute_90s
    )

    filtered_stats = {key: averaged_stats.get(key) for key in SHOOTING_STATS}

    return filtered_stats
