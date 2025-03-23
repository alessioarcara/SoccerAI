from typing import Dict,List,Optional, Any
import time
import statistics
import random
import polars as pl 
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException



def create_webdriver():
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument('user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)"')
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    return webdriver.Chrome(options=chrome_options)



def get_player_id(driver, player_name: str) -> str:
    base_search_url = "https://fbref.com/search/search.fcgi?search="
    query = player_name.replace(" ", "+")
    search_url = base_search_url + query
    
    #print(f"URL visited: {search_url}")

    driver.get(search_url)
    links = driver.find_elements(By.CSS_SELECTOR, "a")
    
    for link in links:
        url = link.get_attribute("href")
        if url and "/en/players/" in url:
            #print(f"Founded valid player URL: {url}")
            player_id = url.split("/")[5]
            return player_id
    
    #print("No valid URL per player founded.")
    return None


def get_player_html(driver, player_name, player_id):
    url = f"https://fbref.com/en/players/{player_id}/{player_name.replace(' ', '-')}"
    try:
        driver.get(url)
        time.sleep(random.uniform(0, 2))
        return driver.page_source
    except WebDriverException as e:
        print(f"Error download on the {player_name}'s page: {e}")
        return None




def enrich_roosters(rooster_df: pl.DataFrame) -> pl.DataFrame:
    rooster_rows = rooster_df.to_dicts()
    updated_rows = []
    driver = create_webdriver()

    for player in rooster_rows[:5]:
        player_name = player["playerNickname"]
        #print(player_name)
        player_id = get_player_id(driver, player_name)
        # print(player_id)
        player_html = get_player_html(driver, player_name, player_id)
        player_metastats = parse_player_metastats(player_html)
        print(f"Metastats added for {player_name}")
        # print(player_metastats)
        player_shooting_stats = extract_shoot_stats(player_html,num_years_back=3,min_minute_90s=5)
        
        player_info = player_metastats | player_shooting_stats
        player.update(player_info)
        updated_rows.append(player)

    enriched_roosters = pl.DataFrame(updated_rows)
    enriched_roosters.write_csv('enrich_roosters.csv')
    driver.quit()
    return enriched_roosters



def compute_shooting_stats_average(season_data, num_years_back=3, min_minutes_90s=5.0):

    indices_2021 = [i for i, (season, _) in enumerate(season_data) if season == '2021-2022']
    if not indices_2021:
        return {}

    last_index_2021 = indices_2021[-1]
    selected_seasons = season_data[max(0, last_index_2021 - num_years_back):last_index_2021 + 1]
    stats_sum = {}
    stats_count = {}
    skip_keys = {'age', 'team', 'country', 'comp_level', 'lg_finish', 'matches'}

    for _, stats in selected_seasons:
        minutes = stats.get('minutes_90s', 0.0)
        if minutes is None or minutes < min_minutes_90s:
            continue

        has_useful_data = any(
            key not in skip_keys and value is not None
            for key, value in stats.items()
        )
        
        if not has_useful_data:
            continue
    
        for key, value in stats.items():
            if key in skip_keys:
                continue
            if value is not None:
                stats_count[key] = stats_count.get(key,0) + 1
                stats_sum[key] = stats_sum.get(key,0) + value
    
    return {
        key: round(stats_sum[key] / stats_count[key],3)
        for key in stats_sum
        if stats_count[key] > 0
    }


def extract_shoot_stats(html_file: str, num_years_back: int = 3, min_minute_90s: int = 5) -> dict:
    soup = BeautifulSoup(html_file, 'html.parser')
    div_all_stats = soup.find('div', id='all_stats_shooting')

    if not div_all_stats:
        return {}

    table = div_all_stats.find('table')
    if not table:
        return {}

    tbody = table.find('tbody')
    if not tbody:
        return {}

    rows = tbody.find_all('tr')

    season_data = []
    for row in rows:
        season_th = row.find('th', {'data-stat': 'year_id'})
        if not season_th:
            continue

        season = season_th.get_text(strip=True)
        data_cells = row.find_all('td')
        if not data_cells:
            continue

        stats = {}
        for cell in data_cells:
            stat_name = cell.get('data-stat')
            stat_value = cell.get_text(strip=True)

            try:
                value = float(stat_value)
            except ValueError:
                value = None

            stats[stat_name] = value

        season_data.append((season, stats))

        
    averaged_stats = compute_shooting_stats_average(
        season_data,
        num_years_back=num_years_back,
        min_minutes_90s=min_minute_90s
    )

    return averaged_stats




def parse_player_metastats(html: str) -> pl.DataFrame:
    soup = BeautifulSoup(html, 'html.parser')
    meta_div = soup.find('div', id='meta')
    
    if not meta_div:
        return {
            "Full Name": None,
            "Position": None,
            "Footed": None,
            "Height": None,
            "Weight": None,
            "Born": None,
            "Birth Date": None,
            "Age Info": None,
            "National Team": None,
            "Weekly Wages": None
        }
        
    data = {}

    h1_full = meta_div.find('h1')
    if h1_full:
        data["Full Name"] = h1_full.get_text(strip=True)
    
    p_pos = None
    for p in meta_div.find_all('p'):
        strong_tag = p.find('strong')
        if strong_tag and "Position:" in strong_tag.get_text():
            p_pos = p
            break
    if p_pos:
        strong_tags = p_pos.find_all('strong')
        if len(strong_tags) >= 2:
            pos_text = strong_tags[0].next_sibling
            foot_text = strong_tags[1].next_sibling
            # Remove the bullet symbol if it exists
            if pos_text:
                pos_text = pos_text.replace("▪", "").strip()
            if foot_text:
                foot_text = foot_text.replace("▪", "").strip()
            data["Position"] = pos_text if pos_text else None
            data["Footed"] = foot_text if foot_text else None

    p_hw = None
    for p in meta_div.find_all('p'):
        text = p.get_text()
        if "cm" in text and "kg" in text:
            p_hw = p
            break
    if p_hw:
        spans = p_hw.find_all('span')
        if len(spans) >= 2:
            data["Height"] = spans[0].get_text(strip=True)
            data["Weight"] = spans[1].get_text(strip=True)
    
    p_born = None
    for p in meta_div.find_all('p'):
        strong = p.find('strong')
        if strong and "Born:" in strong.get_text():
            p_born = p
            break
    if p_born:
        nobr = p_born.find('nobr')
        if nobr:
            data["Age Info"] = nobr.get_text(strip=True)


    p_nat = None
    for p in meta_div.find_all('p'):
        strong = p.find('strong')
        if strong and "National Team:" in strong.get_text():
            p_nat = p
            break
    if p_nat:
        a_nat = p_nat.find('a')
        if a_nat:
            data["National Team"] = a_nat.get_text(strip=True)
    
    p_wages = None
    for p in meta_div.find_all('p'):
        if p.find('strong') and "Wages" in p.get_text():
            p_wages = p
            break
    if p_wages:
        wage_span = p_wages.find("span", style=lambda x: x and "color:#932a12" in x)
        if wage_span:
            data["Weekly Wages"] = wage_span.get_text(strip=True)

    for key, value in data.items():
        if isinstance(value, str):
            data[key] = value.replace('\xa0', ' ')
    
    return data

