from typing import Dict,Optional, Any
import polars as pl 
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import time
import pandas as pd



def get_player_html(player_name: str) -> str:
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--disable-gpu")

    driver = webdriver.Chrome(options=chrome_options)
    url = f"https://fbref.com/en/players/e14ec482/{player_name}"
    
    try:
        driver.get(url)
        html = driver.page_source

    finally:
        driver.quit()
    
    return html 


def extract_shoot_stats(html_file: str) -> pl.DataFrame:

    soup = BeautifulSoup(html_file, 'html.parser')

    div_all_stats = soup.find('div', id='all_stats_shooting')
    if not div_all_stats:
        raise Exception("Div with id 'all_stats_shooting' not found.")

    table = div_all_stats.find('table')
    if not table:
        raise Exception("Table not found within 'all_stats_shooting'.")

    tfoot = table.find('tfoot')
    if not tfoot:
        raise Exception("<tfoot> element not found in the table.")

    # Extract rows from tfoot, getting text from each cell
    rows = []
    for tr in tfoot.find_all('tr'):
        cells = [cell.get_text(strip=True) for cell in tr.find_all(['th', 'td'])]
        rows.append(cells)

    # # Determine maximum number of columns
    # max_cols = max(len(row) for row in rows)

    # # Pad rows with fewer columns with None
    # normalized_rows = [row + [None]*(max_cols - len(row)) for row in rows]

    # # If all rows have the same number of cells and you want to use the first row as header:
    # if normalized_rows and all(len(row) == len(normalized_rows[0]) for row in normalized_rows):
    #     header = normalized_rows[0]
    #     data = normalized_rows[1:]
    #     df = pl.DataFrame(data, schema=header)
    # else:
    df = pd.DataFrame(rows)

    return df



def parse_player_metastats(html: str) -> pl.DataFrame:
    soup = BeautifulSoup(html, 'html.parser')
    meta_div = soup.find('div', id='meta')
    
    if not meta_div:
        schema = {
            "Full Name": pl.Utf8,
            "Position": pl.Utf8,
            "Footed": pl.Utf8,
            "Height": pl.Utf8,
            "Weight": pl.Utf8,
            "Born": pl.Utf8,
            "Birth Date": pl.Utf8,
            "Age Info": pl.Utf8,
            "National Team": pl.Utf8,
            "Weekly Wages": pl.Utf8
        }
        return pl.DataFrame(schema=schema)
    
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


