from typing import Dict,Optional, Any
import polars as pl 
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import time

def get_page_source(url: str, output_file) -> str:
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--disable-gpu")

    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        driver.get(url)
        html = driver.page_source
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)
    
    finally:
        driver.quit()


import pandas as pd

def extract_shoot_stats(html_file: str) -> pl.DataFrame:

    print(html_file)
    
    with open(html_file, 'r', encoding='utf-8') as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')

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



def parse_meta_stats(html: str) -> Dict[str,Any]:

    with open(html, 'r', encoding='utf-8') as f:
        html = f.read()


    soup = BeautifulSoup(html, 'html.parser')
    meta_div = soup.find('div', id='meta')
    if meta_div is None:
        print("Meta div not found.")
        return {}
    
    data = {
        "Full Name": None,
        "Position": None,
        "Footed": None,
        "Height": None,
        "Weight": None,
        "Born": None,
        "Birth Date": None,
        "Age Info": None,
        "Birth Place": None,
        "National Team": None,
        "Club": None
    }
    
    p_full = meta_div.find('p')
    if p_full:
        strong_full = p_full.find('strong')
        if strong_full:
            data["Full Name"] = strong_full.get_text(strip=True)
    
    p_pos = None
    for p in meta_div.find_all('p'):
        if p.find('strong') and "Position:" in p.find('strong').get_text():
            p_pos = p
            break
    if p_pos:
        # Example text: "Position: FW-MF (WM) ▪ Footed: Right"
        # Split the text using the "▪" delimiter.
        parts = p_pos.get_text(separator="|", strip=True).split("▪")
        if len(parts) == 2:
            data["Position"] = parts[0].replace("Position:", "").strip()
            data["Footed"] = parts[1].replace("Footed:", "").strip()
    
    for p in meta_div.find_all('p'):
        text = p.get_text()
        if "cm" in text and "kg" in text:
            spans = p.find_all('span')
            if len(spans) >= 2:
                data["Height"] = spans[0].get_text(strip=True)
                data["Weight"] = spans[1].get_text(strip=True)
            break

    p_born = None
    for p in meta_div.find_all('p'):
        strong = p.find('strong')
        if strong and "Born:" in strong.get_text():
            p_born = p
            break
    if p_born:
        span_birth = p_born.find('span', id="necro-birth")
        if span_birth:
            data["Born"] = span_birth.get_text(strip=True)
            data["Birth Date"] = span_birth.attrs.get("data-birth")
        nobr = p_born.find('nobr')
        if nobr:
            data["Age Info"] = nobr.get_text(strip=True)
        for content in p_born.contents:
            if isinstance(content, str) and content.strip().startswith("in "):
                data["Birth Place"] = content.strip()
                break

    for p in meta_div.find_all('p'):
        strong = p.find('strong')
        if strong and "National Team:" in strong.get_text():
            a_nat = p.find('a')
            if a_nat:
                data["National Team"] = a_nat.get_text(strip=True)
            break

    for p in meta_div.find_all('p'):
        strong = p.find('strong')
        if strong and "Club:" in strong.get_text():
            a_club = p.find('a')
            if a_club:
                data["Club"] = a_club.get_text(strip=True)
            break 

    return data