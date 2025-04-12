import unicodedata
from functools import lru_cache
from typing import Dict, Optional

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


@lru_cache(maxsize=1024)
def normalize(text: str) -> str:
    return (
        "".join(
            c
            for c in unicodedata.normalize("NFKD", text)
            if not unicodedata.combining(c)
        )
        .lower()
        .strip()
    )


def get_api_data(url: str, headers: Optional[Dict[str, str]] = None) -> Optional[dict]:
    try:
        response = requests.get(url, headers=headers)
    except Exception as e:
        print(f"Request error for {url}: {e}")
        return None
    if response.status_code != 200:
        print(f"Error fetching data from {url}: {response.status_code}")
        return None
    try:
        return response.json()
    except Exception as e:
        print(f"JSON decode error for {url}: {e}")
        return None


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
