import unicodedata
from functools import lru_cache
from typing import Dict, Optional

import requests


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
