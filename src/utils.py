import csv
import logging
import time
from typing import Any, Dict, List

import requests

API_KEYS = [
    "lK1jwHrVg47249kaPqBQ_Vfv__bCHBZz",
    "_gPpNVJ0f_ysCqoLRb8X0pazEbgQjEYO",
    "Al9_qYnR6QsRI5OP3J3RXli1cFGuM3sy",
    "i_fOUC69wtNd7xXfuvrvooWZEuW5qPf8",
    "oMHUUEhafW0qWnjRJj5CSrITRlbxgWvT",
    "2zaSEssGyoopXspPyrgL9Q0ESDfyY_9v",
]

API_KEY_COUNTER = 0

SYMBOL_TO_WORD = {
    "c": "close",
    "h": "high",
    "l": "low",
    "o": "open",
    "v": "volume",
    "t": "timestamp",
    "vw": "volume_weighted_average_price",
    "n": "number_of_trades",
}

SEARCH_PARAMS = "?adjusted=true&sort=asc&limit=50000"

logging.basicConfig(
    filename="logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def get_tickers(file_path: str) -> List[str]:
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        next(reader)
        return [row[0] for row in reader]


def call_polygon_api(url: str) -> Dict[str, Any]:
    logging.info(f"Calling the Polygon API with URL: {url}")
    response = requests.get(url)
    if response.status_code != 200:
        logging.error(
            f"Failed to call the Polygon API. Status code: {response.status_code}."
            f"Response: {response.text}"
        )
        raise Exception(
            f"Failed to call the Polygon API. Status code: {response.status_code}."
        )
    time.sleep(12 / len(API_KEYS))
    return response.json()
