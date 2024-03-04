import csv
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import click
import requests
from tqdm import tqdm

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


def get_stonks(ticker: str, date_range: Tuple, multiplier: int) -> List[Dict[str, str]]:
    global API_KEY_COUNTER

    base_url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/hour/"
    )
    url = (
        f"{base_url}{date_range[0]}/{date_range[1]}{SEARCH_PARAMS}"
        f"&apiKey={API_KEYS[API_KEY_COUNTER % len(API_KEYS)]}"
    )
    API_KEY_COUNTER += 1

    response = call_polygon_api(url)
    results: List[Dict[str, str]] = []

    if not response.get("results"):
        logging.info(f"No results found for ticker: {ticker}. Response: {response}")
        return []

    results.extend(response["results"])

    while response.get("next_url") is not None:
        logging.info(
            f"Next URL exists for ticker: {ticker}. URL: {response['next_url']}"
        )

        response = call_polygon_api(
            f"{response['next_url']}&apiKey={API_KEYS[API_KEY_COUNTER % len(API_KEYS)]}"
        )

        API_KEY_COUNTER += 1

        if not response.get("results"):
            logging.info(f"No results found for ticker: {ticker}. Response: {response}")
            continue

        results.extend(response["results"])

    return results


def _write_stocks_data_to_csv(
    ticker: str, results: List[Dict[str, str]], writer: csv.DictWriter
) -> None:
    for result in results:
        result = {SYMBOL_TO_WORD[key]: value for key, value in result.items()}
        result.update({"ticker": ticker})
        writer.writerow(result)


@click.command()
@click.option(
    "--file-path",
    "-p",
    default="data/stocks_data.csv",
    help="The file path to write the data to",
    type=str,
)
@click.option(
    "--date-delta", "-r", default=7, help="The number of days to go back", type=int
)
@click.option(
    "--multiplier", "-m", default=1, help="The multiplier for the range", type=int
)
@click.option("--ticker-range", "-t", help="The range of tickers to process", type=str)
def write_stocks_data_to_csv(
    file_path: str, date_delta: int, multiplier: int, ticker_range: str
) -> None:
    tickers = get_tickers("data/constituents.csv")
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=date_delta)).strftime("%Y-%m-%d")
    if ticker_range is not None:
        _ticker_range = list(map(int, ticker_range.split(",")))
        file_path = (
            f"{file_path.split('.csv')[0]}_{_ticker_range[0]}_{_ticker_range[1]}.csv"
        )
    else:
        _ticker_range = [1, len(tickers)]
    fieldnames = list(SYMBOL_TO_WORD.values())
    fieldnames.append("ticker")
    with open(file_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        for i in tqdm(
            range(_ticker_range[0] - 1, _ticker_range[1]), desc="Processing tickers"
        ):
            results = get_stonks(
                ticker=tickers[i],
                date_range=(start_date, end_date),
                multiplier=multiplier,
            )
            _write_stocks_data_to_csv(tickers[i], results, writer)
            logging.info(f"Processed ticker: {tickers[i]}")


if __name__ == "__main__":
    write_stocks_data_to_csv()
