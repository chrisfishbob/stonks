import csv
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import click
import requests
from tqdm import tqdm

API_KEY = "lK1jwHrVg47249kaPqBQ_Vfv__bCHBZz"

symbol_to_word = {
    "c": "close",
    "h": "high",
    "l": "low",
    "o": "open",
    "v": "volume",
    "t": "timestamp",
    "vw": "volume_weighted_average_price",
    "n": "number_of_trades",
}


def get_tickers(file_path: str) -> List[str]:
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        return [row[0] for row in reader]


def call_polygon_api(ticker: str, date_range: Tuple) -> List[Dict[str, str]]:
    base_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/hour/"
    url = (
        f"{base_url}{date_range[0]}/{date_range[1]}?adjusted=true&sort=asc&limit=20000"
        f"&apiKey={API_KEY}"
    )
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(
            f"Failed to call the Polygon API. Status code: {response.status_code}."
            f"Response: {response.text}"
        )
    return response.json()["results"]


@click.command()
@click.option(
    "--file-path",
    "-p",
    default="data/stocks_data.csv",
    help="The file path to write the data to",
    type=str,
)
@click.option(
    "--date-range", "-r", default=7, help="The number of days to go back", type=int
)
def write_stocks_data_to_csv(file_path: str, date_range: int) -> None:
    tickers = get_tickers("data/constituents.csv")
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=date_range)).strftime("%Y-%m-%d")
    with open(file_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "volume",
                "volume_weighted_average_price",
                "open",
                "close",
                "high",
                "low",
                "timestamp",
                "number_of_trades",
                "ticker",
            ],
        )
        writer.writeheader()
        for ticker in tqdm(tickers, desc="Processing tickers"):
            results = call_polygon_api(ticker, (start_date, end_date))
            start_time = time.time()
            for result in results:
                result = {symbol_to_word[key]: value for key, value in result.items()}
                result.update({"ticker": ticker})
                writer.writerow(result)
            end_time = time.time()
            time.sleep(12 - (end_time - start_time))


if __name__ == "__main__":
    write_stocks_data_to_csv()
