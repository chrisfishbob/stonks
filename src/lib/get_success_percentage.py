from datetime import datetime, timedelta

import click
import pandas as pd

from multi_day_model import get_actual, predict, train
from utils import get_tickers


def _get_success_percentage(
    stock_file_path: str, tickers_file_path: str, date: str, days_ahead: int
) -> float:
    tickers = get_tickers(tickers_file_path)
    models, _, _ = train(stock_file_path, days_ahead)
    correct = 0
    predicted = 0

    data = pd.read_csv(stock_file_path)

    with click.progressbar(
        tickers, label=f"Processing tickers for {date}:"
    ) as progress_bar:
        for ticker in progress_bar:
            try:
                features, prediction = predict(data, models, ticker, date)
                actual = get_actual(data, features, ticker, date, days_ahead)
                if prediction == actual:
                    correct += 1
                predicted += 1

                success_percentage = correct / predicted * 100
                progress_bar.label = (
                    f"Processing tickers for {date}: {success_percentage:.2f}%"
                )
            except Exception as e:
                continue

    if predicted == 0:
        return None
    return correct / predicted


@click.command()
@click.option(
    "--stock-file-path",
    "-sp",
    default="data/stock_data.csv",
    help="The file path to read the stock data from",
    type=str,
)
@click.option(
    "--tickers-file-path",
    "-tp",
    default="data/constituents.csv",
    help="The file path to read the tickers from",
)
@click.option("--date", "-d", help="The start date", type=str)
@click.option(
    "--days-ahead",
    "-da",
    help="The number of days ahead to predict",
    type=int,
    default=1,
)
def get_success_percentage(
    stock_file_path: str,
    tickers_file_path: str,
    date: str,
    days_ahead: int,
) -> None:
    success_percentage = _get_success_percentage(
        stock_file_path, tickers_file_path, date, days_ahead
    )

    print(f"Success Percentage: {success_percentage:.2f}")


if __name__ == "__main__":
    for year in range(2023, 2025):
        for month in range(1, 13):
            # Find the first day of the month
            first_day = datetime(year, month, 1)

            # Find the first Monday of the month
            date = datetime(year, month, 1) + timedelta(
                days=(2 - first_day.weekday()) % 7
            )

            for i in [
                1,
                2,
                5,
                6,
                7,
                8,
                9,
                12,
                13,
                14,
                15,
                16,
                19,
                20,
                21,
                22,
                23,
                26,
                27,
                28,
                29,
                30,
                31,
            ]:
                percentage = _get_success_percentage(
                    "data/stock_data.csv", "data/constituents.csv", date, i
                )
                if percentage:
                    with open("success_percentage.csv", "a") as file:
                        later_date = date + timedelta(days=i)
                        file.write(f"{date},{later_date},{i},{percentage}\n")
