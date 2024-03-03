import click

# from single_day_decision_tree_model import train, predict, get_actual
from multi_day_model import train, predict, get_actual
from utils import get_tickers
import pandas as pd


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
    tickers = get_tickers(tickers_file_path)
    models, _, _ = train(stock_file_path, days_ahead)
    correct = 0
    predicted = 0

    data = pd.read_csv(stock_file_path)

    with click.progressbar(tickers, label="Processing tickers") as progress_bar:
        for ticker in progress_bar:
            features, prediction = predict(data, models, ticker, date)
            actual = get_actual(data, features, ticker, date, days_ahead)
            if prediction == actual:
                correct += 1
            predicted += 1

            success_percentage = correct / predicted * 100
            progress_bar.label = f"Processing tickers: {success_percentage:.2f}%"

    print(f"Success Percentage: {correct / predicted}")


if __name__ == "__main__":
    get_success_percentage()
