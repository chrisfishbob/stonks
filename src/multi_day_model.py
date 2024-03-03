from typing import Dict, Tuple

import click
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from utils import call_polygon_api, SYMBOL_TO_WORD, API_KEYS, API_KEY_COUNTER, get_tickers
from retry import retry


def train(
    stock_data_file,
    days_ahead: int = 3,
) -> Tuple[Dict[str, DecisionTreeRegressor], Dict[str, ndarray], dict[str, object]]:
    df = pd.read_csv(stock_data_file)

    # Step 1: Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Step 2: Extract date from timestamp
    df["date"] = df["timestamp"].dt.date

    # Step 3: Group by ticker and date
    grouped = df.groupby(["ticker", "date"])

    # Step 4: Keep only the first entry of each group (i.e., each ticker and each day)
    filtered_df = grouped.first().reset_index()

    # Step 5: Add an extra column called "later_close" with close price `days_ahead` later for each ticker
    filtered_df["later_close"] = filtered_df.groupby("ticker")["close"].shift(-days_ahead)

    # Step 6: When our data cuts off, there will be a few rows without a later_close date. Drop those rows
    filtered_df.dropna(subset=["later_close"], inplace=True)

    # Step 7: Create a column indicating price change direction (up or down)
    filtered_df["price_change"] = filtered_df.apply(lambda row: 1 if row["later_close"] > row["close"] else 0, axis=1)

    # Step 8: Drop the "later_close" column
    filtered_df.drop(columns=["later_close"], inplace=True)

    # Step 9: Group by the ticket
    grouped_df = filtered_df.groupby("ticker")

    ticker_models = {}
    ticker_predictions = {}
    ticker_mse = {}

    with click.progressbar(grouped_df, label="Training ticker models") as bar:
        for ticker, ticker_df in bar:
            x = ticker_df[
                [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            ]
            y = ticker_df["price_change"]

            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.3, random_state=42
            )

            regressor = DecisionTreeRegressor()

            regressor.fit(x_train, y_train)

            predictions = regressor.predict(x_test)

            mse = mean_squared_error(y_test, predictions)

            # Store the model and predictions
            ticker_models[str(ticker)] = regressor
            ticker_predictions[str(ticker)] = predictions
            ticker_mse[str(ticker)] = mse

    return ticker_models, ticker_predictions, ticker_mse


@retry(tries=3)
def predict(
    models: Dict[str, DecisionTreeClassifier],
    ticker: str,
    date: str,
):
    global API_KEY_COUNTER
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/3/day/{date}/{date}?apiKey={API_KEYS[API_KEY_COUNTER % len(API_KEYS)]}"
    API_KEY_COUNTER += 1
    
    response = call_polygon_api(url)
    features = pd.DataFrame(
        {
            SYMBOL_TO_WORD[k]: next(iter(response["results"]))[k]
            for k in ["o", "h", "l", "c", "v"]
        },
        index=[0],
    )
    prediction = next(iter(models[ticker].predict(features)))
    return features, prediction


@retry(tries=3)
def get_actual(features: pd.DataFrame, ticker: str, date: str):
    global API_KEY_COUNTER
    
    today_price = features["close"].values[0]

    tomorrow = (pd.to_datetime(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/3/day/{tomorrow}/{tomorrow}?apiKey={API_KEYS[API_KEY_COUNTER % len(API_KEYS)]}"
    API_KEY_COUNTER += 1

    response = call_polygon_api(url)
    tmrw_price = next(iter(response["results"]))["c"]

    return int(tmrw_price > today_price)


@click.command()
@click.option(
    "--file-path",
    "-f",
    default="data/stock_data.csv",
    help="The directory containing the training data",
    type=str,
    required=True,
)
@click.option(
    "--ticker",
    "-t",
    help="The ticker to predict the close price for",
    type=str,
    required=True,
)
@click.option(
    "--high",
    help="The high price for the day",
    type=float,
    required=True,
)
@click.option(
    "--low",
    help="The low price for the day",
    type=float,
    required=True,
)
@click.option(
    "--open",
    help="The open price for the day",
    type=float,
    required=True,
)
@click.option(
    "--volume",
    help="The volume for the day",
    type=int,
    required=True,
)
@click.option(
    "--vwap",
    help="The volume weighted average price for the day",
    type=float,
    required=True,
)
@click.option(
    "--num-trades",
    help="The number of trades for the day",
    type=int,
    required=True,
)
@click.option(
    "--close-price",
    help="The number of trades for the day",
    type=float,
    required=True,
)
@click.option(
    "--days-ahead",
    help="The number of trades for the day",
    type=int,
    required=True,
)
def three_day_decision_tree_model(
    file_path: str,
    ticker: str,
    high: float,
    low: float,
    open: float,
    volume: int,
    vwap: float,
    num_trades: int,
    close_price: float,
    days_ahead: int,
) -> None:
    models, _, mse = train(file_path, days_ahead)

    features = pd.DataFrame(
        {
            "close": close_price,
            "high": high,
            "low": low,
            "open": open,
            "volume": volume,
            "volume_weighted_average_price": vwap,
            "number_of_trades": num_trades,
        },
        index=[0],
    )
    click.secho(
        f"\nPredicting close price for {ticker} using the following: \n{features}"
    )
    prediction = predict(ticker, features, models)
    click.secho(
        f"\nThe model predicts that {ticker} will go {prediction} in {days_ahead} day(s)", fg="green", bold=True
    )
    click.secho(f"\nMean squared error: {mse[ticker]}", fg="yellow", bold=True)


if __name__ == "__main__":
    three_day_decision_tree_model()
