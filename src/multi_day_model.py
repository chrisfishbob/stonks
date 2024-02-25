from typing import Dict, Tuple

import click
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def train_decision_tree_models(
    stock_data_file,
) -> Tuple[Dict[str, DecisionTreeRegressor], Dict[str, ndarray], dict[str, object]]:
    df = pd.read_csv(stock_data_file)

    # Step 1: Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Step 2: Extract date from timestamp
    df['date'] = df['timestamp'].dt.date

    # Step 3: Group by ticker and date
    grouped = df.groupby(['ticker', 'date'])

    # Step 4: Keep only the first entry of each group (i.e., each ticker and each day)
    filtered_df = grouped.first().reset_index()

    # Step 5: Add an extra column called "later_close" with close price 3 days later for each ticker
    filtered_df['later_close'] = filtered_df.groupby('ticker')['close'].shift(-3)

    # Step 6: When our data cuts off, there will be a few rows without a later_close date. Drop those rows
    filtered_df.dropna(subset=["later_close"], inplace=True)

    # Step 7: Group by the ticket
    filtered_df = filtered_df.groupby("ticker")

    ticker_models = {}
    ticker_predictions = {}
    ticker_mse = {}

    with click.progressbar(filtered_df, label="Training ticker models") as bar:
        for ticker, ticker_df in bar:
            x = ticker_df[
                [
                    "close",
                    "high",
                    "low",
                    "open",
                    "volume",
                    "volume_weighted_average_price",
                    "number_of_trades",
                ]
            ]
            y = ticker_df["later_close"]

            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42
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


def predict(
    ticker: str, features: DataFrame, models: Dict[str, DecisionTreeRegressor]
) -> float:
    model = models[ticker]
    prediction = model.predict(features)

    return prediction[0]


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
def three_day_decision_tree_model(
    file_path: str,
    ticker: str,
    high: float,
    low: float,
    open: float,
    volume: int,
    vwap: float,
    num_trades: int,
    close_price: float
) -> None:
    models, _, mse = train_decision_tree_models(file_path)

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
        f"\nPredicted close price for {ticker}: {prediction}", fg="green", bold=True
    )
    click.secho(
        f"\nMean squared error: {mse[ticker]}", fg="yellow", bold=True
    )


if __name__ == "__main__":
    three_day_decision_tree_model()