from typing import Dict, Tuple

import click
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from sklearn.metrics import mean_squared_error  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.tree import DecisionTreeRegressor  # type: ignore


def train_decision_tree_models(
    stock_data_file,
) -> Tuple[Dict[str, DecisionTreeRegressor], Dict[str, ndarray]]:
    data = pd.read_csv(stock_data_file)
    click.secho(
        f"Loaded {len(data)} lines of data from {stock_data_file}", fg="blue", bold=True
    )

    grouped_data = data.groupby("ticker")

    ticker_models = {}
    ticker_predictions = {}
    ticker_mse = {}

    with click.progressbar(grouped_data, label="Training ticker models") as bar:
        for ticker, ticker_df in bar:
            x = ticker_df[
                [
                    "high",
                    "low",
                    "open",
                    "volume",
                    "volume_weighted_average_price",
                    "number_of_trades",
                ]
            ]
            y = ticker_df["close"]  # Assuming we want to predict the 'close' price

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

    return ticker_models, ticker_predictions


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
def single_day_decision_tree_model(
    file_path: str,
    ticker: str,
    high: float,
    low: float,
    open: float,
    volume: int,
    vwap: float,
    num_trades: int,
) -> None:
    models, _ = train_decision_tree_models(file_path)

    features = pd.DataFrame(
        {
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


if __name__ == "__main__":
    single_day_decision_tree_model()
