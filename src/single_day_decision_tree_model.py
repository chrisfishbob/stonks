import click
import pandas as pd
from typing import Dict, Tuple
from numpy import ndarray
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def train_decision_tree_models(
    stock_data_file,
) -> Tuple[Dict[str, DecisionTreeClassifier], Dict[str, ndarray]]:
    data = pd.read_csv(stock_data_file)
    click.secho(
        f"Loaded {len(data)} lines of data from {stock_data_file}", fg="blue", bold=True
    )

    grouped_data = data.groupby("ticker")

    ticker_models = {}
    ticker_predictions = {}
    ticker_accuracy = {}

    with click.progressbar(grouped_data, label="Training ticker models") as bar:
        for ticker, ticker_df in bar:
            ticker_df["next_close"] = ticker_df["close"].shift(-1)
            ticker_df["close_change"] = (ticker_df["next_close"] > ticker_df["close"]).astype(int)

            x = ticker_df[
                [
                    "open",  # Today's open
                    "high",  # Today's high
                    "low",  # Today's low
                    "close",  # Today's close
                    "volume",  # Today's volume
                    "close_change",  # 1 if tomorrow's close > today's close else 0
                ]
            ].dropna()  # Drop rows with NaN values

            y = x["close_change"].dropna()  # 1 if tomorrow's close > today's close else 0

            x.drop(columns=["close_change"], inplace=True)  # Drop close_change from features

            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42
            )

            classifier = DecisionTreeClassifier()

            classifier.fit(x_train, y_train)

            predictions = classifier.predict(x_test)

            accuracy = accuracy_score(y_test, predictions)

            # Store the model and predictions
            ticker_models[str(ticker)] = classifier
            ticker_predictions[str(ticker)] = predictions
            ticker_accuracy[str(ticker)] = accuracy

    return ticker_models, ticker_predictions


def predict(
    ticker: str, features: DataFrame, models: Dict[str, DecisionTreeClassifier]
) -> str:
    model = models[ticker]
    prediction = model.predict(features)
    if prediction[0] == 1:
        return "Tomorrow's close price is predicted to be higher than today's close price."
    else:
        return "Tomorrow's close price is predicted to be lower than or equal to today's close price."


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
    help="The ticker to predict whether tomorrow's close price will be higher or lower than today's close price for",
    type=str,
    required=True,
)
@click.option(
    "--open",
    "-o",
    help="Today's open price",
    type=float,
    required=True,
)
@click.option(
    "--high",
    "-hi",
    help="Today's high price",
    type=float,
    required=True,
)
@click.option(
    "--low",
    "-l",
    help="Today's low price",
    type=float,
    required=True,
)
@click.option(
    "--close",
    "-c",
    help="Today's close price",
    type=float,
    required=True,
)
@click.option(
    "--volume",
    "-v",
    help="Today's volume",
    type=int,
    required=True,
)
def single_day_decision_tree_model(
    file_path: str,
    ticker: str,
    open: float,
    high: float,
    low: float,
    close: float,
    volume: int,
) -> None:
    models, _ = train_decision_tree_models(file_path)

    features = pd.DataFrame(
        {
            "open": open,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=[0],
    )
    click.secho(
        f"\nPredicting whether tomorrow's close price will be higher or lower than today's close price for {ticker} using today's prices:\n{features}"
    )
    prediction = predict(ticker, features, models)
    click.secho(prediction, fg="green", bold=True)


if __name__ == "__main__":
    single_day_decision_tree_model()
