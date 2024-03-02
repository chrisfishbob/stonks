import click
import pandas as pd
from typing import Dict, Tuple
from numpy import ndarray
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from utils import call_polygon_api, SYMBOL_TO_WORD, API_KEYS, API_KEY_COUNTER, get_tickers


def train(
    stock_data_file,
) -> Tuple[Dict[str, DecisionTreeClassifier], Dict[str, ndarray]]:
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
            ticker_df["next_close"] = ticker_df["close"].shift(-1)
            ticker_df["close_change"] = (
                ticker_df["next_close"] > ticker_df["close"]
            ).astype(int)

            x = ticker_df[
                [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_change",
                ]
            ].dropna()

            y = x["close_change"].dropna()

            x.drop(columns=["close_change"], inplace=True)

            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42
            )

            classifier = DecisionTreeClassifier()

            classifier.fit(x_train, y_train)

            predictions = classifier.predict(x_test)

            mse = mean_squared_error(y_test, predictions)

            ticker_models[str(ticker)] = classifier
            ticker_predictions[str(ticker)] = predictions
            ticker_mse[str(ticker)] = mse

    return ticker_models, ticker_predictions, ticker_mse


def predict(
    models: Dict[str, DecisionTreeClassifier],
    ticker: str,
    date: str,
) -> None:
    global API_KEY_COUNTER
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{date}/{date}?apiKey={API_KEYS[API_KEY_COUNTER % len(API_KEYS)]}"
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


def get_actual(features: pd.DataFrame, ticker: str, date: str) -> None:
    global API_KEY_COUNTER
    
    today_price = features["close"].values[0]

    tomorrow = (pd.to_datetime(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{tomorrow}/{tomorrow}?apiKey={API_KEYS[API_KEY_COUNTER % len(API_KEYS)]}"
    API_KEY_COUNTER += 1

    response = call_polygon_api(url)
    tmrw_price = next(iter(response["results"]))["c"]

    return int(tmrw_price > today_price)


@click.command()
@click.option(
    "--file-path",
    "-p",
    default="data/stock_data.csv",
    help="The file path to read the data from",
    type=str,
)
@click.option("--ticker", "-t", help="The ticker to predict", type=str)
@click.option("--date", "-d", help="The date to predict", type=str)
def single_day_decision_tree_model(file_path: str, ticker: str, date: str) -> None:
    models, _, mse = train(file_path)
    features, prediction = predict(models, ticker, date)
    actual = get_actual(features, ticker, date)
    
    click.secho(f"Predicted: {prediction}", fg="green", bold=True)
    click.secho(f"Mean Squared Error: {mse[ticker]}", fg="green", bold=True)
    #click.secho(f"Actual: {actual}", fg="green", bold=True)

if __name__ == "__main__":
    single_day_decision_tree_model()
