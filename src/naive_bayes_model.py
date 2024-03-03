from typing import Dict

import click
import pandas as pd

# Load data from the Excel file
data = pd.read_excel("./data/NB/NB_data.xlsx")
data = data.map(lambda x: x.lower() if isinstance(x, str) else x)


class NaiveBayes:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.outcome_probs = {}
        self.conditional_probs = {}

    def train(self) -> None:
        total_instances = len(self.data)
        total_counts = {"up": 0, "down": 0, "no change": 0}

        # calculate probability of UP, DOWN, and NO CHANGE [P('UP'/'DOWN'/'NO CHANGE')]
        for instance in self.data["Market_Change"]:
            total_counts[instance] += 1

        for outcome in total_counts:
            self.outcome_probs[outcome] = total_counts[outcome] / total_instances

        # calculate conditional probabilities (P('attribute')|'UP' or 'DOWN' or 'NO CHANGE')
        for attribute in ["phase", "Temp", "Weather", "Tomato_Change"]:
            for value in set(self.data[attribute]):
                for outcome in total_counts:
                    key = f"{attribute}={value}|Market_Change={outcome}"
                    count = len(
                        self.data[
                            (self.data[attribute] == value)
                            & (self.data["Market_Change"] == outcome)
                        ]
                    )
                    self.conditional_probs[key] = count / total_counts[outcome]

    def predict(self, new_instance) -> Dict[str, float]:
        probabilities: Dict[str, float] = {}
        try:
            for outcome in self.outcome_probs:
                probability = self.outcome_probs[outcome]

                for attribute in ["phase", "Temp", "Weather", "Tomato_Change"]:
                    value = new_instance[attribute]
                    probability *= self.conditional_probs[
                        f"{attribute}={value}|Market_Change={outcome}"
                    ]

                probabilities[outcome] = probability

            total_probability = sum(probabilities.values())

            if total_probability == 0:
                raise ValueError("Probability calculation error")

            probabilities = {k: v / total_probability for k, v in probabilities.items()}

            return probabilities

        except Exception as e:
            print("Error:", str(e))
            return "error"


@click.command()
@click.option(
    "--phase",
    help="Current Lunar Phase: new, first quarter, last quarter, full, waning gibbous, \
              waning crescent, waxing gibbous, waxing crescent",
    type=str,
    required=True,
)
@click.option(
    "--temp",
    help="Current Temp on Wall Street: hot, cold, mild",
    type=str,
    required=True,
)
@click.option(
    "--weather",
    help="Current Weather on Wall Street: clear, misty, rainy",
    type=str,
    required=True,
)
@click.option(
    "--tomato",
    help="Tomato Price Change Today: up, down, no change",
    type=str,
    required=True,
)
def main(phase: str, temp: str, weather: str, tomato: str) -> None:
    new_instance = {
        "phase": phase,
        "Temp": temp,
        "Weather": weather,
        "Tomato_Change": tomato,
    }

    bayes = NaiveBayes(data)
    bayes.train()

    probability = bayes.predict(new_instance)

    input_description = ", ".join(
        f"{attribute}={value}" for attribute, value in new_instance.items()
    )

    for outcome, prob in probability.items():
        print(f"P(Market_Change={outcome}|{input_description}) = {round(prob, 3)}")


if __name__ == "__main__":
    # main()

    updated_bayes = NaiveBayes(data)
    updated_bayes.train()

    # Test the model on the new dataset (NB_data_2.xlsx)
    nb_data_2 = pd.read_excel("./data/NB/NB_data_2.xlsx")
    nb_data_2 = nb_data_2.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    correct_predictions = 0
    total_instances = len(nb_data_2)

    for _, instance in nb_data_2.iterrows():
        new_instance = {
            "phase": instance["phase"],
            "Temp": instance["Temp"],
            "Weather": instance["Weather"],
            "Tomato_Change": instance["Tomato_Change"],
        }

        probability = updated_bayes.predict(new_instance)
        predicted_outcome = max(probability, key=probability.get)

        if predicted_outcome.lower() == instance["Market_Change"].lower():
            correct_predictions += 1

    accuracy = correct_predictions / total_instances
    print(f"Accuracy on the new dataset (NB_data_2.xlsx): {accuracy:.2%}")
