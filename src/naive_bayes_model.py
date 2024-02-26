import pandas as pd
from typing import Dict

# Load data from the Excel file
data = pd.read_excel('./data/NB/NB_data.xlsx')

class NaiveBayes:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.outcome_probs: Dict[str, float] = {}
        self.conditional_probs: Dict[str, float] = {}

    def train(self) -> None:
        total_instances = len(self.data)
        total_counts = {'UP': 0, 'DOWN': 0, 'NO CHANGE': 0}

        # calculate probability of UP, DOWN, and NO CHANGE [P('UP'/'DOWN'/'NO CHANGE')]
        for instance in self.data['Market_Change']:
            total_counts[instance] += 1

        for outcome in total_counts:
            self.outcome_probs[outcome] = total_counts[outcome] / total_instances

        # calculate conditional probabilities (P('attribute')|'UP' or 'DOWN' or 'NO CHANGE')
        for attribute in ['phase', 'Temp', 'Weather', 'Tomato_Change']:
            for value in set(self.data[attribute]):
                for outcome in total_counts:
                    key = f"{attribute}={value}|Market_Change={outcome}"
                    count = len(self.data[(self.data[attribute] == value) & (self.data['Market_Change'] == outcome)])
                    self.conditional_probs[key] = count / total_counts[outcome]

    def predict(self, new_instance) -> Dict[str, float]:
        probabilities: Dict[str, float] = {}
        try:
            for outcome in self.outcome_probs:
                probability = self.outcome_probs[outcome]

                for attribute in ['phase', 'Temp', 'Weather', 'Tomato_Change']:
                    value = new_instance[attribute]
                    probability *= self.conditional_probs[f"{attribute}={value}|Market_Change={outcome}"]

                probabilities[outcome] = probability

            total_probability = sum(probabilities.values())

            if total_probability == 0:
                raise ValueError("Probability calculation error")

            probabilities = {k: v / total_probability for k, v in probabilities.items()}

            return probabilities

        except Exception as e:
            print("Error:", str(e))
            return "error"

if __name__ == "__main__":
    bayes = NaiveBayes(data)
    bayes.train()

    phase = input("Enter Phase: ")
    temp = input("Enter Temperature: ")
    weather = input("Enter Weather: ")
    tomato_change = input("Enter Tomato_Change: ")

    new_instance = {'phase': phase, 'Temp': temp, 'Weather': weather, 'Tomato_Change': tomato_change}
    probability = bayes.predict(new_instance)

    input_description = ', '.join(f'{attribute}={value}' for attribute, value in new_instance.items())

    for outcome, prob in probability.items():
        print(f"P(Market_Change={outcome}|{input_description}) = {round(prob, 3)}")