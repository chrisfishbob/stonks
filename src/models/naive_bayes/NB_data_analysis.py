import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from naive_bayes_model import NaiveBayes

data = pd.read_excel("./data/NB/NB_data.xlsx")
data = data.map(lambda x: x.lower() if isinstance(x, str) else x)

# Load your spreadsheet into a pandas DataFrame
# Replace 'your_data.csv' with your actual file path or URL
df = pd.read_excel("./data/NB/NB_data.xlsx")

# Check the first few rows of the DataFrame
print(df.head())
# df = df.drop('Date', axis=1)

# Check for duplicate rows
duplicates = df[df.duplicated()]
print(f"Duplicate Rows: {duplicates.shape[0]}")

# Remove duplicate rows
# df = df.drop_duplicates()

# Check the balance of the target variable
target_balance = df["Market_Change"].value_counts()
print("Target Variable Balance:")
print(target_balance)

# Descriptive statistics for categorical columns
print(df.describe(include="object"))

# One-hot encode categorical variables
df_encoded = pd.get_dummies(
    df, columns=["phase", "Temp", "Weather", "Tomato_Change"], drop_first=True
)

# Check the first few rows of the encoded DataFrame
print(df_encoded.head())

# Bar plots for categorical columns against 'market_change'
categorical_columns = ["phase", "Temp", "Weather", "Tomato_Change"]
for column in categorical_columns:
    sns.countplot(x=column, hue="Market_Change", data=df)
    plt.title(f"Count plot for {column} vs market_change")
    plt.show()

# Chi-square test for independence between categorical variables and 'market_change'
from scipy.stats import chi2_contingency

for column in categorical_columns:
    contingency_table = pd.crosstab(df[column], df["Market_Change"])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    print(f"Chi-square test for independence between {column} and market_change:")
    print(f"Chi2 Statistic: {chi2}, p-value: {p}\n")

updated_bayes = NaiveBayes(data)
updated_bayes.train()

# Test the model on the new dataset (NB_data_2.xlsx)
nb_data_2 = pd.read_excel("./data/NB/NB_data_2.xlsx")
nb_data_2 = nb_data_2.map(lambda x: x.lower() if isinstance(x, str) else x)

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
