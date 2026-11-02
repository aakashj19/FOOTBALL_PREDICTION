
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load prepared match features
df = pd.read_csv("epl_features.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

feature_columns = [
    "HomeRecentPoints",
    "AwayRecentPoints",
    "HomeGoalsScored",
    "HomeGoalsConceded",
    "AwayGoalsScored",
    "AwayGoalsConceded"
]

X = df[feature_columns]
y = df["FTR"]

# Train on all currently available historical matches
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X, y)

# Save model and the feature names
joblib.dump(
    {
        "model": model,
        "feature_columns": feature_columns
    },
    "football_model.pkl"
)

print("Model trained on all", len(df), "matches.")
print("Saved as football_model.pkl")
print("Features:", feature_columns)
