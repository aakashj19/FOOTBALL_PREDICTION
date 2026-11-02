
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    log_loss
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


# 1. Load and sort match data
df = pd.read_csv("epl_features.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)


# 2. Use all 15 pre-match features
feature_columns = [
    "HomeRecentPoints",
    "AwayRecentPoints",
    "HomeGoalsScored",
    "HomeGoalsConceded",
    "AwayGoalsScored",
    "AwayGoalsConceded",
    "HomeVenuePoints",
    "AwayVenuePoints",
    "HomeElo",
    "AwayElo",

    # New difference features
    "RecentPointsDiff",
    "GoalsScoredDiff",
    "GoalsConcededDiff",
    "VenuePointsDiff",
    "EloDiff"
]

# Check that all required features exist
missing_columns = [
    column for column in feature_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing feature columns: {missing_columns}. "
        "Run recent_form.py first."
    )

X = df[feature_columns]
y = df["FTR"]


# 3. Chronological train/test split
unique_dates = sorted(df["Date"].unique())
split_date = unique_dates[int(len(unique_dates) * 0.8)]

train_mask = df["Date"] < split_date
test_mask = df["Date"] >= split_date

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]
y_train = y.loc[train_mask]
y_test = y.loc[test_mask]


# Give underrepresented outcomes more influence during training
sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)


# 4. Models to compare
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        random_state=42
    )
}


# 5. Train and evaluate each model
results = []

print("Training matches:", len(X_train))
print("Test matches:", len(X_test))
print("Test period starts:", split_date.date())

for name, model in models.items():

    if name == "Gradient Boosting":
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights
        )
    else:
        model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )
    loss = log_loss(
        y_test,
        probabilities,
        labels=model.classes_
    )

    report = classification_report(
        y_test,
        predictions,
        labels=["H", "D", "A"],
        target_names=["Home Win", "Draw", "Away Win"],
        output_dict=True,
        zero_division=0
    )

    draw_recall = report["Draw"]["recall"]

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Log Loss": loss,
        "Draw Recall": draw_recall
    })

    print("\n" + "=" * 45)
    print(name)
    print("=" * 45)
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print(
        "Balanced accuracy:",
        round(balanced_accuracy * 100, 2),
        "%"
    )
    print("Log loss:", round(loss, 4))
    print("Draw recall:", round(draw_recall * 100, 2), "%")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=["H", "D", "A"],
        target_names=["Home Win", "Draw", "Away Win"],
        zero_division=0
    ))


# 6. Compare models using balanced accuracy
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "Balanced Accuracy",
    ascending=False
)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]


# 7. Retrain selected model on all available matches
all_sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y
)

if best_model_name == "Gradient Boosting":
    best_model.fit(
        X,
        y,
        sample_weight=all_sample_weights
    )
else:
    best_model.fit(X, y)


# 8. Save model and expected feature order
joblib.dump(
    {
        "model": best_model,
        "feature_columns": feature_columns
    },
    "football_model.pkl"
)

print(f"\nBest model: {best_model_name}")
print("Saved as football_model.pkl")
