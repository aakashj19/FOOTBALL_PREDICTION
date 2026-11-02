
import pandas as pd

df = pd.read_csv("epl_clean.csv")
df["Date"] = pd.to_datetime(df["Date"])

print("Total matches:", len(df))
print("Missing values:")
print(df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].isna().sum())

print("\nInvalid outcomes:", (~df["FTR"].isin(["H", "D", "A"])).sum())
print("Missing scores:", df[["FTHG", "FTAG"]].isna().any(axis=1).sum())
print("Duplicate fixtures:", df.duplicated(
    subset=["Date", "HomeTeam", "AwayTeam"]
).sum())

print("\nOutcome distribution:")
print(df["FTR"].value_counts())

print("\nLatest date:", df["Date"].max())
print("\nValidation complete.")
