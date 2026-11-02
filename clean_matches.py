
import pandas as pd

# Load the combined dataset
df = pd.read_csv("epl_recent_combined.csv")

# Keep only the columns needed for our first model
df = df[
    ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
].copy()

# Convert dates and scores to the correct types
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")

# Remove rows with missing information
df = df.dropna(
    subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
)

# Keep only valid completed match outcomes
df = df[df["FTR"].isin(["H", "D", "A"])]

# Remove duplicate fixtures, if any
df = df.drop_duplicates(
    subset=["Date", "HomeTeam", "AwayTeam"]
)

# Sort chronologically
df = df.sort_values("Date").reset_index(drop=True)

# Save the cleaned dataset
df.to_csv("epl_clean.csv", index=False)

print("Cleaned matches:", len(df))
print("\nOutcome counts:")
print(df["FTR"].value_counts())

print("\nDate range:")
print(df["Date"].min(), "to", df["Date"].max())

print("\nMissing values:")
print(df.isna().sum())

print("\nSaved as epl_clean.csv")
