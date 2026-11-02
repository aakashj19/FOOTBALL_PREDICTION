
import pandas as pd

files = [
    "epl_2425.csv",
    "epl_2526.csv",
    "epl_2627.csv"
]

# Load each season
season_data = []

for file in files:
    df = pd.read_csv(file)
    season_data.append(df)
    print(f"{file}: {len(df)} matches")

# Combine all seasons
combined = pd.concat(season_data, ignore_index=True, sort=False)

# Convert dates (these files use day/month/year)
combined["Date"] = pd.to_datetime(
    combined["Date"],
    dayfirst=True,
    errors="coerce"
)

# Sort matches by date
combined = combined.sort_values("Date").reset_index(drop=True)

# Save combined data
combined.to_csv("epl_recent_combined.csv", index=False)

print("\nTotal matches:", len(combined))
print("Unique home teams:", combined["HomeTeam"].nunique())
print("Missing dates:", combined["Date"].isna().sum())
print("Duplicate matches:", combined.duplicated(
    subset=["Date", "HomeTeam", "AwayTeam"]
).sum())

print("\nDate range:")
print(combined["Date"].min(), "to", combined["Date"].max())

print("\nSaved as epl_recent_combined.csv")
