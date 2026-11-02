
import pandas as pd

# Load historical Premier League matches
df = pd.read_csv("epl_final.csv")

# Convert match dates into proper date values
df["MatchDate"] = pd.to_datetime(df["MatchDate"], errors="coerce")

# Sort matches from oldest to newest
df = df.sort_values("MatchDate").reset_index(drop=True)

# Keep only the columns needed for our first model
df = df[
    [
        "MatchDate",
        "Season",
        "HomeTeam",
        "AwayTeam",
        "FullTimeHomeGoals",
        "FullTimeAwayGoals",
        "FullTimeResult",
    ]
].copy()

print("Prepared dataset shape:", df.shape)
print("\nFirst 5 matches:")
print(df.head())

print("\nOutcome counts:")
print(df["FullTimeResult"].value_counts())

import pandas as pd

# Load historical Premier League matches
df = pd.read_csv("epl_final.csv")

# Convert dates into proper date format
df["MatchDate"] = pd.to_datetime(df["MatchDate"], errors="coerce")

# Sort matches from oldest to newest
df = df.sort_values("MatchDate").reset_index(drop=True)

# Keep only the columns needed for prediction
df = df[
    [
        "MatchDate",
        "Season",
        "HomeTeam",
        "AwayTeam",
        "FullTimeHomeGoals",
        "FullTimeAwayGoals",
        "FullTimeResult",
    ]
].copy()

print("Prepared dataset shape:", df.shape)
print("\nFirst 5 matches:")
print(df.head())

print("\nMatch outcome counts:")
print(df["FullTimeResult"].value_counts())
