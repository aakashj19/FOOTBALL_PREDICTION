
import pandas as pd
import os

file_path = "epl_final.csv"

if os.path.exists(file_path):
    print("Found file:", file_path)

    try:
        df = pd.read_csv(file_path)

        print("\nDataset shape (rows, columns):")
        print(df.shape)

        print("\nColumn names:")
        print(df.columns.tolist())

        print("\nFirst 5 rows:")
        print(df.head())

    except Exception as e:
        print("Could not read the file:", e)

else:
    print("File not found:", file_path)
print("\nSeasons available:")
print(sorted(df["Season"].dropna().unique()))

print("\nEarliest match:")
print(df["MatchDate"].min())

print("\nLatest match:")
print(df["MatchDate"].max())

print("\nNumber of unique clubs:")
teams = set(df["HomeTeam"].dropna()) | set(df["AwayTeam"].dropna())
print(len(teams))

print("\nClub names:")
print(sorted(teams))
print("\nMatch result counts:")
print(df["FullTimeResult"].value_counts(dropna=False))

print("\nMissing values in important columns:")
print(df[["MatchDate", "HomeTeam", "AwayTeam",
          "FullTimeHomeGoals", "FullTimeAwayGoals",
          "FullTimeResult"]].isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())