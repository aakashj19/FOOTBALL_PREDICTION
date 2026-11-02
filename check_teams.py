import pandas as pd

df = pd.read_csv("epl_clean.csv")

teams = sorted(
    set(df["HomeTeam"].dropna())
    | set(df["AwayTeam"].dropna())
)

print(f"Total unique team names: {len(teams)}")
print("\nAll team names:")
for team in teams:
    print(repr(team))