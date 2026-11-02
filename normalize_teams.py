import pandas as pd

CSV_FILE = "epl_clean.csv"

TEAM_NAME_MAP = {
    "Arsenal FC": "Arsenal",
    "Manchester United FC": "Man United",
    "Manchester City FC": "Man City",
    "Tottenham Hotspur FC": "Tottenham",
    "Newcastle United FC": "Newcastle",
    "Nottingham Forest FC": "Nott'm Forest",
    "Wolverhampton Wanderers FC": "Wolves",
    "West Ham United FC": "West Ham",
    "Brighton & Hove Albion FC": "Brighton",
    "AFC Bournemouth": "Bournemouth",
    "Sunderland AFC": "Sunderland",
    "Leeds United FC": "Leeds",
    "Ipswich Town FC": "Ipswich",
    "Burnley FC": "Burnley",
    "Coventry City FC": "Coventry",
    "Hull City AFC": "Hull",
}


def normalize_team_name(name):
    return TEAM_NAME_MAP.get(name, name)


def main():
    df = pd.read_csv(CSV_FILE)

    df["HomeTeam"] = df["HomeTeam"].apply(normalize_team_name)
    df["AwayTeam"] = df["AwayTeam"].apply(normalize_team_name)

    df = df.drop_duplicates(
        subset=["Date", "HomeTeam", "AwayTeam"],
        keep="last"
    )

    df = df.sort_values("Date")
    df.to_csv(CSV_FILE, index=False)

    teams = sorted(
        set(df["HomeTeam"]).union(set(df["AwayTeam"]))
    )

    print("Dataset normalized.")
    print(f"Total matches: {len(df)}")
    print(f"Unique teams: {len(teams)}")
    print("\nTeams:")
    for team in teams:
        print(team)


if __name__ == "__main__":
    main()