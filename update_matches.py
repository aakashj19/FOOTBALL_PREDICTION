import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("FOOTBALL_API_KEY")
API_URL = "https://api.football-data.org/v4/competitions/PL/matches"

CSV_FILE = "epl_clean.csv"

# Convert API team names to the names already used in your dataset
TEAM_NAME_MAP = {
    "Arsenal FC": "Arsenal",
    "Manchester United FC": "Man United",
    "Manchester City FC": "Man City",
    "Tottenham Hotspur FC": "Tottenham",
    "Newcastle United FC": "Newcastle",
    "Nottingham Forest FC": "Nott'm Forest",
    "Wolverhampton Wanderers FC": "Wolves",
    "West Ham United FC": "West Ham",
    "Leicester City FC": "Leicester",
    "Brighton & Hove Albion FC": "Brighton",
    "AFC Bournemouth": "Bournemouth",
    "Sunderland AFC": "Sunderland",
    "Leeds United FC": "Leeds",
    "Ipswich Town FC": "Ipswich",
    "Burnley FC": "Burnley",
    "Coventry City FC": "Coventry",
    "Hull City AFC": "Hull",
    "Everton FC": "Everton",
    "Liverpool FC": "Liverpool",
    "Chelsea FC": "Chelsea",
    "Aston Villa FC": "Aston Villa",
    "Fulham FC": "Fulham",
    "Brentford FC": "Brentford",
    "Crystal Palace FC": "Crystal Palace",
}


def normalize_team_name(name):
    """Convert a team name to the canonical name used in the dataset."""
    return TEAM_NAME_MAP.get(name, name)


def fetch_matches():
    if not API_KEY:
        raise ValueError(
            "FOOTBALL_API_KEY not found. Check your .env file."
        )

    headers = {"X-Auth-Token": API_KEY}

    response = requests.get(
        API_URL,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    return response.json().get("matches", [])


def main():
    print("Fetching Premier League matches...")

    matches = fetch_matches()
    new_rows = []

    for match in matches:
        # Only include completed matches
        if match.get("status") != "FINISHED":
            continue

        home_team = normalize_team_name(
            match["homeTeam"]["name"]
        )

        away_team = normalize_team_name(
            match["awayTeam"]["name"]
        )

        home_goals = match["score"]["fullTime"]["home"]
        away_goals = match["score"]["fullTime"]["away"]

        if home_goals is None or away_goals is None:
            continue

        if home_goals > away_goals:
            result = "H"
        elif home_goals < away_goals:
            result = "A"
        else:
            result = "D"

        new_rows.append({
            "Date": match["utcDate"][:10],
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "FTHG": home_goals,
            "FTAG": away_goals,
            "FTR": result,
        })

    if not new_rows:
        print("No completed matches returned by the API.")
        return

    new_df = pd.DataFrame(new_rows)

    if os.path.exists(CSV_FILE):
        old_df = pd.read_csv(CSV_FILE)

        # Normalize existing names too
        old_df["HomeTeam"] = old_df["HomeTeam"].apply(
            normalize_team_name
        )
        old_df["AwayTeam"] = old_df["AwayTeam"].apply(
            normalize_team_name
        )

        combined = pd.concat(
            [old_df, new_df],
            ignore_index=True
        )
    else:
        combined = new_df

    # Keep one record per match
    combined["Date"] = pd.to_datetime(
        combined["Date"],
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    combined = combined.drop_duplicates(
        subset=["Date", "HomeTeam", "AwayTeam"],
        keep="last"
    )

    combined = combined.sort_values("Date")
    combined.to_csv(CSV_FILE, index=False)

    print(f"API matches received: {len(new_df)}")
    print(f"Total matches in dataset: {len(combined)}")
    print("Team names normalized and dataset saved.")


if __name__ == "__main__":
    main()