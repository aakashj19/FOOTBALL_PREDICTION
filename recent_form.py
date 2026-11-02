
import pandas as pd
import math

# Load and sort historical matches
df = pd.read_csv("epl_clean.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Store each team's previous match results and Elo rating
history = {}
elo = {}
features = []

INITIAL_ELO = 1500
ELO_K = 32


def get_elo(team):
    return elo.get(team, INITIAL_ELO)


for date, day_matches in df.groupby("Date", sort=True):

    # Build features BEFORE updating ratings or match history
    for _, match in day_matches.iterrows():
        home = match["HomeTeam"]
        away = match["AwayTeam"]

        home_history = history.get(home, [])[-5:]
        away_history = history.get(away, [])[-5:]

        # Recent performance at home and away
        home_venue = [
            x for x in history.get(home, [])
            if x["venue"] == "home"
        ][-5:]

        away_venue = [
            x for x in history.get(away, [])
            if x["venue"] == "away"
        ][-5:]

        # Existing features
        home_recent_points = sum(x["points"] for x in home_history)
        away_recent_points = sum(x["points"] for x in away_history)

        home_goals_scored = sum(x["scored"] for x in home_history)
        home_goals_conceded = sum(x["conceded"] for x in home_history)
        away_goals_scored = sum(x["scored"] for x in away_history)
        away_goals_conceded = sum(x["conceded"] for x in away_history)

        home_venue_points = sum(x["points"] for x in home_venue)
        away_venue_points = sum(x["points"] for x in away_venue)

        home_elo = get_elo(home)
        away_elo = get_elo(away)

        # Add the match features
        features.append({
            "Date": date,
            "HomeTeam": home,
            "AwayTeam": away,

            "HomeRecentPoints": home_recent_points,
            "AwayRecentPoints": away_recent_points,

            "HomeGoalsScored": home_goals_scored,
            "HomeGoalsConceded": home_goals_conceded,
            "AwayGoalsScored": away_goals_scored,
            "AwayGoalsConceded": away_goals_conceded,

            "HomeVenuePoints": home_venue_points,
            "AwayVenuePoints": away_venue_points,

            "HomeElo": home_elo,
            "AwayElo": away_elo,

            # New difference features
            "RecentPointsDiff": (
                home_recent_points - away_recent_points
            ),
            "GoalsScoredDiff": (
                home_goals_scored - away_goals_scored
            ),
            "GoalsConcededDiff": (
                away_goals_conceded - home_goals_conceded
            ),
            "VenuePointsDiff": (
                home_venue_points - away_venue_points
            ),
            "EloDiff": home_elo - away_elo,

            "FTR": match["FTR"]
        })

    # Calculate Elo changes using ratings from before this date
    elo_changes = {}

    for _, match in day_matches.iterrows():
        home = match["HomeTeam"]
        away = match["AwayTeam"]

        hg = int(match["FTHG"])
        ag = int(match["FTAG"])

        home_elo = get_elo(home) + 50  # Home advantage
        away_elo = get_elo(away)

        expected_home = 1 / (
            1 + math.pow(10, (away_elo - home_elo) / 400)
        )

        if hg > ag:
            actual_home = 1
        elif hg < ag:
            actual_home = 0
        else:
            actual_home = 0.5

        change = ELO_K * (actual_home - expected_home)

        elo_changes[home] = elo_changes.get(home, 0) + change
        elo_changes[away] = elo_changes.get(away, 0) - change

    # Update match history after feature generation
    for _, match in day_matches.iterrows():
        home = match["HomeTeam"]
        away = match["AwayTeam"]

        hg = int(match["FTHG"])
        ag = int(match["FTAG"])

        if hg > ag:
            hpts, apts = 3, 0
        elif hg < ag:
            hpts, apts = 0, 3
        else:
            hpts, apts = 1, 1

        history.setdefault(home, []).append({
            "points": hpts,
            "scored": hg,
            "conceded": ag,
            "venue": "home"
        })

        history.setdefault(away, []).append({
            "points": apts,
            "scored": ag,
            "conceded": hg,
            "venue": "away"
        })

    # Apply Elo changes after processing all matches on this date
    for team, change in elo_changes.items():
        elo[team] = get_elo(team) + change


# Save engineered features
result = pd.DataFrame(features)
result.to_csv("epl_features.csv", index=False)

print("Feature rows:", len(result))
print("Columns:", result.columns.tolist())
print(result.head(5).to_string(index=False))
print("Saved as epl_features.csv")
