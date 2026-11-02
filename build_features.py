
import pandas as pd

# Load and sort historical matches
df = pd.read_csv("epl_final.csv")
df["MatchDate"] = pd.to_datetime(df["MatchDate"])
df = df.sort_values("MatchDate").reset_index(drop=True)

# Store each team's previous match information
team_history = {}
features = []

for _, match in df.iterrows():
    home = match["HomeTeam"]
    away = match["AwayTeam"]

    # Get each team's previous results (before this match)
    home_history = team_history.get(home, [])
    away_history = team_history.get(away, [])

    # Calculate points from the previous 5 matches
    home_points = sum(x["points"] for x in home_history[-5:])
    away_points = sum(x["points"] for x in away_history[-5:])

    # Save the features for this match
    features.append({
        "HomeTeam": home,
        "AwayTeam": away,
        "HomeRecentPoints": home_points,
        "AwayRecentPoints": away_points,
        "FullTimeResult": match["FullTimeResult"]
    })

    # Update history only AFTER recording the features
    home_goals = match["FullTimeHomeGoals"]
    away_goals = match["FullTimeAwayGoals"]

    if home not in team_history:
        team_history[home] = []
    if away not in team_history:
        team_history[away] = []

    if home_goals > away_goals:
        home_pts, away_pts = 3, 0
    elif home_goals < away_goals:
        home_pts, away_pts = 0, 3
    else:
        home_pts, away_pts = 1, 1

    team_history[home].append({"points": home_pts})
    team_history[away].append({"points": away_pts})

# Convert features into a DataFrame
features_df = pd.DataFrame(features)

print("Feature dataset shape:", features_df.shape)
print("\nFirst 10 rows:")
print(features_df.head(10).to_string(index=False))

# Save the prepared features
features_df.to_csv("match_features.csv", index=False)
print("\nSaved features to match_features.csv")

