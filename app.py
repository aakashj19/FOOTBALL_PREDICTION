from flask import Flask, render_template, request
import pandas as pd
import joblib
from pathlib import Path
import json
import time
from urllib.request import Request, urlopen

app = Flask(__name__)

# -----------------------------
# PROJECT FILES
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "football_model.pkl"
FEATURES_PATH = BASE_DIR / "epl_features.csv"
SCORES_PATH = BASE_DIR / "epl_clean.csv"

# -----------------------------
# GLOBAL DATA
# -----------------------------
model = None
feature_columns = []
df = pd.DataFrame()
scores_df = pd.DataFrame()
teams = []
last_modified = {}

standings_cache = []
standings_cache_time = 0
STANDINGS_CACHE_SECONDS = 3600


# -----------------------------
# LOAD MODEL AND DATA
# -----------------------------
def load_data():
    global model, feature_columns, df, scores_df, teams, last_modified

    files = [MODEL_PATH, FEATURES_PATH, SCORES_PATH]
    missing = [str(path) for path in files if not path.exists()]

    if missing:
        raise FileNotFoundError("Missing files: " + ", ".join(missing))

    modified = {str(path): path.stat().st_mtime for path in files}

    if modified == last_modified:
        return

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_columns = saved["feature_columns"]

    df = pd.read_csv(FEATURES_PATH)
    scores_df = pd.read_csv(SCORES_PATH)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    scores_df["Date"] = pd.to_datetime(
        scores_df["Date"], errors="coerce"
    )

    for column in ["FTHG", "FTAG"]:
        scores_df[column] = pd.to_numeric(
            scores_df[column], errors="coerce"
        )

    scores_df = scores_df.dropna(
        subset=["Date", "FTHG", "FTAG"]
    ).sort_values("Date")

    teams = sorted(
        set(df["HomeTeam"].dropna())
        | set(df["AwayTeam"].dropna())
    )

    last_modified = modified
    print(f"Loaded model and data: {len(teams)} teams")


# -----------------------------
# RECENT MATCH HISTORY
# -----------------------------
def get_team_history(team_name, limit=5):
    if scores_df.empty:
        return []

    home_matches = scores_df[
        scores_df["HomeTeam"] == team_name
    ].copy()

    away_matches = scores_df[
        scores_df["AwayTeam"] == team_name
    ].copy()

    home_matches["is_home"] = True
    away_matches["is_home"] = False

    matches = pd.concat(
        [home_matches, away_matches],
        ignore_index=True
    ).sort_values("Date", ascending=False).head(limit)

    history = []

    for _, match in matches.iterrows():
        is_home = bool(match["is_home"])

        if is_home:
            opponent = match["AwayTeam"]
            team_goals = int(match["FTHG"])
            opponent_goals = int(match["FTAG"])
        else:
            opponent = match["HomeTeam"]
            team_goals = int(match["FTAG"])
            opponent_goals = int(match["FTHG"])

        if team_goals > opponent_goals:
            result = "W"
        elif team_goals < opponent_goals:
            result = "L"
        else:
            result = "D"

        history.append({
            "opponent": opponent,
            "team_goals": team_goals,
            "opponent_goals": opponent_goals,
            "result": result,
            "is_home": is_home,
        })

    return history


# -----------------------------
# TEAM FORM FEATURES
# -----------------------------
def get_team_form(team_name):
    matches = df[
        (df["HomeTeam"] == team_name)
        | (df["AwayTeam"] == team_name)
    ].sort_values("Date").tail(5)

    form = {
        "recent_points": 0,
        "goals_scored": 0,
        "goals_conceded": 0,
        "venue_points": 0,
        "elo": 1500,
    }

    for _, match in matches.iterrows():
        is_home = match["HomeTeam"] == team_name

        if is_home:
            scored = match.get("HomeGoalsScored", 0)
            conceded = match.get("HomeGoalsConceded", 0)
            venue_points = match.get("HomeVenuePoints", 0)
            elo = match.get("HomeElo", 1500)
        else:
            scored = match.get("AwayGoalsScored", 0)
            conceded = match.get("AwayGoalsConceded", 0)
            venue_points = match.get("AwayVenuePoints", 0)
            elo = match.get("AwayElo", 1500)

        scored = 0 if pd.isna(scored) else float(scored)
        conceded = 0 if pd.isna(conceded) else float(conceded)
        venue_points = (
            0 if pd.isna(venue_points) else float(venue_points)
        )

        form["goals_scored"] += scored
        form["goals_conceded"] += conceded
        form["recent_points"] += venue_points
        form["venue_points"] += venue_points

        if pd.notna(elo):
            form["elo"] = float(elo)

    return form


# -----------------------------
# PREDICTION
# -----------------------------
def predict_match(home_team, away_team):
    home = get_team_form(home_team)
    away = get_team_form(away_team)

    values = {
        "HomeRecentPoints": home["recent_points"],
        "AwayRecentPoints": away["recent_points"],
        "HomeGoalsScored": home["goals_scored"],
        "HomeGoalsConceded": home["goals_conceded"],
        "AwayGoalsScored": away["goals_scored"],
        "AwayGoalsConceded": away["goals_conceded"],
        "HomeVenuePoints": home["venue_points"],
        "AwayVenuePoints": away["venue_points"],
        "HomeElo": home["elo"],
        "AwayElo": away["elo"],
        "RecentPointsDiff": (
            home["recent_points"] - away["recent_points"]
        ),
        "GoalsScoredDiff": (
            home["goals_scored"] - away["goals_scored"]
        ),
        "GoalsConcededDiff": (
            home["goals_conceded"] - away["goals_conceded"]
        ),
        "VenuePointsDiff": (
            home["venue_points"] - away["venue_points"]
        ),
        "EloDiff": home["elo"] - away["elo"],
    }

    input_df = pd.DataFrame(
        [[values.get(column, 0) for column in feature_columns]],
        columns=feature_columns
    )

    probabilities = model.predict_proba(input_df)[0]
    labels = list(model.classes_)

    probability_map = {
        str(label): float(probability) * 100
        for label, probability in zip(labels, probabilities)
    }

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home": round(probability_map.get("H", 0), 1),
        "draw": round(probability_map.get("D", 0), 1),
        "away": round(probability_map.get("A", 0), 1),
    }


# -----------------------------
# LIVE PREMIER LEAGUE STANDINGS
# -----------------------------
def fetch_json(url):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with urlopen(req, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def get_live_standings():
    global standings_cache, standings_cache_time

    now = time.time()

    if (
        standings_cache
        and now - standings_cache_time < STANDINGS_CACHE_SECONDS
    ):
        return standings_cache

    try:
        team_data = fetch_json(
            "https://fantasy.premierleague.com/api/bootstrap-static/"
        )["teams"]

        fixtures = fetch_json(
            "https://fantasy.premierleague.com/api/fixtures/"
        )

        table = {}

        for team in team_data:
            table[team["id"]] = {
                "team": team["name"],
                "played": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "goals_for": 0,
                "goals_against": 0,
                "goal_difference": 0,
                "points": 0,
                "form": [],
            }

        finished = [
            fixture for fixture in fixtures
            if fixture.get("finished")
            and fixture.get("team_h_score") is not None
            and fixture.get("team_a_score") is not None
        ]

        finished.sort(
            key=lambda fixture: fixture.get("kickoff_time") or ""
        )

        for fixture in finished:
            home_id = fixture["team_h"]
            away_id = fixture["team_a"]
            home_goals = fixture["team_h_score"]
            away_goals = fixture["team_a_score"]

            if home_id not in table or away_id not in table:
                continue

            home = table[home_id]
            away = table[away_id]

            home["played"] += 1
            away["played"] += 1

            home["goals_for"] += home_goals
            home["goals_against"] += away_goals
            away["goals_for"] += away_goals
            away["goals_against"] += home_goals

            if home_goals > away_goals:
                home["wins"] += 1
                away["losses"] += 1
                home["points"] += 3
                home["form"].append("W")
                away["form"].append("L")

            elif home_goals < away_goals:
                away["wins"] += 1
                home["losses"] += 1
                away["points"] += 3
                home["form"].append("L")
                away["form"].append("W")

            else:
                home["draws"] += 1
                away["draws"] += 1
                home["points"] += 1
                away["points"] += 1
                home["form"].append("D")
                away["form"].append("D")

        standings = list(table.values())

        for team in standings:
            team["goal_difference"] = (
                team["goals_for"] - team["goals_against"]
            )
            team["form"] = team["form"][-5:]

        standings.sort(
            key=lambda team: (
                -team["points"],
                -team["goal_difference"],
                -team["goals_for"],
                team["team"],
            )
        )

        for position, team in enumerate(standings, start=1):
            team["position"] = position

        standings_cache = standings
        standings_cache_time = now
        return standings

    except Exception as error:
        print("Standings update failed:", error)
        return standings_cache


# -----------------------------
# HOME PAGE AND PREDICTION
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    load_data()

    prediction = None
    home_history = []
    away_history = []
    selected_home = ""
    selected_away = ""

    if request.method == "POST":
        selected_home = request.form.get("home_team", "")
        selected_away = request.form.get("away_team", "")

        if not selected_home or not selected_away:
            prediction = {"error": "Please select both teams."}

        elif selected_home == selected_away:
            prediction = {"error": "Choose two different teams."}

        elif (
            selected_home not in teams
            or selected_away not in teams
        ):
            prediction = {"error": "Invalid team selection."}

        else:
            prediction = predict_match(
                selected_home,
                selected_away
            )
            home_history = get_team_history(selected_home)
            away_history = get_team_history(selected_away)

    return render_template(
        "index.html",
        teams=teams,
        prediction=prediction,
        home_history=home_history,
        away_history=away_history,
        selected_home=selected_home,
        selected_away=selected_away,
    )


# -----------------------------
# TEAMS DASHBOARD
# -----------------------------
@app.route("/teams")
def teams_dashboard():
    return render_template(
        "teams.html",
        standings=get_live_standings()
    )

@app.route("/standings")
def standings_dashboard():
    return render_template(
        "standings.html",
        standings=get_live_standings()
    )
# -----------------------------
# START FLASK
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)