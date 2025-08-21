import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
import os

### ---------- CONFIG ----------
BASE_ELO = 1500
K = 20
HOME_ADVANTAGE = 65

EXCEL_FILE = "games.xlsx"
HIST_SHEET = "games"
SCHEDULE_SHEET = "2025 schedule"

NFL_FULL_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers",
    "LA": "Los Angeles Rams", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SF": "San Francisco 49ers", "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders"
}

### ---------- HELPERS ----------
def map_team_name(name):
    if not name:
        return "Unknown"
    name = str(name).strip()
    key = name.upper()
    if key in NFL_FULL_NAMES:
        return NFL_FULL_NAMES[key]
    for full in NFL_FULL_NAMES.values():
        if name.lower() == full.lower():
            return full
    return name

def get_abbr(team_full):
    """Return abbreviation (BUF, KC, etc.) from full name."""
    for abbr, full in NFL_FULL_NAMES.items():
        if full == team_full:
            return abbr
    return None

def safe_logo(abbr, width=64):
    """Safely load a logo or show a placeholder if missing."""
    path = f"Logos/{abbr}.png"
    if abbr and os.path.exists(path):
        st.image(path, width=width)
    else:
        st.markdown(
            f"<div style='width:{width}px; height:{width}px; background:#e5e7eb; display:flex; align-items:center; justify-content:center; border-radius:50%; font-size:12px; color:#475569;'>{abbr or '?'}</div>",
            unsafe_allow_html=True,
        )

### ---------- ELO FUNCTIONS ----------
def expected_score(r1, r2):
    return 1 / (1 + 10 ** ((r2 - r1) / 400))

def regress_preseason(elo_ratings, reg=0.65, base=BASE_ELO):
    for t in elo_ratings:
        elo_ratings[t] = base + reg * (elo_ratings[t] - base)

def update_ratings(elo_ratings, team1, team2, score1, score2, home_team):
    r1, r2 = elo_ratings[team1], elo_ratings[team2]

    if home_team == team1:
        r1 += HOME_ADVANTAGE
    elif home_team == team2:
        r2 += HOME_ADVANTAGE

    expected1 = expected_score(r1, r2)
    actual1 = 1 if score1 > score2 else 0

    margin = abs(score1 - score2)
    if margin == 0: margin = 1
    mov_mult = np.log(margin + 1) * (2.2 / ((r1 - r2) * 0.001 + 2.2))

    elo_ratings[team1] += K * mov_mult * (actual1 - expected1)
    elo_ratings[team2] += K * mov_mult * ((1 - actual1) - expected_score(r2, r1))

def run_elo_pipeline(df):
    elo_ratings = defaultdict(lambda: BASE_ELO)
    if "season" in df.columns and "week" in df.columns:
        df = df.sort_values(["season", "week"])
        seasons = df["season"].dropna().unique().tolist()
        for i, s in enumerate(seasons):
            if i > 0:
                regress_preseason(elo_ratings)
            games = df[df["season"] == s]
            for _, row in games.iterrows():
                team1, team2 = map_team_name(row.get("team1")), map_team_name(row.get("team2"))
                score1, score2 = row.get("score1",0), row.get("score2",0)
                home_team = map_team_name(row.get("home_team", team2))
                update_ratings(elo_ratings, team1, team2, score1, score2, home_team)
    return dict(elo_ratings)

### ---------- MAIN ----------
st.set_page_config(page_title="NFL Elo Projections", layout="wide")
st.title("🏈 NFL Elo Projections")

try:
    hist_df = pd.read_excel(EXCEL_FILE, sheet_name=HIST_SHEET)
    sched_df = pd.read_excel(EXCEL_FILE, sheet_name=SCHEDULE_SHEET)
except Exception as e:
    st.error(f"Error loading Excel file or sheets: {e}")
    st.stop()

# --- Compute season-specific averages with regression ---
NFL_AVG_TOTALS = {}
overall_avg = 44
alpha = 50
if {"score1","score2","season"} <= set(hist_df.columns):
    hist_df["total_points"] = hist_df["score1"] + hist_df["score2"]
    overall_avg = hist_df["total_points"].mean()
    grouped = hist_df.groupby("season")["total_points"].agg(["mean","count"])
    for season, row in grouped.iterrows():
        season_avg, n = row["mean"], row["count"]
        blended = (season_avg * n + overall_avg * alpha) / (n + alpha)
        NFL_AVG_TOTALS[season] = blended

ratings = run_elo_pipeline(hist_df)

HOME_COL = "team2"
AWAY_COL = "team1"

available_weeks = sorted(sched_df['week'].dropna().unique().astype(int).tolist())
selected_week = st.selectbox("Select Week", available_weeks, index=len(available_weeks)-1)
week_games = sched_df[sched_df['week']==selected_week]

for _, row in week_games.iterrows():
    team_home, team_away = map_team_name(row[HOME_COL]), map_team_name(row[AWAY_COL])
    elo_home, elo_away = ratings.get(team_home, BASE_ELO), ratings.get(team_away, BASE_ELO)

    prob_home = expected_score(elo_home+HOME_ADVANTAGE, elo_away)
    prob_away = 1 - prob_home

    elo_diff = (elo_home + HOME_ADVANTAGE) - elo_away
    spread_home = elo_diff / 25
    season = row.get("season")
    avg_total = NFL_AVG_TOTALS.get(season, overall_avg)

    predicted_home_score = round((avg_total / 2) + (spread_home / 2), 1)
    predicted_away_score = round((avg_total / 2) - (spread_home / 2), 1)

    home_abbr = get_abbr(team_home)
    away_abbr = get_abbr(team_away)

    col1, col2 = st.columns([1, 1])

    with col1:
        if away_abbr:
            safe_logo(away_abbr, width=64)
        st.markdown(f"### {team_away}")
        st.write(f"**Projected Points:** {predicted_away_score}")
        st.write(f"**Win Probability:** {prob_away*100:.1f}%")

    with col2:
        if home_abbr:
            safe_logo(home_abbr, width=64)
        st.markdown(f"### {team_home}")
        st.write(f"**Projected Points:** {predicted_home_score}")
        st.write(f"**Win Probability:** {prob_home*100:.1f}%")
