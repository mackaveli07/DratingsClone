import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict

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

TEAM_LOGOS = {v: f"https://upload.wikimedia.org/wikipedia/en/{abbr}" for abbr,v in zip(NFL_FULL_NAMES.keys(), [
    "9/9e/Arizona_Cardinals_logo.svg", "c/c3/Atlanta_Falcons_logo.svg", "1/16/Baltimore_Ravens_logo.svg",
    "7/77/Buffalo_Bills_logo.svg", "7/7e/Carolina_Panthers_logo.svg", "6/63/Chicago_Bears_logo.svg",
    "2/24/Cincinnati_Bengals_logo.svg", "4/4b/Cleveland_Browns_logo.svg", "2/2e/Dallas_Cowboys.svg",
    "4/44/Denver_Broncos_logo.svg", "7/7e/Detroit_Lions_logo.svg", "5/50/Green_Bay_Packers_logo.svg",
    "2/28/Houston_Texans.svg", "7/7e/Indianapolis_Colts_logo.svg", "8/8e/Jacksonville_Jaguars_logo.svg",
    "7/72/Kansas_City_Chiefs_logo.svg", "9/9b/Las_Vegas_Raiders_logo.svg", "8/88/Los_Angeles_Chargers_logo.svg",
    "7/7a/Los_Angeles_Rams_logo.svg", "f/fd/Miami_Dolphins_logo.svg", "f/fb/Minnesota_Vikings_logo.svg",
    "b/b9/New_England_Patriots_logo.svg", "9/9f/New_Orleans_Saints_logo.svg", "6/6b/New_York_Giants_logo.svg",
    "6/6e/New_York_Jets_logo.svg", "8/8e/Philadelphia_Eagles_logo.svg", "6/6d/Pittsburgh_Steelers_logo.svg",
    "4/4f/San_Francisco_49ers_logo.svg", "7/7e/Seattle_Seahawks_logo.svg", "6/6c/Tampa_Bay_Buccaneers_logo.svg",
    "9/9e/Tennessee_Titans_logo.svg", "1/1e/Washington_Commanders_logo.svg"
])}

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

### ---------- CSS + UI ----------
APP_CSS = """
<style>
body { background: linear-gradient(120deg, #f0f4f8, #d9e2ec); font-family: "Segoe UI", sans-serif; color: #1f2937; }
h1 { color: #0f172a; font-weight: 800; }
.matchup-card { background: #ffffffcc; border-radius: 15px; padding: 16px; margin: 12px 8px; box-shadow: 0 12px 24px rgb(0 0 0 / 0.1); }
.team-block { display: flex; align-items: center; gap: 16px; margin-bottom: 6px; }
.team-logo { width: 56px; height: 56px; border-radius: 50%; object-fit: contain; background: white; }
.team-name { font-weight: 700; font-size: 20px; color: #1e293b; flex-grow: 1; }
.info-line { font-size: 0.95rem; margin-top: 4px; color: #475569; font-weight: 600; }
.prob-bar { height: 14px; border-radius: 8px; overflow: hidden; background: #e2e8f0; margin-top: 6px; }
.prob-fill { height: 14px; }
.home-color { background: #2563eb; }
.away-color { background: #ef4444; }
</style>
"""

### ---------- MAIN ----------
st.set_page_config(page_title="NFL Elo Projections", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)
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

    st.markdown(f"<div class='matchup-card'>", unsafe_allow_html=True)
    cols = st.columns(2)
    with cols[0]:
        logo_url = TEAM_LOGOS.get(team_away, "")
        st.markdown(f"""
        <div class="team-block">
            <img src="{logo_url}" class="team-logo"/>
            <div>
                <div class="team-name">{team_away}</div>
                <div class="info-line">Projected Points: {predicted_away_score}</div>
                <div class="info-line">Win Probability: {prob_away*100:.1f}%</div>
                <div class="prob-bar"><div class="prob-fill away-color" style="width:{prob_away*100:.1f}%"></div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        logo_url = TEAM_LOGOS.get(team_home, "")
        st.markdown(f"""
        <div class="team-block" style="justify-content:flex-end;">
            <div style="text-align:right;">
                <div class="team-name">{team_home}</div>
                <div class="info-line">Projected Points: {predicted_home_score}</div>
                <div class="info-line">Win Probability: {prob_home*100:.1f}%</div>
                <div class="prob-bar"><div class="prob-fill home-color" style="width:{prob_home*100:.1f}%"></div></div>
            </div>
            <img src="{logo_url}" class="team-logo"/>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
