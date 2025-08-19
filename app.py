import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from collections import defaultdict

### ---------- CONFIG ----------
BASE_ELO = 1500
K = 20
HOME_ADVANTAGE = 65

EXCEL_FILE = "games.xlsx"
HIST_SHEET = "games"
SCHEDULE_SHEET = "2025 schedule"

NFL_FULL_NAMES = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB":  "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC":  "Kansas City Chiefs",
    "LV":  "Las Vegas Raiders",
    "LAC": "Los Angeles Chargers",
    "LA":  "Los Angeles Rams",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE":  "New England Patriots",
    "NO":  "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SF":  "San Francisco 49ers",
    "SEA": "Seattle Seahawks",
    "TB":  "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders"
}

TEAM_LOGOS = {v: f"https://upload.wikimedia.org/wikipedia/en/{abbr}" for abbr,v in zip(NFL_FULL_NAMES.keys(), [
    "9/9e/Arizona_Cardinals_logo.svg",
    "c/c3/Atlanta_Falcons_logo.svg",
    "1/16/Baltimore_Ravens_logo.svg",
    "7/77/Buffalo_Bills_logo.svg",
    "7/7e/Carolina_Panthers_logo.svg",
    "6/63/Chicago_Bears_logo.svg",
    "2/24/Cincinnati_Bengals_logo.svg",
    "4/4b/Cleveland_Browns_logo.svg",
    "2/2e/Dallas_Cowboys.svg",
    "4/44/Denver_Broncos_logo.svg",
    "7/7e/Detroit_Lions_logo.svg",
    "5/50/Green_Bay_Packers_logo.svg",
    "2/28/Houston_Texans.svg",
    "7/7e/Indianapolis_Colts_logo.svg",
    "8/8e/Jacksonville_Jaguars_logo.svg",
    "7/72/Kansas_City_Chiefs_logo.svg",
    "9/9b/Las_Vegas_Raiders_logo.svg",
    "8/88/Los_Angeles_Chargers_logo.svg",
    "7/7a/Los_Angeles_Rams_logo.svg",
    "f/fd/Miami_Dolphins_logo.svg",
    "f/fb/Minnesota_Vikings_logo.svg",
    "b/b9/New_England_Patriots_logo.svg",
    "9/9f/New_Orleans_Saints_logo.svg",
    "6/6b/New_York_Giants_logo.svg",
    "6/6e/New_York_Jets_logo.svg",
    "8/8e/Philadelphia_Eagles_logo.svg",
    "6/6d/Pittsburgh_Steelers_logo.svg",
    "4/4f/San_Francisco_49ers_logo.svg",
    "7/7e/Seattle_Seahawks_logo.svg",
    "6/6c/Tampa_Bay_Buccaneers_logo.svg",
    "9/9e/Tennessee_Titans_logo.svg",
    "1/1e/Washington_Commanders_logo.svg"
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

def update_ratings(elo_ratings, team1, team2, score1, score2, home_team):
    r1, r2 = elo_ratings[team1], elo_ratings[team2]
    if home_team == team1:
        r1 += HOME_ADVANTAGE
    elif home_team == team2:
        r2 += HOME_ADVANTAGE
    expected1 = expected_score(r1, r2)
    actual1 = 1 if score1 > score2 else 0
    elo_ratings[team1] += K * (actual1 - expected1)
    elo_ratings[team2] += K * ((1 - actual1) - expected_score(r2, r1))

def run_elo_pipeline(df):
    elo_ratings = defaultdict(lambda: BASE_ELO)
    grouped = df.groupby(["season","week"]) if "season" in df.columns else [(None, df)]
    for _, games in grouped:
        for _, row in games.iterrows():
            team1, team2 = map_team_name(row.get("team1")), map_team_name(row.get("team2"))
            score1, score2 = row.get("score1",0), row.get("score2",0)
            home_team = map_team_name(row.get("home_team", team2))
            update_ratings(elo_ratings, team1, team2, score1, score2, home_team)
    return dict(elo_ratings)

### ---------- TEAMRANKINGS SCRAPER ----------
@st.cache_data(ttl=3600)
def get_teamrankings_odds():
    url = "https://www.teamrankings.com/nfl/odds/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.183"
        )
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.select_one("table.tr-table")
    if not table:
        return {}

    rows = table.select("tbody tr")
    if not rows:
        return {}

    odds_index = {}
    for tr in rows:
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cols) < 6:
            continue
        team_away, team_home = cols[0], cols[1]
        spread_away, spread_home = cols[2], cols[3]
        ml_away, ml_home = cols[4], cols[5]

        key = frozenset([map_team_name(team_home), map_team_name(team_away)])
        odds_index[key] = {
            "moneyline": {
                map_team_name(team_home): ml_home,
                map_team_name(team_away): ml_away
            },
            "spread": {
                map_team_name(team_home): spread_home,
                map_team_name(team_away): spread_away
            },
            "bookmaker": "TeamRankings"
        }
    return odds_index

### ---------- PROBABILITY HELPERS ----------
def probability_to_moneyline(prob):
    if prob is None: return "N/A"
    if prob>=0.5: return f"-{round(100*prob/(1-prob))}"
    else: return f"+{round(100*(1-prob)/prob)}"

def probability_to_spread(prob):
    b=0.23
    prob = max(min(prob,0.999),0.001)
    spread=np.log(prob/(1-prob))/b
    spread=round(spread*2)/2
    return float(spread)

### ---------- CSS + UI ----------
APP_CSS = """
<style>
body { background: linear-gradient(120deg, #f0f4f8, #d9e2ec); font-family: "Segoe UI", sans-serif; color: #1f2937; }
h1 { color: #0f172a; font-weight: 800; letter-spacing: 1.2px; }
.matchup-card { background: #ffffffcc; border-radius: 15px; padding: 16px; margin: 12px 8px; box-shadow: 0 12px 24px rgb(0 0 0 / 0.1); }
.team-block { display: flex; align-items: center; gap: 16px; margin-bottom: 6px; }
.team-logo { width: 56px; height: 56px; border-radius: 50%; object-fit: contain; background: white; }
.team-name { font-weight: 700; font-size: 20px; color: #1e293b; flex-grow: 1; }
.ml-badge { font-weight: 700; padding: 5px 10px; border-radius: 8px; background: #e0e7ff; color: #3730a3; font-size: 0.9rem; margin-right: 8px; }
.prob-bar { height: 14px; border-radius: 8px; overflow: hidden; background: #e2e8f0; margin-top: 6px; }
.prob-fill { height: 14px; }
.home-color { background: #2563eb; }
.away-color { background: #ef4444; }
.prob-text { font-size: 0.9rem; margin-top: 4px; color: #475569; font-weight: 600; }
</style>
"""

def render_matchup_card(team_home, team_away, logos, odds_book,
                        prob_home, prob_away,
                        spread_home, spread_away,
                        predicted_ml_home, predicted_ml_away,
                        live_ml_home, live_ml_away,
                        live_spread_home, live_spread_away):
    st.markdown(f"<div class='matchup-card'>", unsafe_allow_html=True)
    cols=st.columns(2)
    with cols[0]:
        logo_url = logos.get(team_away, "")
        st.markdown(f"""
        <div class="team-block">
            <img src="{logo_url}" class="team-logo"/>
            <div>
                <div class="team-name">{team_away}</div>
                <div><span class="ml-badge">Model ML: {predicted_ml_away}</span> <span class="ml-badge">Live ML: {live_ml_away}</span></div>
                <div>Model Spread: <strong>{spread_away:+.1f}</strong> | Live Spread: <strong>{live_spread_away}</strong></div>
                <div class="prob-bar"><div class="prob-fill away-color" style="width:{prob_away*100:.1f}%"></div></div>
                <div class="prob-text">{prob_away*100:.1f}% Win Probability</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        logo_url = logos.get(team_home, "")
        st.markdown(f"""
        <div class="team-block" style="justify-content:flex-end;">
            <div style="text-align:right;">
                <div class="team-name">{team_home}</div>
                <div><span class="ml-badge">Model ML: {predicted_ml_home}</span> <span class="ml-badge">Live ML: {live_ml_home}</span></div>
                <div>Model Spread: <strong>{spread_home:+.1f}</strong> | Live Spread: <strong>{live_spread_home}</strong></div>
                <div class="prob-bar"><div class="prob-fill home-color" style="width:{prob_home*100:.1f}%"></div></div>
                <div class="prob-text">{prob_home*100:.1f}% Win Probability</div>
            </div>
            <img src="{logo_url}" class="team-logo"/>
        </div>
        """, unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center; margin-top:12px; font-weight:700; color:#475569;'>Bookmaker: {odds_book}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

### ---------- MAIN ----------
st.set_page_config(page_title="NFL Elo + Odds Dashboard", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)
st.title("🏈 NFL Elo Betting Dashboard")

try:
    hist_df = pd.read_excel(EXCEL_FILE, sheet_name=HIST_SHEET)
    sched_df = pd.read_excel(EXCEL_FILE, sheet_name=SCHEDULE_SHEET)
except Exception as e:
    st.error(f"Error loading Excel file or sheets: {e}")
    st.stop()

ratings = run_elo_pipeline(hist_df)

HOME_COL = "team2"
AWAY_COL = "team1"

available_weeks = sorted(sched_df['week'].dropna().unique().astype(int).tolist())
selected_week = st.selectbox("Select Week", available_weeks, index=len(available_weeks)-1)
week_games = sched_df[sched_df['week']==selected_week]
if week_games.empty:
    st.info(f"No games found for week {selected_week}.")
    st.stop()

odds_index = {}
try:
    odds_index = get_teamrankings_odds()
except Exception as e:
    st.error(f"Error fetching odds: {e}")

for idx,row in week_games.iterrows():
    team_home = map_team_name(row[HOME_COL])
    team_away = map_team_name(row[AWAY_COL])

    elo_home = ratings.get(team_home, BASE_ELO)
    elo_away = ratings.get(team_away, BASE_ELO)
    prob_home = expected_score(elo_home+HOME_ADVANTAGE, elo_away)
    prob_away = 1-prob_home
    predicted_ml_home = probability_to_moneyline(prob_home)
    predicted_ml_away = probability_to_moneyline(prob_away)

    spread_value = probability_to_spread(max(prob_home, prob_away))
    if prob_home > prob_away:
        spread_home = -abs(spread_value)
        spread_away = +abs(spread_value)
    else:
        spread_home = +abs(spread_value)
        spread_away = -abs(spread_value)

    odds_key = frozenset([team_home, team_away])
    live_ml_home, live_ml_away, live_spread_home, live_spread_away, bookmaker_name = "N/A","N/A","N/A","N/A","N/A"
    if odds_key in odds_index:
        data=odds_index[odds_key]
        bookmaker_name=data.get("bookmaker","N/A")
        ml=data.get("moneyline",{})
        sp=data.get("spread",{})
        live_ml_home=ml.get(team_home,"N/A")
        live_ml_away=ml.get(team_away,"N/A")
        live_spread_home=sp.get(team_home,"N/A")
        live_spread_away=sp.get(team_away,"N/A")

    render_matchup_card(team_home, team_away, TEAM_LOGOS, bookmaker_name,
                        prob_home, prob_away,
                        spread_home, spread_away,
                        predicted_ml_home, predicted_ml_away,
                        live_ml_home, live_ml_away,
                        live_spread_home, live_spread_away)

st.markdown('<div class="footer">NFL Elo Dashboard — Data & Predictions updated live from TeamRankings</div>', unsafe_allow_html=True)
