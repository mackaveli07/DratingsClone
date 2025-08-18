import streamlit as st
import pandas as pd
import numpy as np
import requests
from collections import defaultdict

### ---------- CONFIG ----------
API_KEY = "a17f19558b3402206053bc01787a6b1b"  # TheOddsAPI key
SPORT_KEY = "americanfootball_nfl"
REGION = "us"
MARKETS = "h2h,spreads"
BOOKMAKER_PREFERENCE = None

BASE_ELO = 1500
K = 20
HOME_ADVANTAGE = 65

EXCEL_FILE = "games.xlsx"
HIST_SHEET = "games"
SCHEDULE_SHEET = "2025 schedule"

# Full names as used by TheOddsAPI
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

# Logos mapped by full name
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
    """Convert abbreviation or full name to the standard full name."""
    name = str(name).strip()
    if name in NFL_FULL_NAMES:  # abbreviation
        return NFL_FULL_NAMES[name]
    if name in NFL_FULL_NAMES.values():  # already full
        return name
    return name  # fallback (unknown team stays as-is)

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
            team1, team2 = row.get("team1"), row.get("team2")
            score1, score2 = row.get("score1",0), row.get("score2",0)
            home_team = row.get("home_team", team2)
            # map team names
            team1, team2, home_team = map_team_name(team1), map_team_name(team2), map_team_name(home_team)
            update_ratings(elo_ratings, team1, team2, score1, score2, home_team)
    return dict(elo_ratings)

### ---------- ODDS API ----------
@st.cache_data(ttl=30)
def get_theoddsapi_odds(api_key):
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"
    params = {"apiKey": api_key,"regions": REGION,"markets": MARKETS,"oddsFormat":"american","dateFormat":"iso"}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()

def pick_bookmaker(bookmakers):
    if not bookmakers: return None
    if BOOKMAKER_PREFERENCE:
        for b in bookmakers:
            if b.get("key") == BOOKMAKER_PREFERENCE:
                return b
    return bookmakers[0]

def parse_odds_data(api_data):
    odds_index = {}
    for game in api_data:
        teams = game.get("teams", [])
        if len(teams) != 2: continue
        t0, t1 = teams
        key = frozenset([t0, t1])
        bm = pick_bookmaker(game.get("bookmakers",[]))
        if not bm: continue
        ml, sp = {}, {}
        for m in bm.get("markets", []):
            if m.get("key")=="h2h":
                for o in m.get("outcomes", []):
                    ml[o.get("name","")] = o.get("price")
            elif m.get("key")=="spreads":
                for o in m.get("outcomes", []):
                    sp[o.get("name","")] = o.get("point")
        odds_index[key] = {"moneyline":ml, "spread":sp, "bookmaker":bm.get("title",bm.get("key"))}
    return odds_index

def probability_to_moneyline(prob):
    if prob is None: return "N/A"
    if prob>=0.5: return f"-{round(100*prob/(1-prob))}"
    else: return f"+{round(100*(1-prob)/prob)}"

def probability_to_spread(prob, team_is_favorite=True):
    b=0.23
    prob = max(min(prob,0.999),0.001)
    spread=np.log(prob/(1-prob))/b
    spread=round(spread*2)/2
    return float(spread if team_is_favorite else -spread)

### ---------- CSS ----------
APP_CSS = """ ... (unchanged) ... """

def render_matchup_card(...):  # unchanged
    ...

### ---------- MAIN ----------
st.set_page_config(page_title="NFL Elo + Odds Dashboard", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)
st.title("🏈 NFL Elo Betting Dashboard")

# Load data
try:
    hist_df = pd.read_excel(EXCEL_FILE, sheet_name=HIST_SHEET)
    sched_df = pd.read_excel(EXCEL_FILE, sheet_name=SCHEDULE_SHEET)
except Exception as e:
    st.error(f"Error loading Excel file or sheets: {e}")
    st.stop()

ratings = run_elo_pipeline(hist_df)

# Sidebar controls
st.sidebar.header("Controls")
use_api = st.sidebar.checkbox("Fetch live odds from TheOddsAPI", value=True)
api_key_input = st.sidebar.text_input("TheOddsAPI key (override)", value="")
if api_key_input.strip(): API_KEY=api_key_input.strip()
prefer_book = st.sidebar.text_input("Preferred bookmaker key (optional)", value="")
if prefer_book.strip(): BOOKMAKER_PREFERENCE=prefer_book.strip()

# Fixed schedule mapping
HOME_COL = "team2"   # home team
AWAY_COL = "team1"   # away team

available_weeks = sorted(sched_df['week'].dropna().unique().astype(int).tolist())
selected_week = st.selectbox("Select Week", available_weeks, index=len(available_weeks)-1)
week_games = sched_df[sched_df['week']==selected_week]
if week_games.empty:
    st.info(f"No games found for week {selected_week}.")
    st.stop()

# Fetch odds if enabled
odds_index={}
if use_api:
    try:
        api_data = get_theoddsapi_odds(API_KEY)
        odds_index = parse_odds_data(api_data)
    except Exception as e:
        st.error(f"Error fetching odds: {e}")

# Render matchups
for idx,row in week_games.iterrows():
    team_home = map_team_name(row[HOME_COL])
    team_away = map_team_name(row[AWAY_COL])

    elo_home = ratings.get(team_home, BASE_ELO)
    elo_away = ratings.get(team_away, BASE_ELO)
    prob_home = expected_score(elo_home+HOME_ADVANTAGE, elo_away)
    prob_away = 1-prob_home
    predicted_ml_home = probability_to_moneyline(prob_home)
    predicted_ml_away = probability_to_moneyline(prob_away)
    predicted_spread = probability_to_spread(prob_home, team_is_favorite=True)

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
                        prob_home, prob_away, predicted_spread,
                        predicted_ml_home, predicted_ml_away,
                        live_ml_home, live_ml_away,
                        live_spread_home, live_spread_away)

st.markdown('<div class="footer">NFL Elo Dashboard — Data & Predictions updated live</div>', unsafe_allow_html=True)
