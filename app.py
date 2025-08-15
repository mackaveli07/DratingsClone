import streamlit as st
import pandas as pd
import numpy as np
import requests
from collections import defaultdict
from difflib import get_close_matches

### ---------- CONFIG ----------
API_KEY = "a17f19558b3402206053bc01787a6b1b"  # Replace with your TheOddsAPI key
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

TEAM_LOGOS = {
   "ARI": "https://upload.wikimedia.org/wikipedia/en/9/9e/Arizona_Cardinals_logo.svg",
   "ATL": "https://upload.wikimedia.org/wikipedia/en/c/c3/Atlanta_Falcons_logo.svg",
   "BAL": "https://upload.wikimedia.org/wikipedia/en/1/16/Baltimore_Ravens_logo.svg",
   "BUF": "https://upload.wikimedia.org/wikipedia/en/7/77/Buffalo_Bills_logo.svg",
   "CAR": "https://upload.wikimedia.org/wikipedia/en/7/7e/Carolina_Panthers_logo.svg",
   "CHI": "https://upload.wikimedia.org/wikipedia/6/63/Chicago_Bears_logo.svg",
   "CIN": "https://upload.wikimedia.org/wikipedia/2/24/Cincinnati_Bengals_logo.svg",
   "CLE": "https://upload.wikimedia.org/wikipedia/4/4b/Cleveland_Browns_logo.svg",
   "DAL": "https://upload.wikimedia.org/wikipedia/2/2e/Dallas_Cowboys.svg",
   "DEN": "https://upload.wikimedia.org/wikipedia/en/4/44/Denver_Broncos_logo.svg",
   "DET": "https://upload.wikimedia.org/wikipedia/7/7e/Detroit_Lions_logo.svg",
   "GB": "https://upload.wikimedia.org/wikipedia/commons/5/50/Green_Bay_Packers_logo.svg",
   "HOU": "https://upload.wikimedia.org/wikipedia/en/2/28/Houston_Texans.svg",
   "IND": "https://upload.wikimedia.org/wikipedia/7/7e/Indianapolis_Colts_logo.svg",
   "JAX": "https://upload.wikimedia.org/wikipedia/en/8/8e/Jacksonville_Jaguars_logo.svg",
   "KC": "https://upload.wikimedia.org/wikipedia/en/7/72/Kansas_City_Chiefs_logo.svg",
   "LV": "https://upload.wikimedia.org/wikipedia/en/9/9b/Las_Vegas_Raiders_logo.svg",
   "LAC": "https://upload.wikimedia.org/wikipedia/8/88/Los_Angeles_Chargers_logo.svg",
   "LA": "https://upload.wikimedia.org/wikipedia/en/7/7a/Los_Angeles_Rams_logo.svg",
   "MIA": "https://upload.wikimedia.org/wikipedia/en/f/fd/Miami_Dolphins_logo.svg",
   "MIN": "https://upload.wikimedia.org/wikipedia/en/f/fb/Minnesota_Vikings_logo.svg",
   "NE": "https://upload.wikimedia.org/wikipedia/en/b/b9/New_England_Patriots_logo.svg",
   "NO": "https://upload.wikimedia.org/wikipedia/en/9/9f/New_Orleans_Saints_logo.svg",
   "NYG": "https://upload.wikimedia.org/wikipedia/6/6b/New_York_Giants_logo.svg",
   "NYJ": "https://upload.wikimedia.org/wikipedia/en/6/6e/New_York_Jets_logo.svg",
   "PHI": "https://upload.wikimedia.org/wikipedia/en/8/8e/Philadelphia_Eagles_logo.svg",
   "PIT": "https://upload.wikimedia.org/wikipedia/en/6/6d/Pittsburgh_Steelers_logo.svg",
   "SF": "https://upload.wikimedia.org/wikipedia/4/4f/San_Francisco_49ers_logo.svg",
   "SEA": "https://upload.wikimedia.org/wikipedia/en/7/7e/Seattle_Seahawks_logo.svg",
   "TB": "https://upload.wikimedia.org/wikipedia/en/6/6c/Tampa_Bay_Buccaneers_logo.svg",
   "TEN": "https://upload.wikimedia.org/wikipedia/en/9/9e/Tennessee_Titans_logo.svg",
   "WAS": "https://upload.wikimedia.org/wikipedia/en/1/1e/Washington_Commanders_logo.svg"
}

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
    elo_ratings[team2] += K * ((1 - actual1) - (1 - expected1))

def run_elo_pipeline(df):
    elo_ratings = defaultdict(lambda: BASE_ELO)
    grouped = df.groupby(["season", "week"])
    for (_, _), games in grouped:
        for _, row in games.iterrows():
            update_ratings(elo_ratings, row.team1, row.team2, row.score1, row.score2, row.home_team)
    return dict(elo_ratings)

### ---------- ODDS API ----------
@st.cache_data(ttl=900)
def get_theoddsapi_odds(api_key):
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"
    params = {"apiKey": api_key, "regions": REGION, "markets": MARKETS, "oddsFormat": "american", "dateFormat": "iso"}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()

def pick_bookmaker(bookmakers):
    if not bookmakers:
        return None
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
        key = frozenset([t0.lower(), t1.lower()])
        bm = pick_bookmaker(game.get("bookmakers", []))
        if not bm: continue
        ml, sp = {}, {}
        for market in bm.get("markets", []):
            if market.get("key") == "h2h":
                for outcome in market.get("outcomes", []):
                    ml[outcome.get("name", "").lower()] = outcome.get("price")
            elif market.get("key") == "spreads":
                for outcome in market.get("outcomes", []):
                    sp[outcome.get("name", "").lower()] = outcome.get("point")
        odds_index[key] = {"moneyline": ml, "spread": sp, "bookmaker": bm.get("title", bm.get("key"))}
    return odds_index

### ---------- HELPERS ----------
def moneyline_to_probability(ml):
    try:
        if ml in [None, "N/A", ""]: return None
        s = str(ml)
        if s.startswith('+'):
            val = float(s.replace('+',''))
            return 100.0 / (val + 100.0)
        if s.startswith('-'):
            val = float(s.replace('-',''))
            return val / (val + 100.0)
        val = float(s)
        return 100.0 / (val + 100.0) if val > 0 else abs(val)/(abs(val)+100.0)
    except:
        return None

def probability_to_moneyline(prob):
    if prob is None: return "N/A"
    if prob >= 0.5:
        return f"-{round(100 * prob / (1 - prob))}"
    else:
        return f"+{round(100 * (1 - prob) / prob)}"

def probability_to_spread(prob, team_is_favorite=True):
    b = 0.23
    prob = max(min(prob, 0.999), 0.001)
    spread = np.log(prob / (1 - prob)) / b
    spread = round(spread * 2)/2
    return spread if team_is_favorite else -spread

def format_edge_badge(edge):
    if edge is None: return ""
    if edge > 0.05: return f'<span style="color:#16a34a;font-weight:700">▲ +{edge:.1%}</span>'
    if edge < -0.05: return f'<span style="color:#dc2626;font-weight:700">▼ -{abs(edge):.1%}</span>'
    return f'<span style="color:#6b7280;font-weight:700">≈ {edge:.1%}</span>'

def fuzzy_find_team_in_odds(team_name, odds_index_keys):
    name = team_name.lower()
    for key in odds_index_keys:
        for tk in key:
            if name == tk: return key
    candidates = list({tk for key in odds_index_keys for tk in key})
    matches = get_close_matches(name, candidates, n=1, cutoff=0.6)
    if matches:
        best = matches[0]
        for k in odds_index_keys:
            if best in k: return k
    return None

### ---------- CSS ----------
APP_CSS = """..."""  # Keep your CSS as is

### ---------- RENDER MATCHUP ----------
def render_matchup_card(team_home, team_away, logos, odds_book,
                        prob_home, prob_away, predicted_spread,
                        predicted_ml_home, predicted_ml_away,
                        live_ml_home, live_ml_away,
                        live_spread_home, live_spread_away,
                        edge_home=None, edge_away=None,
                        is_value_home=False, is_value_away=False):
    value_icon_svg = """..."""  # Keep SVG as is
    st.markdown(f"<div class='matchup-card'>", unsafe_allow_html=True)
    cols = st.columns([1,1])
    for col, team, prob, ml_model, ml_live, spread_model, spread_live, edge, is_value in [
        (cols[0], team_away, prob_away, predicted_ml_away, live_ml_away, -predicted_spread, -float(spread_live if spread_live != 'N/A' else 0), edge_away, is_value_away),
        (cols[1], team_home, prob_home, predicted_ml_home, live_ml_home, predicted_spread, spread_live_home, edge_home, is_value_home)
    ]:
        with col:
            logo_url = logos.get(team.upper(), "")
            value_html = f'<span class="value-badge">{value_icon_svg} VALUE</span>' if is_value else ""
            st.markdown(f"""
            <div class="team-block">
                <img src="{logo_url}" class="team-logo"/>
                <div>
                    <div class="team-name">{team} {value_html}</div>
                    <div>
                        <span class="ml-badge">Model ML: {ml_model}</span>
                        <span class="ml-badge">Live ML: {ml_live}</span>
                    </div>
                    <div>
                        Model Spread: <strong>{spread_model:+.1f}</strong> |
                        Live Spread: <strong>{spread_live}</strong>
                    </div>
                    <div class="prob-bar">
                        <div class="prob-fill {'home-color' if col==cols[1] else 'away-color'}" style="width: {max(prob*100,1):.1f}%;"></div>
                    </div>
                    <div class="prob-text">{prob*100:.1f}% Win Probability</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align:center; margin-top: 12px; font-weight:700; color:#475569;">
        Predicted Spread: {predicted_spread:+.1f} &nbsp;&nbsp;|&nbsp;&nbsp; Bookmaker: {odds_book}
    </div>
    """, unsafe_allow_html=True)
    if edge_home or edge_away:
        st.markdown(f"""
        <div style="text-align:center; margin-top: 6px; font-weight:700; font-size: 1.1rem; color:#334155;">
            Home Edge: {format_edge_badge(edge_home)} &nbsp;&nbsp;|&nbsp;&nbsp; Away Edge: {format_edge_badge(edge_away)}
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

### ---------- MAIN ----------
st.set_page_config(page_title="NFL Elo + Odds Dashboard", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)
st.title("🏈 NFL Elo Betting Dashboard")
st.markdown("""<div style="background: linear-gradient(90deg, #2563eb, #3b82f6); padding: 16px; border-radius: 16px; margin-bottom: 24px; color: white; font-weight: 800; font-size: 2.2rem; text-align: center; box-shadow: 0 8px 32px rgb(59 130 246 / 0.3);">
NFL Elo & Live Odds Dashboard — Value Bets Highlighted
</div>""", unsafe_allow_html=True)

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
if api_key_input.strip(): API_KEY = api_key_input.strip()
prefer_book = st.sidebar.text_input("Preferred bookmaker key (optional)", value="")
if prefer_book.strip(): BOOKMAKER_PREFERENCE = prefer_book.strip()

available_weeks = sorted(sched_df['week'].dropna().unique().astype(int).tolist())
selected_week = st.selectbox("Select Week", available_weeks, index=len(available_weeks)-1)
week_games = sched_df[sched_df['week'] == selected_week]
if week_games.empty:
    st.info(f"No games found for week {selected_week}.")
    st.stop()

# Fetch odds
odds_index = {}
if use_api:
    try: odds_index = parse_odds_data(get_theoddsapi_odds(API_KEY))
    except Exception as e: st.error(f"Error fetching odds: {e}")

# Render each matchup
for _, row in week_games.iterrows():
    team_away, team_home = row['team1'], row['team2']
    r_away = ratings.get(team_away, BASE_ELO)
    r_home = ratings.get(team_home, BASE_ELO) + HOME_ADVANTAGE
    exp_home = expected_score(r_home, r_away)
    exp_away = 1 - exp_home

    team_fav = exp_home >= exp_away
    pred_spread = probability_to_spread(exp_home if team_fav else exp_away, team_is_favorite=team_fav)
    pred_ml_home = probability_to_moneyline(exp_home)
    pred_ml_away = probability_to_moneyline(exp_away)

    odds_key = fuzzy_find_team_in_odds(team_away, odds_index.keys())
    live_ml_home = live_ml_away = live_spread_home = live_spread_away = "N/A"
    bookmaker_name = "N/A"
    if odds_key:
        live_ml_away = odds_index[odds_key]["moneyline"].get(team_away.lower(), "N/A")
        live_ml_home = odds_index[odds_key]["moneyline"].get(team_home.lower(), "N/A")
        live_spread_away = odds_index[odds_key]["spread"].get(team_away.lower(), "N/A")
        live_spread_home = odds_index[odds_key]["spread"].get(team_home.lower(), "N/A")
        bookmaker_name = odds_index[odds_key]["bookmaker"]

    # Calculate edge
    try:
        edge_home = moneyline_to_probability(live_ml_home) - exp_home if live_ml_home != "N/A" else None
        edge_away = moneyline_to_probability(live_ml_away) - exp_away if live_ml_away != "N/A" else None
    except: edge_home = edge_away = None

    render_matchup_card(
        team_home=team_home,
        team_away=team_away,
        logos=TEAM_LOGOS,
        odds_book=bookmaker_name,
        prob_home=exp_home,
        prob_away=exp_away,
        predicted_spread=pred_spread,
        predicted_ml_home=pred_ml_home,
        predicted_ml_away=pred_ml_away,
        live_ml_home=live_ml_home,
        live_ml_away=live_ml_away,
        live_spread_home=live_spread_home,
        live_spread_away=live_spread_away,
        edge_home=edge_home,
        edge_away=edge_away,
        is_value_home=edge_home is not None and edge_home >= 0.05,
        is_value_away=edge_away is not None and edge_away >= 0.05
    )

st.markdown('<div class="footer">NFL Elo Dashboard — Data & Predictions updated live</div>', unsafe_allow_html=True)
