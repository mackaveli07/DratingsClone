import streamlit as st
import pandas as pd
import numpy as np
import requests
from collections import defaultdict
from difflib import get_close_matches

### ---------- CONFIG ----------
API_KEY = "4c39fd0413dbcc55279d85ab18bcc6f0"  # Default TheOddsAPI key
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
    "kc": "https://a.espncdn.com/i/teamlogos/nfl/500/kc.png",
    "sf": "https://a.espncdn.com/i/teamlogos/nfl/500/sf.png",
    "gb": "https://a.espncdn.com/i/teamlogos/nfl/500/gb.png",
    "dal": "https://a.espncdn.com/i/teamlogos/nfl/500/dal.png",
    # Add remaining teams here...
}

TEAM_ALIASES = {
    "49ers": "sf",
    "packers": "gb",
    "chiefs": "kc",
    "cowboys": "dal",
    # Add more nickname → abbreviation mappings
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
    expected2 = expected_score(r2, r1)

    actual1 = 1 if score1 > score2 else 0
    actual2 = 1 - actual1

    elo_ratings[team1] += K * (actual1 - expected1)
    elo_ratings[team2] += K * (actual2 - expected2)

def run_elo_pipeline(df):
    elo_ratings = defaultdict(lambda: BASE_ELO)
    grouped = df.groupby(["season", "week"])
    for (_, _), games in grouped:
        for _, row in games.iterrows():
            update_ratings(elo_ratings, row.team1, row.team2, row.score1, row.score2, row.home_team)
    return dict(elo_ratings)

### ---------- ODDS API ----------
@st.cache_data(ttl=30)
def get_theoddsapi_odds(api_key):
    try:
        url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"
        params = {
            "apiKey": api_key,
            "regions": REGION,
            "markets": MARKETS,
            "oddsFormat": "american",
            "dateFormat": "iso"
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Failed to fetch odds: {e}")
        return []

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
        if len(teams) != 2:
            continue
        t0, t1 = teams
        key = frozenset([t0.lower(), t1.lower()])
        bookmakers = game.get("bookmakers", [])
        bm = pick_bookmaker(bookmakers)
        if not bm:
            continue
        markets = bm.get("markets", [])
        ml = {}
        sp = {}
        for market in markets:
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
        if ml in [None, "N/A", ""]:
            return None
        val = int(str(ml).replace("+","").replace("-",""))
        if str(ml).startswith("-"):
            return val / (val + 100)
        return 100 / (val + 100)
    except Exception:
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
    spread = round(spread * 2) / 2
    if not team_is_favorite:
        spread = -spread
    return float(spread)

def format_edge_badge(edge):
    if edge is None:
        return ""
    if edge > 0.05:
        return f'<span style="color:#16a34a;font-weight:700">▲ +{edge:.1%}</span>'
    if edge < -0.05:
        return f'<span style="color:#dc2626;font-weight:700">▼ -{abs(edge):.1%}</span>'
    return f'<span style="color:#6b7280;font-weight:700">≈ {edge:.1%}</span>'

def fuzzy_find_team_in_odds(team_name, odds_index_keys):
    name = TEAM_ALIASES.get(team_name.lower(), team_name.lower())
    for key in odds_index_keys:
        for tk in key:
            tk_alias = TEAM_ALIASES.get(tk, tk)
            if name == tk_alias:
                return key
    # Fallback to difflib matching
    candidates = []
    for key in odds_index_keys:
        for tk in key:
            candidates.append(TEAM_ALIASES.get(tk, tk))
    candidates = list(set(candidates))
    matches = get_close_matches(name, candidates, n=1, cutoff=0.6)
    if matches:
        best = matches[0]
        for k in odds_index_keys:
            for tk in k:
                if TEAM_ALIASES.get(tk, tk) == best:
                    return k
    return None

### ---------- CSS ----------
APP_CSS = """
<style>
body {background: linear-gradient(120deg, #f0f4f8, #d9e2ec); font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif; color: #1f2937;}
h1 {color: #0f172a; font-weight: 800; letter-spacing: 1.2px;}
h2 {color: #334155; font-weight: 700; margin-bottom: 0.5rem;}
.matchup-card {background: #ffffffcc; border-radius: 15px; padding: 16px; margin: 12px 8px; box-shadow: 0 12px 24px rgb(0 0 0 / 0.1); transition: transform 0.2s ease, box-shadow 0.2s ease;}
.matchup-card:hover {transform: translateY(-8px); box-shadow: 0 20px 40px rgb(0 0 0 / 0.15);}
.team-block {display: flex; align-items: center; gap: 16px; margin-bottom: 6px;}
.team-logo {width: 56px; height: 56px; border-radius: 50%; box-shadow: 0 4px 10px rgb(0 0 0 / 0.1); object-fit: contain; background: white;}
.team-name {font-weight: 700; font-size: 20px; color: #1e293b; flex-grow: 1;}
.ml-badge {font-weight: 700; padding: 5px 10px; border-radius: 8px; background: #e0e7ff; color: #3730a3; font-size: 0.9rem; margin-right: 8px;}
.value-badge {background: linear-gradient(45deg, #16a34a, #22c55e); color: white; font-weight: 700; padding: 4px 12px; border-radius: 12px; font-size: 0.8rem; margin-left: 10px; display: inline-flex; align-items: center; gap: 6px;}
.value-badge svg {fill: white; width: 16px; height: 16px;}
.prob-bar {height: 14px; border-radius: 8px; overflow: hidden; background: #e2e8f0; margin-top: 6px;}
.prob-fill {height: 14px;}
.home-color { background: #2563eb; }
.away-color { background: #ef4444; }
.prob-text {font-size: 0.9rem; margin-top: 4px; color: #475569; font-weight: 600;}
.footer {font-size: 0.85rem; text-align: center; margin-top: 2rem; color: #94a3b8;}
</style>
"""

### ---------- MAIN ----------
st.set_page_config(page_title="NFL Elo + Odds Dashboard", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)
st.title("🏈 NFL Elo Betting Dashboard")
st.markdown("""
<div style="background: linear-gradient(90deg, #2563eb, #3b82f6);
            padding: 16px; border-radius: 16px; margin-bottom: 24px; color: white;
            font-weight: 800; font-size: 2.2rem; text-align: center;
            box-shadow: 0 8px 32px rgb(59 130 246 / 0.3);">
    NFL Elo & Live Odds Dashboard — Value Bets Highlighted
</div>
""", unsafe_allow_html=True)

# Load data
try:
    hist_df = pd.read_excel(EXCEL_FILE, sheet_name=HIST_SHEET)
    sched_df = pd.read_excel(EXCEL_FILE, sheet_name=SCHEDULE_SHEET)
except Exception as e:
    st.error(f"Error loading Excel file or sheets: {e}")
    st.stop()

ratings = run_elo_pipeline(hist_df)

st.sidebar.header("Controls")
use_api = st.sidebar.checkbox("Fetch live odds from TheOddsAPI", value=True)
api_key_input = st.sidebar.text_input("TheOddsAPI key (override)", value="")
if api_key_input.strip():
    API_KEY = api_key_input.strip()
prefer_book = st.sidebar.text_input("Preferred bookmaker key (optional)", value="")
if prefer_book.strip():
    BOOKMAKER_PREFERENCE = prefer_book.strip()
edge_threshold = st.sidebar.slider("Minimum Edge for Value Bets", 0.0, 0.3, 0.05, 0.01)

available_weeks = sorted(sched_df['week'].dropna().unique().astype(int).tolist())
selected_week = st.selectbox("Select Week", available_weeks, index=len(available_weeks) - 1)
week_games = sched_df[sched_df['week'] == selected_week]
if week_games.empty:
    st.info(f"No games found for week {selected_week}.")
    st.stop()

# Fetch odds
odds_index = {}
if use_api:
    api_data = get_theoddsapi_odds(API_KEY)
    odds_index = parse_odds_data(api_data)

value_bets = []

# Render matchups
for idx in range(0, len(week_games), 2):
    cols = st.columns(2)
    for i in range(2):
        if idx + i >= len(week_games):
            break
        row = week_games.iloc[idx + i]

        # Correctly assign home/away teams
        home_team = row['home_team']
        away_team = row['team2'] if row['team1'] == home_team else row['team1']

        r_home = ratings.get(home_team, BASE_ELO) + HOME_ADVANTAGE
        r_away = ratings.get(away_team, BASE_ELO)
        exp_home = expected_score(r_home, r_away)
        exp_away = 1 - exp_home

        pred_spread = probability_to_spread(exp_home, team_is_favorite=True)
        pred_ml_home = probability_to_moneyline(exp_home)
        pred_ml_away = probability_to_moneyline(exp_away)

        # Live odds lookup
        odds_key = fuzzy_find_team_in_odds(home_team, odds_index.keys())
        live_ml_home = live_ml_away = "N/A"
        live_spread_home = live_spread_away = "N/A"
        bookmaker_name = "N/A"

        if odds_key:
            live_ml_home = odds_index[odds_key]["moneyline"].get(home_team.lower(), "N/A")
            live_ml_away = odds_index[odds_key]["moneyline"].get(away_team.lower(), "N/A")
            live_spread_home = odds_index[odds_key]["spread"].get(home_team.lower(), "N/A")
            live_spread_away = odds_index[odds_key]["spread"].get(away_team.lower(), "N/A")
            bookmaker_name = odds_index[odds_key].get("bookmaker", "N/A")

        implied_home = moneyline_to_probability(live_ml_home)
        implied_away = moneyline_to_probability(live_ml_away)
        edge_home = (exp_home - implied_home) if implied_home else None
        edge_away = (exp_away - implied_away) if implied_away else None

        is_value_home = edge_home is not None and edge_home >= edge_threshold
        is_value_away = edge_away is not None and edge_away >= edge_threshold

        if is_value_home:
            value_bets.append({"Week": selected_week, "Team": home_team, "Opponent": away_team,
                               "Edge": edge_home, "Market ML": live_ml_home, "Model Prob": exp_home,
                               "Bookmaker": bookmaker_name, "Side": "Home"})
        if is_value_away:
            value_bets.append({"Week": selected_week, "Team": away_team, "Opponent": home_team,
                               "Edge": edge_away, "Market ML": live_ml_away, "Model Prob": exp_away,
                               "Bookmaker": bookmaker_name, "Side": "Away"})

        with cols[i]:
            # Render matchup card
            render_matchup_card(
                team_home=home_team,
                team_away=away_team,
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
                is_value_home=is_value_home,
                is_value_away=is_value_away
            )

# Value Bets Summary
st.markdown("---")
st.subheader("💰 Value Bets Summary")
if value_bets:
    df_val = pd.DataFrame(value_bets)
    df_val["Edge (%)"] = df_val["Edge"].apply(lambda x: f"{x:.1%}")
    st.dataframe(df_val.style.set_properties(**{
        "background-color": "#d1fae5",
        "color": "#065f46",
        "font-weight": "600"
    }), use_container_width=True)
else:
    st.info("No value bets found for this week.")

st.markdown("""
<div class="footer">
    Powered by Elo Ratings & TheOddsAPI | Made with ❤️ by Phil
</div>
""", unsafe_allow_html=True)
