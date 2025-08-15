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
    expected2 = expected_score(r2, r1)
    actual1 = 1 if score1 > score2 else 0
    actual2 = 1 - actual1
    elo_ratings[team1] += K * (actual1 - expected1)
    elo_ratings[team2] += K * (actual2 - expected2)

def run_elo_pipeline(df):
    elo_ratings = defaultdict(lambda: BASE_ELO)
    grouped = df.groupby(["season","week"])
    for (_, _), games in grouped:
        for _, row in games.iterrows():
            update_ratings(elo_ratings, row.team1, row.team2, row.score1, row.score2, row.home_team)
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
        key = frozenset([t0.lower(), t1.lower()])
        bm = pick_bookmaker(game.get("bookmakers", []))
        if not bm: continue
        markets = bm.get("markets", [])
        ml, sp = {}, {}
        for market in markets:
            if market.get("key") == "h2h":
                for outcome in market.get("outcomes", []):
                    ml[outcome.get("name","").lower()] = outcome.get("price")
            elif market.get("key") == "spreads":
                for outcome in market.get("outcomes", []):
                    sp[outcome.get("name","").lower()] = outcome.get("point")
        odds_index[key] = {"moneyline": ml,"spread": sp,"bookmaker": bm.get("title",bm.get("key"))}
    return odds_index

### ---------- HELPERS ----------
def moneyline_to_probability(ml):
    try:
        if ml in [None,"N/A",""]: return None
        s = str(ml)
        if s.startswith('+'): val=int(s.replace('+','')); return 100/(val+100)
        if s.startswith('-'): val=int(s.replace('-','')); return val/(val+100)
        val=int(s)
        if val>0: return 100/(val+100)
        else: val=abs(val); return val/(val+100)
    except: return None

def probability_to_moneyline(prob):
    if prob is None: return "N/A"
    if prob >= 0.5: return f"-{round(100*prob/(1-prob))}"
    else: return f"+{round(100*(1-prob)/prob)}"

def probability_to_spread(prob, team_is_favorite=True):
    b = 0.23
    prob = max(min(prob, 0.999),0.001)
    spread = np.log(prob/(1-prob))/b
    spread = round(spread*2)/2
    if not team_is_favorite: spread=-spread
    return float(spread)

def format_edge_badge(edge):
    if edge is None: return ""
    if edge>0.05: return f'<span style="color:#16a34a;font-weight:700">▲ +{edge:.1%}</span>'
    if edge<-0.05: return f'<span style="color:#dc2626;font-weight:700">▼ -{abs(edge):.1%}</span>'
    return f'<span style="color:#6b7280;font-weight:700">≈ {edge:.1%}</span>'

def fuzzy_find_team_in_odds(team_name, odds_index_keys):
    name=team_name.lower()
    for key in odds_index_keys:
        for tk in key:
            if name==tk: return key
    candidates=list({tk for k in odds_index_keys for tk in k})
    matches=get_close_matches(name, candidates, n=1, cutoff=0.6)
    if matches:
        best=matches[0]
        for k in odds_index_keys:
            if best in k: return k
    return None

### ---------- CSS ----------
APP_CSS="""
<style>
body { background: linear-gradient(120deg,#f0f4f8,#d9e2ec); font-family:"Segoe UI",Tahoma,Geneva,Verdana,sans-serif; color:#1f2937;}
h1{color:#0f172a;font-weight:800;letter-spacing:1.2px;}
.matchup-card{background:#ffffffcc;border-radius:15px;padding:16px;margin:12px 8px;box-shadow:0 12px 24px rgb(0 0 0 / 0.1);transition: transform 0.2s ease,box-shadow 0.2s ease;}
.team-block{display:flex;align-items:center;gap:16px;margin-bottom:6px;}
.team-logo{width:56px;height:56px;border-radius:50%;box-shadow:0 4px 10px rgb(0 0 0 / 0.1);object-fit:contain;background:white;}
.team-name{font-weight:700;font-size:20px;color:#1e293b;flex-grow:1;}
.ml-badge{font-weight:700;padding:5px 10px;border-radius:8px;background:#e0e7ff;color:#3730a3;font-size:0.9rem;margin-right:8px;}
.value-badge{background:linear-gradient(45deg,#16a34a,#22c55e);color:white;font-weight:700;padding:4px 12px;border-radius:12px;font-size:0.8rem;margin-left:10px;display:inline-flex;align-items:center;gap:6px;}
.prob-bar{height:14px;border-radius:8px;overflow:hidden;background:#e2e8f0;margin-top:6px;}
.prob-fill{height:14px;}
.home-color{background:#2563eb;}
.away-color{background:#ef4444;}
.prob-text{font-size:0.9rem;margin-top:4px;color:#475569;font-weight:600;}
.footer{font-size:0.85rem;text-align:center;margin-top:2rem;color:#94a3b8;}
</style>
"""

### ---------- RENDER MATCHUP ----------
def render_matchup_card(team_home, team_away, logos, odds_book,
                        prob_home, prob_away, predicted_spread,
                        predicted_ml_home, predicted_ml_away,
                        live_ml_home, live_ml_away,
                        live_spread_home, live_spread_away,
                        edge_home=None, edge_away=None,
                        is_value_home=False, is_value_away=False):
    value_icon_svg="""
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-star" viewBox="0 0 24 24">
      <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
    </svg>
    """
    st.markdown(f"<div class='matchup-card'>", unsafe_allow_html=True)
    cols=st.columns([1,1])

    # Away
    with cols[0]:
        logo_url=logos.get(team_away.upper(),"")
        value_html=f'<span class="value-badge">{value_icon_svg} VALUE</span>' if is_value_away else ""
        st.markdown(f"""
        <div class="team-block">
            <img src="{logo_url}" class="team-logo"/>
            <div>
                <div class="team-name">{team_away} {value_html}</div>
                <div>
                    <span class="ml-badge">Model ML: {predicted_ml_away}</span>
                    <span class="ml-badge">Live ML: {live_ml_away}</span>
                </div>
                <div>
                    Model Spread: <strong>{-predicted_spread:.1f}</strong> |
                    Live Spread: <strong>{-float(live_spread_away) if live_spread_away!="N/A" else "N/A"}</strong>
                </div>
                <div class="prob-bar">
                    <div class="prob-fill away-color" style="width:{prob_away*100:.1f}%;"></div>
                </div>
                <div class="prob-text">{prob_away*100:.1f}% Win Probability</div>
            </div>
        </div>
        """,unsafe_allow_html=True)

    # Home
    with cols[1]:
        logo_url=logos.get(team_home.upper(),"")
        value_html=f'<span class="value-badge">{value_icon_svg} VALUE</span>' if is_value_home else ""
        st.markdown(f"""
        <div class="team-block" style="justify-content:flex-end;">
            <div style="text-align:right;">
                <div class="team-name">{team_home} {value_html}</div>
                <div>
                    <span class="ml-badge">Model ML: {predicted_ml_home}</span>
                    <span class="ml-badge">Live ML: {live_ml_home}</span>
                </div>
                <div>
                    Model Spread: <strong>{predicted_spread:+.1f}</strong> |
                    Live Spread: <strong>{live_spread_home}</strong>
                </div>
                <div class="prob-bar">
                    <div class="prob-fill home-color" style="width:{prob_home*100:.1f}%;"></div>
                </div>
                <div class="prob-text">{prob_home*100:.1f}% Win Probability</div>
            </div>
            <img src="{logo_url}" class="team-logo"/>
        </div>
        """,unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center; margin-top: 12px; font-weight:700; color:#475569;">
        Predicted Spread: {predicted_spread:+.1f} &nbsp;&nbsp;|&nbsp;&nbsp; Bookmaker: {odds_book}
    </div>
    """,unsafe_allow_html=True)

    edge_home_html=format_edge_badge(edge_home) if edge_home else ""
    edge_away_html=format_edge_badge(edge_away) if edge_away else ""
    if edge_home_html or edge_away_html:
        st.markdown(f"""
        <div style="text-align:center; margin-top:6px; font-weight:700; font-size:1.1rem; color:#334155;">
            Home Edge: {edge_home_html} &nbsp;&nbsp;|&nbsp;&nbsp; Away Edge: {edge_away_html}
        </div>
        """,unsafe_allow_html=True)

    st.markdown("</div>",unsafe_allow_html=True)

### ---------- MAIN ----------
st.set_page_config(page_title="NFL Elo + Odds Dashboard", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)
st.title("🏈 NFL Elo Betting Dashboard")

# Load data
try:
    hist_df=pd.read_excel(EXCEL_FILE, sheet_name=HIST_SHEET)
    sched_df=pd.read_excel(EXCEL_FILE, sheet_name=SCHEDULE_SHEET)
except Exception as e:
    st.error(f"Error loading Excel file: {e}")
    st.stop()

ratings=run_elo_pipeline(hist_df)

st.sidebar.header("Controls")
use_api=st.sidebar.checkbox("Fetch live odds from TheOddsAPI", value=True)
api_key_input=st.sidebar.text_input("TheOddsAPI key (override)", value="")
if api_key_input.strip(): API_KEY=api_key_input.strip()
prefer_book=st.sidebar.text_input("Preferred bookmaker key (optional)", value="")
if prefer_book.strip(): BOOKMAKER_PREFERENCE=prefer_book.strip()

available_weeks=sorted(sched_df['week'].dropna().unique().astype(int).tolist())
selected_week=st.selectbox("Select Week", available_weeks, index=len(available_weeks)-1)

week_games=sched_df[sched_df['week']==selected_week]
if week_games.empty:
    st.info(f"No games found for week {selected_week}")
    st.stop()

odds_index={}
if use_api:
    try: api_data=get_theoddsapi_odds(API_KEY); odds_index=parse_odds_data(api_data)
    except Exception as e: st.error(f"Error fetching odds: {e}")

# Render matchups
for idx in range(len(week_games)):
    row=week_games.iloc[idx]
    team_home=row['team_home']
    team_away=row['team_away']
    elo_home=ratings.get(team_home, BASE_ELO)
    elo_away=ratings.get(team_away, BASE_ELO)
    prob_home=expected_score(elo_home, elo_away)
    prob_away=1-prob_home
    predicted_spread=probability_to_spread(prob_home)
    predicted_ml_home=probability_to_moneyline(prob_home)
    predicted_ml_away=probability_to_moneyline(prob_away)

    # Live odds
    odds_key=fuzzy_find_team_in_odds(team_home, odds_index.keys())
    live_ml_home, live_ml_away = "N/A","N/A"
    live_spread_home, live_spread_away = "N/A","N/A"
    odds_book=""
    if odds_key and odds_key in odds_index:
        data=odds_index[odds_key]
        odds_book=data.get("bookmaker","")
        ml=data.get("moneyline",{})
        sp=data.get("spread",{})
        live_ml_home=ml.get(team_home.lower(),"N/A")
        live_ml_away=ml.get(team_away.lower(),"N/A")
        live_spread_home=sp.get(team_home.lower(),"N/A")
        live_spread_away=sp.get(team_away.lower(),"N/A")

    # Edge
    edge_home = moneyline_to_probability(live_ml_home)-prob_home if live_ml_home!="N/A" else None
    edge_away = moneyline_to_probability(live_ml_away)-prob_away if live_ml_away!="N/A" else None
    is_value_home = edge_home>0.05 if edge_home else False
    is_value_away = edge_away>0.05 if edge_away else False

    render_matchup_card(team_home, team_away, TEAM_LOGOS, odds_book,
                        prob_home, prob_away, predicted_spread,
                        predicted_ml_home, predicted_ml_away,
                        live_ml_home, live_ml_away,
                        live_spread_home, live_spread_away,
                        edge_home, edge_away,
                        is_value_home, is_value_away)

st.markdown("<div class='footer'>NFL Elo + Odds Dashboard &copy; 2025</div>", unsafe_allow_html=True)
