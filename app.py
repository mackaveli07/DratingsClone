import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from difflib import get_close_matches
import time
from bs4 import BeautifulSoup
import subprocess
import sys

# --- Ensure webdriver_manager is installed ---
try:
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium import webdriver
    from webdriver_manager.chrome import ChromeDriverManager
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "selenium", "webdriver-manager"])
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium import webdriver
    from webdriver_manager.chrome import ChromeDriverManager

# --- CONFIG ---
BASE_ELO = 1500
K = 20
HOME_ADVANTAGE = 65
EXCEL_FILE = "games.xlsx"
HIST_SHEET = "games"
SCHEDULE_SHEET = "2025 schedule"

TEAM_LOGOS = {
    "kansas city chiefs": "https://a.espncdn.com/i/teamlogos/nfl/500/kc.png",
    "san francisco 49ers": "https://a.espncdn.com/i/teamlogos/nfl/500/sf.png",
    "green bay packers": "https://a.espncdn.com/i/teamlogos/nfl/500/gb.png",
    "dallas cowboys": "https://a.espncdn.com/i/teamlogos/nfl/500/dal.png",
}

# --- ELO FUNCTIONS ---
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
    for _, games in grouped:
        for _, row in games.iterrows():
            update_ratings(elo_ratings, row.team1, row.team2, row.score1, row.score2, row.home_team)
    return dict(elo_ratings)

# --- HEADLESS DRIVER ---
def start_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

# --- SCRAPERS ---
def scrape_fanduel():
    driver = start_driver()
    driver.get("https://sportsbook.fanduel.com/navigation/nfl")
    time.sleep(3)
    soup = BeautifulSoup(driver.page_source, "lxml")
    driver.quit()
    odds_data = {}
    for event in soup.select("div.event"):
        teams = [t.get_text(strip=True) for t in event.select("span.participant-name")]
        mls = [p.get_text(strip=True) for p in event.select("span.odds")]
        spreads = [p.get_text(strip=True) for p in event.select("span.point-spread")]
        if len(teams) == 2:
            odds_data[frozenset([teams[0].lower(), teams[1].lower()])] = {
                "moneyline": {teams[0].lower(): mls[0], teams[1].lower(): mls[1]},
                "spread": {teams[0].lower(): spreads[0], teams[1].lower(): spreads[1]},
                "bookmaker": "FanDuel"
            }
    return odds_data

def scrape_draftkings():
    driver = start_driver()
    driver.get("https://sportsbook.draftkings.com/leagues/football/nfl")
    time.sleep(3)
    soup = BeautifulSoup(driver.page_source, "lxml")
    driver.quit()
    odds_data = {}
    for event in soup.select("div.event-cell"):
        teams = [t.get_text(strip=True) for t in event.select("div.event-cell__name")]
        mls = [p.get_text(strip=True) for p in event.select("span.sportsbook-odds")]
        spreads = [p.get_text(strip=True) for p in event.select("span.sportsbook-outcome-cell__line")]
        if len(teams) == 2:
            odds_data[frozenset([teams[0].lower(), teams[1].lower()])] = {
                "moneyline": {teams[0].lower(): mls[0], teams[1].lower(): mls[1]},
                "spread": {teams[0].lower(): spreads[0], teams[1].lower(): spreads[1]},
                "bookmaker": "DraftKings"
            }
    return odds_data

def scrape_betonline():
    driver = start_driver()
    driver.get("https://www.betonline.ag/sportsbook/football/nfl")
    time.sleep(3)
    soup = BeautifulSoup(driver.page_source, "lxml")
    driver.quit()
    odds_data = {}
    for event in soup.select("div.event"):
        teams = [t.get_text(strip=True) for t in event.select("span.team-name")]
        mls = [p.get_text(strip=True) for p in event.select("span.odds")]
        spreads = [p.get_text(strip=True) for p in event.select("span.spread")]
        if len(teams) == 2:
            odds_data[frozenset([teams[0].lower(), teams[1].lower()])] = {
                "moneyline": {teams[0].lower(): mls[0], teams[1].lower(): mls[1]},
                "spread": {teams[0].lower(): spreads[0], teams[1].lower(): spreads[1]},
                "bookmaker": "BetOnline"
            }
    return odds_data

# --- HELPERS ---
def moneyline_to_probability(ml):
    try:
        if isinstance(ml, str):
            if ml.startswith('+'):
                val = int(ml[1:])
                return 100.0 / (val + 100.0)
            if ml.startswith('-'):
                val = int(ml[1:])
                return val / (val + 100.0)
    except:
        return None
    return None

def probability_to_spread(prob, team_is_favorite=True):
    b = 0.23
    prob = max(min(prob, 0.999), 0.001)
    spread = np.log(prob / (1 - prob)) / b
    spread = round(spread * 2) / 2
    return spread if team_is_favorite else -spread

def format_edge_badge(edge):
    if edge is None: return ""
    if edge > 0.05:
        return f'<span style="color:green;font-weight:bold">VALUE +{edge:.1%}</span>'
    if edge < -0.05:
        return f'<span style="color:red;font-weight:bold">NEG {edge:.1%}</span>'
    return f'<span style="color:gray">{edge:.1%}</span>'

def fuzzy_find_team_in_odds(team_name, odds_index_keys):
    name = team_name.lower()
    for key in odds_index_keys:
        if name in key:
            return key
    matches = get_close_matches(name, [t for key in odds_index_keys for t in key], n=1, cutoff=0.6)
    if matches:
        for k in odds_index_keys:
            if matches[0] in k:
                return k
    return None

# --- UI ---
st.set_page_config(layout="wide", page_title="NFL Elo Betting Dashboard")
st.title("🏈 NFL Elo Betting Dashboard")

source = st.sidebar.selectbox("Select Odds Source", ["FanDuel", "DraftKings", "BetOnline"])
fetch_button = st.sidebar.button("Fetch Live Odds")

# Load Excel
try:
    hist_df = pd.read_excel(EXCEL_FILE, sheet_name=HIST_SHEET)
    sched_df = pd.read_excel(EXCEL_FILE, sheet_name=SCHEDULE_SHEET)
except FileNotFoundError:
    st.error(f"Excel file '{EXCEL_FILE}' not found.")
    st.stop()

ratings = run_elo_pipeline(hist_df)
weeks = sorted(sched_df['week'].dropna().unique().astype(int).tolist())
week_choice = st.selectbox("Select Week", weeks, index=len(weeks)-1)
week_games = sched_df[sched_df['week'] == week_choice]

# Fetch odds on-demand
odds_index = {}
if fetch_button:
    st.info("Fetching live odds, please wait...")
    try:
        if source == "FanDuel":
            odds_index = scrape_fanduel()
        elif source == "DraftKings":
            odds_index = scrape_draftkings()
        else:
            odds_index = scrape_betonline()
    except Exception as e:
        st.error(f"Could not fetch odds from {source}: {e}")
        odds_index = {}
    st.success("Odds fetched successfully!")

# --- Render matchup cards ---
CARD_CSS = """
<style>
.matchup-card{border-radius:10px;padding:12px;margin-bottom:12px;box-shadow:0 4px 18px rgba(0,0,0,0.08);background:linear-gradient(180deg,#fff,#f9f9f9);}
.team-block{display:flex;align-items:center;gap:12px;}
.team-name{font-weight:700;font-size:18px;}
.small-muted{color:#6b7280;font-size:12px;}
.prob-bar{height:10px;border-radius:6px;overflow:hidden;background:#eee;margin-top:6px;}
.prob-fill{height:10px;}
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)

for _, row in week_games.iterrows():
    t1, t2 = row['team1'], row['team2']
    home = row.get('home_team', t1)

    r1 = ratings.get(t1, BASE_ELO) + (HOME_ADVANTAGE if home == t1 else 0)
    r2 = ratings.get(t2, BASE_ELO) + (HOME_ADVANTAGE if home == t2 else 0)
    p1, p2 = expected_score(r1, r2), 1 - expected_score(r1, r2)

    pred_spread_t1 = probability_to_spread(p1, p1 > p2)
    pred_spread_t2 = -pred_spread_t1

    match_key = fuzzy_find_team_in_odds(t1, odds_index.keys())
    ml_t1 = ml_t2 = sp_t1 = sp_t2 = "N/A"
    book = "N/A"
    if match_key:
        entry = odds_index[match_key]
        ml_t1, ml_t2 = entry['moneyline'].get(t1.lower(), "N/A"), entry['moneyline'].get(t2.lower(), "N/A")
        sp_t1, sp_t2 = entry['spread'].get(t1.lower(), "N/A"), entry['spread'].get(t2.lower(), "N/A")
        book = entry['bookmaker']

    implied_p1 = moneyline_to_probability(ml_t1)
    implied_p2 = moneyline_to_probability(ml_t2)
    edge_t1 = None if implied_p1 is None else p1 - implied_p1
    edge_t2 = None if implied_p2 is None else p2 - implied_p2

    cols = st.columns([1, 1])
    with cols[0]:
        logo1 = TEAM_LOGOS.get(t1.lower(), "")
        st.markdown(f"<div class='team-block'><img src='{logo1}' width='56'/>"
                    f"<div><div class='team-name'>{t1}</div>"
                    f"<div class='small-muted'>ML: {ml_t1} | Spread: {sp_t1} | Pred Spread: {pred_spread_t1:+.1f}</div></div></div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='prob-bar'><div class='prob-fill' style='width:{p1*100:.1f}%;background:{'#16a34a' if edge_t1 and edge_t1>0.05 else '#ef4444' if edge_t1 and edge_t1<-0.05 else '#3b82f6'}'></div></div>", unsafe_allow_html=True)
        st.markdown(format_edge_badge(edge_t1), unsafe_allow_html=True)

    with cols[1]:
        logo2 = TEAM_LOGOS.get(t2.lower(), "")
        st.markdown(f"<div class='team-block'><img src='{logo2}' width='56'/>"
                    f"<div><div class='team-name'>{t2}</div>"
                    f"<div class='small-muted'>ML: {ml_t2} | Spread: {sp_t2} | Pred Spread: {pred_spread_t2:+.1f}</div></div></div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='prob-bar'><div class='prob-fill' style='width:{p2*100:.1f}%;background:{'#16a34a' if edge_t2 and edge_t2>0.05 else '#ef4444' if edge_t2 and edge_t2<-0.05 else '#3b82f6'}'></div></div>", unsafe_allow_html=True)
        st.markdown(format_edge_badge(edge_t2), unsafe_allow_html=True)
