import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from difflib import get_close_matches
import requests

# --- CONFIG ---
BASE_ELO = 1500
K = 20
HOME_ADVANTAGE = 65
EXCEL_FILE = "games.xlsx"
HIST_SHEET = "games"
SCHEDULE_SHEET = "2025 schedule"
CSV_FILE = "odds.csv"

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

# --- VEGASINSIDER SCRAPER USING REQUESTS ---
VEGASINSIDER_URL = "https://www.vegasinsider.com/nfl/odds/las-vegas/"

def scrape_vegasinsider():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
    }
    resp = requests.get(VEGASINSIDER_URL, headers=headers)
    resp.raise_for_status()
    
    tables = pd.read_html(resp.text)
    df = tables[0]  # main odds table
    df = df.dropna(axis=1, how='all')

    col_lower = [c.lower() for c in df.columns]
    def find_col(keywords):
        for i, c in enumerate(col_lower):
            if any(k in c for k in keywords):
                return df.columns[i]
        return None

    team1_col = find_col(["team", "visitor"])
    team2_col = find_col(["team", "home"])
    spread1_col = find_col(["spread", "line"])
    spread2_col = find_col(["spread.1", "line.1"])
    ml1_col = find_col(["ml", "moneyline"])
    ml2_col = find_col(["ml.1", "moneyline.1"])

    if not all([team1_col, team2_col, spread1_col, spread2_col, ml1_col, ml2_col]):
        raise ValueError("Could not detect all required columns in VegasInsider table.")

    df_clean = pd.DataFrame({
        "Team1": df[team1_col],
        "Spread1": df[spread1_col],
        "ML1": df[ml1_col],
        "Team2": df[team2_col],
        "Spread2": df[spread2_col],
        "ML2": df[ml2_col],
    })

    return df_clean

def process_and_save(df):
    df.to_csv(CSV_FILE, index=False)

def load_odds():
    try:
        df = pd.read_csv(CSV_FILE)
        odds_index = {}
        for _, row in df.iterrows():
            teams = [row['Team1'].lower(), row['Team2'].lower()]
            odds_index[frozenset(teams)] = {
                "moneyline": {teams[0]: row['ML1'], teams[1]: row['ML2']},
                "spread": {teams[0]: row['Spread1'], teams[1]: row['Spread2']},
                "bookmaker": "VegasInsider"
            }
        return odds_index
    except Exception as e:
        st.warning(f"Failed to load odds CSV: {e}")
        return {}

# --- HELPERS ---
def moneyline_to_probability(ml):
    try:
        ml = str(ml)
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

def probability_to_cover(prob_team, spread):
    win_prob_cover = 1 / (1 + np.exp(-0.23 * (-spread)))
    return win_prob_cover if prob_team > 0.5 else 1 - win_prob_cover

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

# --- Refresh CSV & load odds ---
try:
    df = scrape_vegasinsider()
    process_and_save(df)
except Exception as e:
    st.warning(f"Failed to scrape VegasInsider, using last saved CSV if available: {e}")

odds_index = load_odds()

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

# --- Render Cards ---
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

    cover_prob_t1 = probability_to_cover(p1, float(pred_spread_t1))
    cover_prob_t2 = probability_to_cover(p2, float(pred_spread_t2))

    match_key = fuzzy_find_team_in_odds(t1, odds_index.keys())
    ml_t1 = ml_t2 = sp_t1 = sp_t2 = "N/A"
    book = "N/A"
    implied_str_t1 = implied_str_t2 = "N/A"
    edge_t1 = edge_t2 = None

    if match_key:
        entry = odds_index[match_key]
        ml_t1, ml_t2 = entry['moneyline'].get(t1.lower(), "N/A"), entry['moneyline'].get(t2.lower(), "N/A")
        sp_t1, sp_t2 = entry['spread'].get(t1.lower(), "N/A"), entry['spread'].get(t2.lower(), "N/A")
        book = entry['bookmaker']

        implied_p1 = moneyline_to_probability(ml_t1)
        implied_p2 = moneyline_to_probability(ml_t2)
        implied_str_t1 = f"{implied_p1:.1%}" if implied_p1 is not None else "N/A"
        implied_str_t2 = f"{implied_p2:.1%}" if implied_p2 is not None else "N/A"
        edge_t1 = None if implied_p1 is None else p1 - implied_p1
        edge_t2 = None if implied_p2 is None else p2 - implied_p2

    cols = st.columns([1, 1])
    with cols[0]:
        logo1 = TEAM_LOGOS.get(t1.lower(), "")
        st.markdown(f"<div class='team-block'><img src='{logo1}' width='56'/>"
                    f"<div><div class='team-name'>{t1}</div>"
                    f"<div class='small-muted'>ML: {ml_t1} | Spread: {sp_t1} | Pred Spread: {pred_spread_t1:+.1f} | Cover Prob: {cover_prob_t1:.1%} | Implied: {implied_str_t1}</div></div></div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='prob-bar'><div class='prob-fill' style='width:{p1*100:.1f}%;background:{'#16a34a' if edge_t1 and edge_t1>0.05 else '#ef4444' if edge_t1 and edge_t1<-0.05 else '#3b82f6'}'></div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>{p1:.1%} win probability</div>", unsafe_allow_html=True)
        st.markdown(format_edge_badge(edge_t1), unsafe_allow_html=True)

    with cols[1]:
        logo2 = TEAM_LOGOS.get(t2.lower(), "")
        st.markdown(f"<div class='team-block' style='justify-content:flex-end'><div>"
                    f"<div class='team-name' style='text-align:right'>{t2}</div>"
                    f"<div class='small-muted' style='text-align:right'>ML: {ml_t2} | Spread: {sp_t2} | Pred Spread: {pred_spread_t2:+.1f} | Cover Prob: {cover_prob_t2:.1%} | Implied: {implied_str_t2}</div></div>"
                    f"<img src='{logo2}' width='56'/></div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='prob-bar'><div class='prob-fill' style='width:{p2*100:.1f}%;background:{'#16a34a' if edge_t2 and edge_t2>0.05 else '#ef4444' if edge_t2 and edge_t2<-0.05 else '#3b82f6'}'></div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted' style='text-align:right'>{p2:.1%} win probability</div>", unsafe_allow_html=True)
        st.markdown(format_edge_badge(edge_t2), unsafe_allow_html=True)

    st.markdown(f"<small>Bookmaker: {book}</small>")
    st.markdown("---")
