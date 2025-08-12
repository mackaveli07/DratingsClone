import streamlit as st
import pandas as pd
import numpy as np
import requests
from collections import defaultdict
from difflib import get_close_matches

# Optional faster fuzzy matching
try:
    import Levenshtein  # type: ignore
    has_lev = True
except Exception:
    has_lev = False

### ---------- CONFIG ----------
API_KEY = "YOUR_API_KEY_HERE"  # <-- Replace with your TheOddsAPI key
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
    "kansas city chiefs": "https://a.espncdn.com/i/teamlogos/nfl/500/kc.png",
    "san francisco 49ers": "https://a.espncdn.com/i/teamlogos/nfl/500/sf.png",
    "green bay packers": "https://a.espncdn.com/i/teamlogos/nfl/500/gb.png",
    "dallas cowboys": "https://a.espncdn.com/i/teamlogos/nfl/500/dal.png",
    # add more teams here...
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
    if not {"season", "week", "team1", "team2", "score1", "score2", "home_team"}.issubset(df.columns):
        raise ValueError("Historical games sheet missing required columns.")
    grouped = df.groupby(["season", "week"])
    for (season, week), games in grouped:
        for _, row in games.iterrows():
            update_ratings(elo_ratings, row.team1, row.team2, row.score1, row.score2, row.home_team)
    return dict(elo_ratings)

### ---------- ODDS API ----------
@st.cache_data(ttl=30)
def get_theoddsapi_odds(api_key):
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
        s = str(ml)
        if s.startswith('+'):
            val = int(s.replace('+',''))
            return 100.0 / (val + 100.0)
        if s.startswith('-'):
            val = int(s.replace('-',''))
            return val / (val + 100.0)
        val = int(s)
        if val > 0:
            return 100.0 / (val + 100.0)
        else:
            val = abs(val)
            return val / (val + 100.0)
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
        return f'<div style="display:inline-block;padding:4px 8px;border-radius:6px;background:#d4f9d4;color:#0b6623;font-weight:700">VALUE +{edge:.1%}</div>'
    if edge < -0.05:
        return f'<div style="display:inline-block;padding:4px 8px;border-radius:6px;background:#ffd6d6;color:#8b0000;font-weight:700">NEG -{edge:.1%}</div>'
    return f'<div style="display:inline-block;padding:4px 8px;border-radius:6px;background:#efefef;color:#333;font-weight:700">Close {edge:.1%}</div>'

def fuzzy_find_team_in_odds(team_name, odds_index_keys):
    name = team_name.lower()
    for key in odds_index_keys:
        for tk in key:
            if name == tk:
                return key
    for key in odds_index_keys:
        combined = " ".join(list(key))
        if name in combined:
            return key
    candidates = []
    for key in odds_index_keys:
        for tk in key:
            candidates.append(tk)
    candidates = list(set(candidates))
    if not candidates:
        return None
    if has_lev:
        best = max(candidates, key=lambda c: Levenshtein.ratio(c, name))
        for k in odds_index_keys:
            if best in k:
                return k
    else:
        matches = get_close_matches(name, candidates, n=1, cutoff=0.6)
        if matches:
            best = matches[0]
            for k in odds_index_keys:
                if best in k:
                    return k
    return None

### ---------- CARD CSS & RENDER ----------
CARD_CSS = """
<style>
.matchup-card{border-radius:10px;padding:12px;margin-bottom:12px;box-shadow:0 4px 18px rgba(0,0,0,0.08);background:linear-gradient(180deg, rgba(255,255,255,0.98), rgba(250,250,250,0.98));}
.team-block{display:flex;align-items:center;gap:12px;}
.team-name{font-weight:700;font-size:18px;}
.small-muted{color:#6b7280;font-size:12px;}
.prob-bar{height:10px;border-radius:6px;overflow:hidden;background:#eee;margin-top:6px;}
.prob-fill{height:10px;}
.ml-badge{font-weight:700;padding:6px 8px;border-radius:6px;background:#f3f4f6;}
.bookmaker{font-size:11px;color:#666;margin-left:8px;}
.value-badge{display:inline-block;padding:4px 8px;border-radius:6px;background:#d4f9d4;color:#0b6623;font-weight:700;margin-left:6px;}
</style>
"""

def render_matchup_card(team_home, team_away, logos, odds_book,
                        prob_home, prob_away, predicted_spread,
                        predicted_ml_home, predicted_ml_away,
                        live_ml_home, live_ml_away,
                        live_spread_home, live_spread_away,
                        edge_home=None, edge_away=None,
                        is_value_home=False, is_value_away=False):
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    st.markdown("<div class='matchup-card'>", unsafe_allow_html=True)

    cols = st.columns([1,1])
    # Left / Home (on right side as requested)
    with cols[1]:
        logo_url = logos.get(team_home.lower(), "")
        value_badge = '<span class="value-badge">VALUE</span>' if is_value_home else ""
        st.markdown(
            f"<div class='team-block' style='justify-content:flex-end'>"
            f"<div><div class='team-name' style='text-align:right'>{team_home} {value_badge}</div>"
            f"<div class='small-muted' style='text-align:right'>"
            f"Model ML: <span class='ml-badge'>{predicted_ml_home}</span> | "
            f"Live ML: <span class='ml-badge'>{live_ml_home}</span><br>"
            f"Model Spread: <strong>{-predicted_spread:.1f}</strong> | "
            f"Live Spread: <strong>{-float(live_spread_home) if live_spread_home != 'N/A' else 'N/A'}</strong>"
            f"</div></div> "
            f"<img src='{logo_url}' width='56' style='border-radius:6px'/></div>",
            unsafe_allow_html=True)

        pct = prob_home if prob_home is not None else 0.5
        fill_color = "#16a34a" if edge_home and edge_home > 0.05 else ("#ef4444" if edge_home and edge_home < -0.05 else "#3b82f6")
        st.markdown(f"<div class='prob-bar'><div class='prob-fill' style='width:{pct*100:.1f}%; background:{fill_color}'></div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted' style='text-align:right'>{(prob_home*100):.1f}% win probability</div>" if prob_home is not None else "", unsafe_allow_html=True)

    # Right / Away (on left side)
    with cols[0]:
        logo_url2 = logos.get(team_away.lower(), "")
        value_badge2 = '<span class="value-badge">VALUE</span>' if is_value_away else ""
        st.markdown(
            f"<div class='team-block'>"
            f"<img src='{logo_url2}' width='56' style='border-radius:6px'/> "
            f"<div><div class='team-name'>{team_away} {value_badge2}</div>"
            f"<div class='small-muted'>"
            f"Model ML: <span class='ml-badge'>{predicted_ml_away}</span> | "
            f"Live ML: <span class='ml-badge'>{live_ml_away}</span><br>"
            f"Model Spread: <strong>{predicted_spread:.1f}</strong> | "
            f"Live Spread: <strong>{live_spread_away}</strong>"
            f"</div></div></div>",
            unsafe_allow_html=True)

        pct2 = prob_away if prob_away is not None else 0.5
        fill_color2 = "#16a34a" if edge_away and edge_away > 0.05 else ("#ef4444" if edge_away and edge_away < -0.05 else "#3b82f6")
        st.markdown(f"<div class='prob-bar'><div class='prob-fill' style='width:{pct2*100:.1f}%; background:{fill_color2}'></div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>{(prob_away*100):.1f}% win probability</div>" if prob_away is not None else "", unsafe_allow_html=True)

    st.markdown(f"<div style='margin-top:8px'><strong>Predicted Spread:</strong> {predicted_spread:+.1f} &nbsp;&nbsp; <span class='bookmaker'>Bookmaker: {odds_book}</span></div>", unsafe_allow_html=True)

    edge_html = ""
    if edge_home is not None:
        edge_html += f"Home edge: {format_edge_badge(edge_home)}&nbsp;&nbsp;"
    if edge_away is not None:
        edge_html += f"Away edge: {format_edge_badge(edge_away)}"
    if edge_html:
        st.markdown(f"<div style='margin-top:8px'>{edge_html}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("")  # spacer

### ---------- APP LAYOUT ----------
st.set_page_config(page_title="NFL Elo + TheOddsAPI — Matchup Cards", layout="wide")
st.title("🏈 NFL Elo Betting Dashboard (Matchup Cards)")
st.caption("Elo predictions + live odds from TheOddsAPI — value bets highlighted")

try:
    hist_df = pd.read_excel(EXCEL_FILE, sheet_name=HIST_SHEET)
    sched_df = pd.read_excel(EXCEL_FILE, sheet_name=SCHEDULE_SHEET)
except FileNotFoundError:
    st.error(f"Excel file '{EXCEL_FILE}' not found. Put it in the app folder or update EXCEL_FILE path.")
    st.stop()
except Exception as e:
    st.error(f"Error reading Excel file: {e}")
    st.stop()

if 'week' not in sched_df.columns:
    st.error("Schedule sheet must contain a 'week' column.")
    st.stop()

try:
    ratings = run_elo_pipeline(hist_df)
except Exception as e:
    st.error(f"Error computing Elo: {e}")
    st.stop()

st.sidebar.header("Controls")
use_api = st.sidebar.checkbox("Fetch live odds from TheOddsAPI", value=True)
api_key_input = st.sidebar.text_input("TheOddsAPI key (overrides config)", value="")
if api_key_input.strip():
    API_KEY = api_key_input.strip()
prefer_book = st.sidebar.text_input("Prefer bookmaker key (optional)", value="")
if prefer_book.strip():
    BOOKMAKER_PREFERENCE = prefer_book.strip()

available_weeks = sorted(sched_df['week'].dropna().unique().astype(int).tolist())
if not available_weeks:
    st.error("No weeks found in schedule sheet.")
    st.stop()

default_index = len(available_weeks) - 1 if len(available_weeks) > 0 else 0
selected_week = st.selectbox("Select Week to view", available_weeks, index=default_index)

st.markdown("---")

week_games = sched_df[sched_df['week'] == selected_week].copy()
if week_games.empty:
    st.info(f"No games found for week {selected_week}.")
else:
    odds_index = {}
    if use_api:
        try:
            api_data = get_theoddsapi_odds(API_KEY)
            odds_index = parse_odds_data(api_data)
        except Exception as e:
            st.error(f"Could not fetch odds: {e}")
            odds_index = {}

    # Collect value bets to display in summary table
    value_bets_list = []

    for _, row in week_games.iterrows():
        team1 = row['team1']
        team2 = row['team2']
        home_team = row.get('home_team', team1)

        r1 = ratings.get(team1, BASE_ELO) + (HOME_ADVANTAGE if home_team == team1 else 0)
        r2 = ratings.get(team2, BASE_ELO) + (HOME_ADVANTAGE if home_team == team2 else 0)
        prob1 = expected_score(r1, r2)
        prob2 = 1 - prob1

        predicted_spread = probability_to_spread(prob1, team_is_favorite=(prob1 > prob2))

        predicted_ml_team1 = probability_to_moneyline(prob1)
        predicted_ml_team2 = probability_to_moneyline(prob2)

        live_ml_team1 = live_ml_team2 = live_spread_team1 = live_spread_team2 = "N/A"
        bookmaker_title = "N/A"

        if odds_index:
            match_key = fuzzy_find_team_in_odds(team1, odds_index.keys())
            if match_key is None:
                match_key = fuzzy_find_team_in_odds(team2, odds_index.keys())
            if match_key:
                entry = odds_index.get(match_key, {})
                bookmaker_title = entry.get("bookmaker", "N/A")
                ml = entry.get("moneyline", {}) or {}
                sp = entry.get("spread", {}) or {}

                live_ml_team1 = ml.get(team1.lower(), next(iter(ml.values()), "N/A"))
                live_ml_team2 = ml.get(team2.lower(), next(iter(ml.values()), "N/A"))
                live_spread_team1 = sp.get(team1.lower(), next(iter(sp.values()), "N/A"))
                live_spread_team2 = sp.get(team2.lower(), next(iter(sp.values()), "N/A"))

        # Compute value edges
        implied_home_prob = moneyline_to_probability(live_ml_team1)
        implied_away_prob = moneyline_to_probability(live_ml_team2)

        edge_home = (prob1 - implied_home_prob) if implied_home_prob is not None else None
        edge_away = (prob2 - implied_away_prob) if implied_away_prob is not None else None

        is_value_home = edge_home is not None and edge_home > 0.05
        is_value_away = edge_away is not None and edge_away > 0.05

        if is_value_home:
            value_bets_list.append({
                "Week": selected_week,
                "Team": team1,
                "Opponent": team2,
                "Edge (%)": f"{edge_home:.1%}",
                "Market ML": live_ml_team1,
                "Model Prob": f"{prob1:.2f}",
                "Bookmaker": bookmaker_title,
                "Side": "Home"
            })
        if is_value_away:
            value_bets_list.append({
                "Week": selected_week,
                "Team": team2,
                "Opponent": team1,
                "Edge (%)": f"{edge_away:.1%}",
                "Market ML": live_ml_team2,
                "Model Prob": f"{prob2:.2f}",
                "Bookmaker": bookmaker_title,
                "Side": "Away"
            })

        render_matchup_card(
            team_home=team1,
            team_away=team2,
            logos=TEAM_LOGOS,
            odds_book=bookmaker_title,
            prob_home=prob1,
            prob_away=prob2,
            predicted_spread=predicted_spread,
            predicted_ml_home=predicted_ml_team1,
            predicted_ml_away=predicted_ml_team2,
            live_ml_home=live_ml_team1,
            live_ml_away=live_ml_team2,
            live_spread_home=live_spread_team1,
            live_spread_away=live_spread_team2,
            edge_home=edge_home,
            edge_away=edge_away,
            is_value_home=is_value_home,
            is_value_away=is_value_away,
        )

    # Display summary of value bets
    if value_bets_list:
        st.markdown("---")
        st.subheader("💰 Value Bets Summary")
        df_val = pd.DataFrame(value_bets_list)
        st.dataframe(df_val, use_container_width=True)
    else:
        st.info("No value bets found for this week.")


