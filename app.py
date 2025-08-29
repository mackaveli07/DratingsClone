# NFL Elo Projections App — Full Rebuilt with Neon Scoreboard
import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
import os, base64, requests, datetime, pytz

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

TEAM_COLORS = {
    "ARI": "#97233F", "ATL": "#A71930", "BAL": "#241773", "BUF": "#00338D",
    "CAR": "#0085CA", "CHI": "#0B162A", "CIN": "#FB4F14", "CLE": "#311D00",
    "DAL": "#003594", "DEN": "#FB4F14", "DET": "#0076B6", "GB": "#203731",
    "HOU": "#03202F", "IND": "#002C5F", "JAX": "#006778", "KC": "#E31837",
    "LV": "#000000", "LAC": "#0080C6", "LA": "#003594", "MIA": "#008E97",
    "MIN": "#4F2683", "NE": "#002244", "NO": "#D3BC8D", "NYG": "#0B2265",
    "NYJ": "#125740", "PHI": "#004C54", "PIT": "#FFB612", "SF": "#AA0000",
    "SEA": "#002244", "TB": "#D50A0A", "TEN": "#0C2340", "WAS": "#5A1414"
}

TEAM_NAME_FIXES = {
    "Clevland Browns": "Cleveland Browns",
    "NY Jets": "New York Jets",
    "NY Giants": "New York Giants",
    "Jags": "Jacksonville Jaguars",
}

### ---------- HELPERS ----------
def map_team_name(name):
    if not name:
        return "Unknown"
    name = str(name).strip()
    if name in TEAM_NAME_FIXES:
        name = TEAM_NAME_FIXES[name]
    if name.upper() in NFL_FULL_NAMES:
        return NFL_FULL_NAMES[name.upper()]
    for full in NFL_FULL_NAMES.values():
        if name.lower() == full.lower():
            return full
    return name

def get_abbr(team_full):
    for abbr, full in NFL_FULL_NAMES.items():
        if full == team_full:
            return abbr
    return None

def safe_logo(abbr, width=64):
    path = f"Logos/{abbr}.png"
    if abbr and os.path.exists(path):
        try:
            st.image(path, width=width)
        except Exception:
            st.markdown(
                f"<div style='width:{width}px; height:{width}px; background:#222; "
                f"display:flex; align-items:center; justify-content:center; border-radius:50%; "
                f"font-size:12px; color:#aaa;'>{abbr or '?'}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f"<div style='width:{width}px; height:{width}px; background:#222; "
            f"display:flex; align-items:center; justify-content:center; border-radius:50%; "
            f"font-size:12px; color:#aaa;'>{abbr or '?'}</div>",
            unsafe_allow_html=True,
        )

def neon_text(text, abbr, size=24):
    color = TEAM_COLORS.get(abbr, "#39ff14")
    return f"""
    <span style="
        color: {color};
        font-size: {size}px;
        font-weight: bold;
        text-shadow:
            0 0 5px {color},
            0 0 10px {color},
            0 0 20px {color},
            0 0 40px {color},
            0 0 80px {color};
    ">{text}</span>
    """

### ---------- ELO CALC ----------
def expected_score(r1, r2):
    return 1 / (1 + 10 ** ((r2 - r1) / 400))

def regress_preseason(elo_ratings, reg=0.65, base=BASE_ELO):
    for t in list(elo_ratings.keys()):
        elo_ratings[t] = base + reg * (elo_ratings[t] - base)

def update_ratings(elo_ratings, team1, team2, score1, score2, home_team):
    r1, r2 = elo_ratings[team1], elo_ratings[team2]
    if home_team == team1:
        r1 += HOME_ADVANTAGE
    elif home_team == team2:
        r2 += HOME_ADVANTAGE
    expected1 = expected_score(r1, r2)
    actual1 = 1 if (score1 or 0) > (score2 or 0) else 0
    margin = abs((score1 or 0) - (score2 or 0)) or 1
    mov_mult = np.log(margin + 1) * (2.2 / ((r1 - r2) * 0.001 + 2.2))
    elo_ratings[team1] += K * mov_mult * (actual1 - expected1)
    elo_ratings[team2] += K * mov_mult * ((1 - actual1) - expected_score(r2, r1))

def run_elo_pipeline(df):
    elo_ratings = defaultdict(lambda: BASE_ELO)
    if {"season","week"} <= set(df.columns):
        df = df.sort_values(["season","week"])
        for i, s in enumerate(df["season"].dropna().unique()):
            if i > 0:
                regress_preseason(elo_ratings)
            for _, row in df[df["season"]==s].iterrows():
                t1 = map_team_name(row.get("team1"))
                t2 = map_team_name(row.get("team2"))
                home = map_team_name(row.get("home_team", t2))
                update_ratings(
                    elo_ratings,
                    t1, t2,
                    row.get("score1", 0) or 0,
                    row.get("score2", 0) or 0,
                    home
                )
    return dict(elo_ratings)

### ---------- STREAMLIT CONFIG ----------
st.set_page_config(page_title="NFL Elo Projections", layout="wide")
st.markdown("<h1 style='text-align:center; color:white; text-shadow:0 0 10px #d50a0a,0 0 20px #013369;'>NFL Elo Projections</h1>", unsafe_allow_html=True)

# Load data
try:
    hist_df = pd.read_excel(EXCEL_FILE, sheet_name=HIST_SHEET)
    sched_df = pd.read_excel(EXCEL_FILE, sheet_name=SCHEDULE_SHEET)
except Exception as e:
    st.error(f"Error loading Excel: {e}")
    st.stop()

ratings = run_elo_pipeline(hist_df)
HOME_COL, AWAY_COL = "team2", "team1"

tabs = st.tabs(["Matchups", "Power Rankings", "Pick Winners", "Scoreboard"])

# --- Matchups Tab ---
with tabs[0]:
    week_series_num = pd.to_numeric(sched_df.get("week"), errors="coerce")
    available_weeks = sorted(set(week_series_num.dropna().astype(int).tolist()))
    if not available_weeks:
        st.warning("No valid weeks found in schedule.")
    selected_week = st.selectbox(
        "Select Week",
        options=available_weeks,
        index=max(0, len(available_weeks)-1),
        key="week_matchups"
    )
    mask = (week_series_num == selected_week)
    week_games = sched_df.loc[mask.fillna(False)]

    for _, row in week_games.iterrows():
        team_home = map_team_name(row.get(HOME_COL))
        team_away = map_team_name(row.get(AWAY_COL))
        abbr_home, abbr_away = get_abbr(team_home), get_abbr(team_away)

        adj_home = ratings.get(team_home, BASE_ELO)
        adj_away = ratings.get(team_away, BASE_ELO)
        win_prob_home = expected_score(adj_home + HOME_ADVANTAGE, adj_away)
        win_prob_away = 1 - win_prob_home

        total_pts = 44  # fallback average points
        proj_home = int(round(win_prob_home * total_pts))
        proj_away = int(round(win_prob_away * total_pts))

        st.markdown(
            f"""
            <div style='background: rgba(0,0,0,0.25); backdrop-filter: blur(14px);
                        border-radius: 24px; padding: 20px; margin: 16px 0;
                        box-shadow: 0 0 20px rgba(0,0,0,0.6); border:2px solid #d50a0a;
                        display:flex; justify-content:space-between; align-items:center;'>
                <div style='text-align:center; width:30%;'>
                    {neon_text(team_away, abbr_away, 28)}
                    <p style='color:white;'>Win %: {win_prob_away:.1%}</p>
                </div>
                <div style='text-align:center; width:20%;'>
                    <h2 style='color:white; margin:0;'>{proj_away} – {proj_home}</h2>
                    <p style='color:white; margin:0;'>Projected Score</p>
                </div>
                <div style='text-align:center; width:30%;'>
                    {neon_text(team_home, abbr_home, 28)}
                    <p style='color:white;'>Win %: {win_prob_home:.1%}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# --- Power Rankings Tab ---
with tabs[1]:
    st.markdown("<h2 style='text-align:center; color:white;'>📊 Elo Power Rankings</h2>", unsafe_allow_html=True)
    pr_df = pd.DataFrame(sorted(ratings.items(), key=lambda x: x[1], reverse=True), columns=["Team","Adj Elo"])
    pr_df.index = pr_df.index + 1
    pr_df.index.name = "Rank"

    for rank, row in pr_df.iterrows():
        team = row["Team"]
        abbr = get_abbr(team)
        elo_val = int(round(row["Adj Elo"]))
        st.markdown(
            f"""
            <div style='background: rgba(0,0,0,0.25); padding:12px; margin:6px 0;
                        border-radius:12px; display:flex; align-items:center;
                        justify-content:space-between; color:white;'>
                <div><strong>#{rank}</strong></div>
                <div>{neon_text(team, abbr, 20)}</div>
                <div><strong>{elo_val}</strong></div>
            </div>
            """,
            unsafe_allow_html=True
        )

# --- Pick Winners Tab ---
with tabs[2]:
    st.markdown("<h2 style='text-align:center; color:white;'>📝 Weekly Pick’em</h2>", unsafe_allow_html=True)
    week_series_num = pd.to_numeric(sched_df.get("week"), errors="coerce")
    available_weeks = sorted(set(week_series_num.dropna().astype(int).tolist()))
    week = st.selectbox("Select Week", available_weeks, key="week_picks")
    games = sched_df.loc[(week_series_num == week).fillna(False)]
    picks = {}
    for _, row in games.iterrows():
        t_home, t_away = map_team_name(row.get(HOME_COL)), map_team_name(row.get(AWAY_COL))
        abbr_home, abbr_away = get_abbr(t_home), get_abbr(t_away)
        st.markdown(
            f"<div style='background: rgba(0,0,0,0.2); padding:16px; border-radius:18px; margin:12px 0;'>",
            unsafe_allow_html=True
        )
        c1, c2, c3 = st.columns([3,2,3])
        with c1:
            safe_logo(abbr_away, 80)
            st.markdown(neon_text(t_away, abbr_away, 20), unsafe_allow_html=True)
        with c2:
            st.markdown("<h5 style='text-align:center; color:white;'>Your Pick ➡️</h5>", unsafe_allow_html=True)
        with c3:
            safe_logo(abbr_home, 80)
            st.markdown(neon_text(t_home, abbr_home, 20), unsafe_allow_html=True)
        choice = st.radio("", [t_away, t_home], horizontal=True, key=f"pick_{t_home}_{t_away}")
        picks[f"{t_away} @ {t_home}"] = choice
        st.markdown("</div>", unsafe_allow_html=True)

# --- Scoreboard Tab ---
with tabs[3]:
    st.markdown("<h2 style='text-align:center; color:white;'>🏟️ NFL Scoreboard</h2>", unsafe_allow_html=True)
    def fetch_nfl_scores():
        url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        try:
            resp = requests.get(url, timeout=8)
            data = resp.json()
        except:
            return []
        games = []
        for event in data.get("events", []):
            comp = event.get("competitions", [{}])[0]
            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                continue
            away = next((t for t in competitors if t.get("homeAway")=="away"), None)
            home = next((t for t in competitors if t.get("homeAway")=="home"), None)
            if not away or not home:
                continue
            status = comp.get("status", {})
            state = status.get("type", {}).get("state","")
            if state=="in":
                status_text = f"Q{status.get('period','')} {status.get('displayClock','')}"
            elif state=="post":
                status_text = "FINAL"
            else:
                status_text = "Scheduled"
            games.append({"away":away,"home":home,"state":state,"status":status_text,"competition":comp})
        return games

    games = fetch_nfl_scores()
    if not games:
        st.info("No NFL games today or scheduled.")

    for game in games:
        away, home = game["away"], game["home"]
        state = game.get("state", "pre")
        comp = game.get("competition")
        situation = comp.get("situation", {}) if comp else {}

        possession_id = situation.get("possession", {}).get("id")
        last_play = situation.get("lastPlay", {}).get("text", "")
        desc = situation.get("shortDownDistanceText")
        yard_line = situation.get("yardLine")
        drive_summary = f"{desc} on {yard_line}" if desc else None

        highlight_home = state=="post" and int(home.get("score",0)) > int(away.get("score",0))
        highlight_away = state=="post" and int(away.get("score",0)) > int(home.get("score",0))

        st.markdown(
            f"""
            <div style='background: rgba(0,0,0,0.25); backdrop-filter: blur(16px);
                        border-radius: 24px; padding: 20px; margin: 16px 0;
                        box-shadow: 0 0 20px rgba(0,0,0,0.6);
                        border: 3px solid;
                        border-image: linear-gradient(90deg, #d50a0a, #013369) 1;
                        color:white; display:flex; justify-content:space-between; align-items:center;'>
                <div style='text-align:center; width:30%;'>
                    <img src="{away['team']['logo']}" width="60">
                    {neon_text(away['team']['displayName'], get_abbr(away['team']['displayName']),24)}
                    <h1 style='color:{TEAM_COLORS.get(get_abbr(away['team']['displayName']),'#39ff14') if highlight_away else "#FFFFFF"};
                               text-shadow:0 0 12px {TEAM_COLORS.get(get_abbr(away['team']['displayName']),'#39ff14')};
                               font-size:48px;'>{'🏈 ' if str(away['team'].get('id'))==str(possession_id) else ''}{away.get('score','0')}</h1>
                </div>
                <div style='text-align:center; width:20%;'>
                    <h3 style='margin:10px 0;'>{game.get('status','')}</h3>
                    {f"<p>📋 {drive_summary}</p>" if drive_summary else ""}
                    {f"<p>📝 {last_play}</p>" if last_play else ""}
                </div>
                <div style='text-align:center; width:30%;'>
                    <img src="{home['team']['logo']}" width="60">
                    {neon_text(home['team']['displayName'], get_abbr(home['team']['displayName']),24)}
                    <h1 style='color:{TEAM_COLORS.get(get_abbr(home['team']['displayName']),'#39ff14') if highlight_home else "#FFFFFF"};
                               text-shadow:0 0 12px {TEAM_COLORS.get(get_abbr(home['team']['displayName']),'#39ff14')};
                               font-size:48px;'>{'🏈 ' if str(home['team'].get('id'))==str(possession_id) else ''}{home.get('score','0')}</h1>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

