import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
import os, base64, requests, datetime, pytz

# ==========================
# Config & Constants
# ==========================
BASE_ELO = 1500
K = 20
HOME_ADVANTAGE = 65

EXCEL_FILE = "games.xlsx"
HIST_SHEET = "games"
SCHEDULE_SHEET = "2025 schedule"
PICKS_SHEET = "Picks"

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

# ==========================
# Helper Functions
# ==========================
def map_team_name(name):
    if not name:
        return "Unknown"
    name = str(name).strip()
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
        st.image(path, width=width)
    else:
        st.markdown(
            f"<div style='width:{width}px; height:{width}px; background:#e5e7eb; "
            f"display:flex; align-items:center; justify-content:center; border-radius:50%; "
            f"font-size:12px; color:#475569;'>{abbr or '?'}</div>",
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

# ==========================
# Elo System
# ==========================
def expected_score(r1, r2):
    return 1 / (1 + 10 ** ((r2 - r1) / 400))

def regress_preseason(elo_ratings, reg=0.65, base=BASE_ELO):
    for t in list(elo_ratings.keys()):
        elo_ratings[t] = base + reg * (elo_ratings[t] - base)

def update_ratings(elo_ratings, team1, team2, score1, score2, home_team):
    r1, r2 = elo_ratings[team1], elo_ratings[team2]
    # Home advantage
    if home_team == team1:
        r1 += HOME_ADVANTAGE
    elif home_team == team2:
        r2 += HOME_ADVANTAGE

    exp1 = expected_score(r1, r2)
    actual1 = 1 if score1 > score2 else 0
    margin = abs(score1 - score2) or 1
    mov_mult = np.log(margin + 1) * (2.2 / ((r1 - r2) * 0.001 + 2.2))

    elo_ratings[team1] += K * mov_mult * (actual1 - exp1)
    elo_ratings[team2] += K * mov_mult * ((1 - actual1) - expected_score(r2, r1))

def run_elo_pipeline(df):
    elo_ratings = defaultdict(lambda: BASE_ELO)
    if {"season", "week"} <= set(df.columns):
        df = df.sort_values(["season", "week"])
        for i, s in enumerate(df["season"].dropna().unique()):
            if i > 0:
                regress_preseason(elo_ratings)
            season_df = df[df["season"] == s]
            for _, row in season_df.iterrows():
                t1, t2 = map_team_name(row.get("team1")), map_team_name(row.get("team2"))
                home_team = map_team_name(row.get("home_team", t2))
                update_ratings(
                    elo_ratings,
                    t1, t2,
                    int(row.get("score1", 0) or 0),
                    int(row.get("score2", 0) or 0),
                    home_team
                )
    return dict(elo_ratings)

# ==========================
# ESPN Scoreboard Helpers
# ==========================
@st.cache_data(ttl=5)
def fetch_nfl_scores():
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    try:
        resp = requests.get(url, timeout=6)
    except Exception:
        return []
    if resp.status_code != 200:
        return []
    data = resp.json()
    games = []
    today = datetime.date.today().isoformat()
    for event in data.get("events", []):
        # Only today's games
        if (event.get("date", "").split("T")[0]) != today:
            continue
        comp = event.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue
        away = next((t for t in competitors if t.get("homeAway") == "away"), None)
        home = next((t for t in competitors if t.get("homeAway") == "home"), None)
        if not away or not home:
            continue
        status = comp.get("status", {})
        situation = comp.get("situation", {})
        info = {
            "quarter": f"Q{status.get('period', 'N/A')}",
            "clock": status.get("displayClock", ""),
            "possession": situation.get("possession", {}).get("displayName", ""),
            "shortDetail": status.get("type", {}).get("shortDetail", "")
        }
        games.append({"away": away, "home": home, "info": info})
    return games

# ==========================
# Injuries
# ==========================
ESPN_TEAM_IDS = {
    "ARI": 22, "ATL": 1, "BAL": 33, "BUF": 2, "CAR": 29, "CHI": 3, "CIN": 4, "CLE": 5, "DAL": 6,
    "DEN": 7, "DET": 8, "GB": 9, "HOU": 34, "IND": 11, "JAX": 30, "KC": 12, "LV": 13, "LAC": 24,
    "LA": 14, "MIA": 15, "MIN": 16, "NE": 17, "NO": 18, "NYG": 19, "NYJ": 20, "PHI": 21, "PIT": 23,
    "SF": 25, "SEA": 26, "TB": 27, "TEN": 10, "WAS": 28
}

def fetch_injuries_espn(team_abbr):
    team_id = ESPN_TEAM_IDS.get(team_abbr)
    if not team_id:
        return []
    url = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/{team_id}/injuries"
    try:
        r = requests.get(url, timeout=6)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []
    players = []
    for e in data.get("entries", []):
        players.append({
            "name": e.get("athlete", {}).get("displayName"),
            "position": e.get("position", {}).get("abbreviation"),
            "status": e.get("status", {}).get("type", "")
        })
    return players

def injury_adjustment(players):
    penalty = 0
    for p in players:
        s = (p.get("status") or "").lower()
        pos = (p.get("position") or "").upper()
        if pos == "QB" and s in ["out", "doubtful"]:
            penalty -= 50
        elif pos in ["RB", "WR", "TE"] and s in ["out", "doubtful"]:
            penalty -= 15
        elif s in ["out", "doubtful"]:
            penalty -= 10
    return penalty

# ==========================
# Weather
# ==========================
STADIUMS = {
    "Buffalo Bills": {"lat": 42.7738, "lon": -78.7868},
    "Green Bay Packers": {"lat": 44.5013, "lon": -88.0622},
    "Chicago Bears": {"lat": 41.8625, "lon": -87.6166},
    "Kansas City Chiefs": {"lat": 39.0490, "lon": -94.4840},
    "New England Patriots": {"lat": 42.0909, "lon": -71.2643},
    "Philadelphia Eagles": {"lat": 39.9008, "lon": -75.1675},
}

OWM_API_KEY = os.getenv("OWM_API_KEY", "")

def get_weather(team, kickoff_unix):
    if team not in STADIUMS or not OWM_API_KEY:
        return None
    lat, lon = STADIUMS[team]["lat"], STADIUMS[team]["lon"]
    url = (
        f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}"
        f"&appid={OWM_API_KEY}&units=imperial"
    )
    try:
        resp = requests.get(url, timeout=6)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None
    forecasts = data.get("list", [])
    if not forecasts:
        return None
    closest = min(forecasts, key=lambda x: abs(int(x.get("dt", 0)) - int(kickoff_unix)))
    dt_diff = abs(int(closest.get("dt", 0)) - int(kickoff_unix))
    if dt_diff > 432000:  # > 5 days away
        return None
    return {
        "temp": closest["main"]["temp"],
        "wind_speed": closest["wind"]["speed"],
        "condition": closest["weather"][0]["main"],
    }

def weather_adjustment(weather):
    pen = 0
    if weather["wind_speed"] > 20:
        pen -= 2
    if (weather["condition"] or "").lower() in ["rain", "snow"]:
        pen -= 3
    if weather["temp"] < 25:
        pen -= 1
    return pen

def default_kickoff_unix(game_date):
    if isinstance(game_date, str):
        try:
            game_date = datetime.datetime.strptime(game_date, "%Y-%m-%d")
        except Exception:
            return 0
    elif not isinstance(game_date, datetime.datetime):
        return 0
    est = pytz.timezone("US/Eastern")
    kickoff = est.localize(datetime.datetime(game_date.year, game_date.month, game_date.day, 13, 0, 0))
    return int(kickoff.timestamp())

# ==========================
# UI Headers
# ==========================
def load_local_logo(path="NFL.png"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

NFL_LOGO_B64 = load_local_logo()

def nfl_header(title):
    logo_html = f"<img src='data:image/png;base64,{NFL_LOGO_B64}' height='60'>" if NFL_LOGO_B64 else ""
    st.markdown(
        r"""
        <div style='background: linear-gradient(90deg, #013369, #d50a0a); 
                    padding: 20px; border-radius: 15px; text-align:center; display:flex; 
                    align-items:center; justify-content:center; gap:16px;'>
            """ + logo_html + f"""
            <h1 style='color:white; margin:0; font-size:42px;'>{title}</h1>
            {logo_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

def nfl_subheader(text, icon="📊"):
    logo_html = (
        f"<img src='data:image/png;base64,{NFL_LOGO_B64}' height='32' style='margin-right:8px;'/>"
        if NFL_LOGO_B64 else ""
    )
    st.markdown(
        r"""
        <div style='background: linear-gradient(90deg, #d50a0a, #013369); 
                    padding: 12px; border-radius: 12px; text-align:center; display:flex; 
                    align-items:center; justify-content:center; gap:10px;'>
        """ + logo_html + f"""
            <h2 style='color:white; margin:0;'>{icon} {text}</h2>
            {logo_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==========================
# App
# ==========================
st.set_page_config(page_title="NFL Elo Projections", layout="wide")
nfl_header("NFL Elo Projections")

# Load data
try:
    hist_df = pd.read_excel(EXCEL_FILE, sheet_name=HIST_SHEET)
    sched_df = pd.read_excel(EXCEL_FILE, sheet_name=SCHEDULE_SHEET)
except Exception as e:
    st.error(f"Error loading Excel: {e}")
    st.stop()

# Season scoring baselines (optional)
NFL_AVG_TOTALS = {}
alpha = 50
if {"score1", "score2", "season"} <= set(hist_df.columns):
    hist_df["total_points"] = (hist_df["score1"].fillna(0) + hist_df["score2"].fillna(0)).astype(float)
    overall_avg = float(hist_df["total_points"].mean()) if len(hist_df) else 44.0
    grouped = hist_df.groupby("season")["total_points"].agg(["mean", "count"]).reset_index()
    for _, r in grouped.iterrows():
        NFL_AVG_TOTALS[int(r["season"]) if pd.notnull(r["season"]) else -1] = (
            (float(r["mean"]) * float(r["count"])) + overall_avg * alpha
        ) / (float(r["count"]) + alpha)

# Elo ratings from history
ratings = run_elo_pipeline(hist_df)
HOME_COL, AWAY_COL = "team2", "team1"  # home is team2 in your sheets

# Tabs
tabs = st.tabs(["Matchups", "Power Rankings", "Pick Winners", "Scoreboard"])

# ------------------
# Matchups
# ------------------
with tabs[0]:
    available_weeks = (
        sched_df['week']
        .dropna()
        .apply(lambda x: int(x) if str(x).isdigit() else None)
        .dropna()
        .unique()
        .tolist()
    )
    available_weeks = sorted(set(available_weeks))
    selected_week = st.selectbox("Select Week", available_weeks, index=max(0, len(available_weeks) - 1), key="week_matchups")
    week_games = sched_df[sched_df['week'].astype(str) == str(selected_week)]

    for _, row in week_games.iterrows():
        team_home, team_away = map_team_name(row.get(HOME_COL)), map_team_name(row.get(AWAY_COL))
        abbr_home, abbr_away = get_abbr(team_home), get_abbr(team_away)
        elo_home = ratings.get(team_home, BASE_ELO)
        elo_away = ratings.get(team_away, BASE_ELO)

        # Add home advantage for win prob calculation
        prob_home = expected_score(elo_home + HOME_ADVANTAGE, elo_away)

        st.markdown(
            "<div style='background: rgba(255,255,255,0.12); backdrop-filter: blur(14px); "
            "border-radius: 24px; padding: 25px; margin: 22px 0; box-shadow: 0 8px 25px rgba(0,0,0,0.25);'>",
            unsafe_allow_html=True,
        )

        col1, col_mid, col2 = st.columns([2, 3, 2])
        with col1:
            safe_logo(abbr_away, 110)
            st.markdown(f"<div style='text-align:center'>{neon_text(team_away, abbr_away, 28)}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center; font-size:14px; opacity:.85'>ELO: {elo_away:.0f}</div>", unsafe_allow_html=True)
        with col_mid:
            st.markdown(
                f"""
                <div style='text-align:center'>
                    <div style='font-size:34px; margin-top:18px'>@</div>
                    <div style='margin-top:8px; font-size:14px;'>Home Win Prob: <strong>{prob_home*100:0.1f}%</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            safe_logo(abbr_home, 110)
            st.markdown(f"<div style='text-align:center'>{neon_text(team_home, abbr_home, 28)}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center; font-size:14px; opacity:.85'>ELO: {elo_home:.0f}</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ------------------
# Power Rankings
# ------------------
with tabs[1]:
    nfl_subheader("Power Rankings (Elo)")
    if ratings:
        pr = (
            pd.DataFrame({"team": list(ratings.keys()), "elo": list(ratings.values())})
            .sort_values("elo", ascending=False)
            .reset_index(drop=True)
        )
        pr.index = pr.index + 1
        pr["abbr"] = pr["team"].apply(get_abbr)
        pr = pr[["abbr", "team", "elo"]]
        st.dataframe(
            pr.style.format({"elo": "{:.0f}"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No rating data available.")

# ------------------
# Pick Winners
# ------------------
with tabs[2]:
    nfl_subheader("Weekly Pick'em", "📝")
    available_weeks = (
        sched_df['week']
        .dropna()
        .apply(lambda x: int(x) if str(x).isdigit() else None)
        .dropna()
        .unique()
        .tolist()
    )
    available_weeks = sorted(set(available_weeks))
    selected_week = st.selectbox("Week", available_weeks, index=max(0, len(available_weeks) - 1), key="week_picks")
    week_games = sched_df[sched_df['week'].astype(str) == str(selected_week)]

    picks = {}
    for _, row in week_games.iterrows():
        t_home, t_away = map_team_name(row.get(HOME_COL)), map_team_name(row.get(AWAY_COL))
        a_home, a_away = get_abbr(t_home), get_abbr(t_away)
        st.markdown("<div style='background:rgba(255,255,255,0.08); border-radius:18px; padding:16px; margin:12px 0;'>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([3, 2, 3])
        with c1:
            safe_logo(a_away, 72)
            st.markdown(neon_text(t_away, a_away, 20), unsafe_allow_html=True)
        with c2:
            st.markdown("<div style='text-align:center; padding-top:22px'>Your Pick ➡️</div>", unsafe_allow_html=True)
        with c3:
            safe_logo(a_home, 72)
            st.markdown(neon_text(t_home, a_home, 20), unsafe_allow_html=True)
        choice = st.radio("", [t_away, t_home], horizontal=True, key=f"pick_{a_away}_at_{a_home}")
        picks[f"{a_away} @ {a_home}"] = choice
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Show My Picks Summary"):
        summary = pd.DataFrame([{"Matchup": k, "Pick": v} for k, v in picks.items()])
        st.dataframe(summary, use_container_width=True, hide_index=True)

# ------------------
# Scoreboard
# ------------------
with tabs[3]:
    nfl_subheader("Live NFL Scoreboard", "🏟️")
    games = fetch_nfl_scores()
    if not games:
        st.info("No NFL games today or unable to fetch the live scoreboard.")
    for game in games:
        away, home, info = game["away"], game["home"], game["info"]
        a_abbr = away.get("team", {}).get("abbreviation", "")
        h_abbr = home.get("team", {}).get("abbreviation", "")
        a_name = away.get("team", {}).get("displayName", a_abbr)
        h_name = home.get("team", {}).get("displayName", h_abbr)
        a_logo = away.get("team", {}).get("logo") or away.get("team", {}).get("logos", [{}])[0].get("href")
        h_logo = home.get("team", {}).get("logo") or home.get("team", {}).get("logos", [{}])[0].get("href")
        a_score = away.get("score", "0")
        h_score = home.get("score", "0")

        col1, col2, col3 = st.columns([3, 2, 3])
        with col1:
            st.markdown(
                f"""
                <div style='background: linear-gradient(135deg, #013369, #d50a0a); border-radius: 12px; padding: 12px; text-align:center;'>
                    {neon_text(a_name, a_abbr, 26)}
                    {f"<img src='{a_logo}' width='96'/>" if a_logo else ''}
                    <p style='font-size: 36px; margin: 10px 0; color:white'>{a_score}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div style='text-align:center;'>
                    <p><strong>{info.get('shortDetail') or (info.get('quarter') + ' ' + info.get('clock'))}</strong></p>
                    <p>Possession: {info.get('possession') or '-'}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""
                <div style='background: linear-gradient(135deg, #d50a0a, #013369); border-radius: 12px; padding: 12px; text-align:center;'>
                    {neon_text(h_name, h_abbr, 26)}
                    {f"<img src='{h_logo}' width='96'/>" if h_logo else ''}
                    <p style='font-size: 36px; margin: 10px 0; color:white'>{h_score}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
