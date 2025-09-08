# app.py
# NFL Elo Projections App — Full rebuild with Kelly & Prediction Tracking (no Articles tab)
import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
import os, base64, requests, datetime, pytz, math
from sklearn.metrics import brier_score_loss

### ---------- CONFIG ----------
BASE_ELO = 1500
K = 20
HOME_ADVANTAGE = 65

EXCEL_FILE = "games.xlsx"
HIST_SHEET = "games"
SCHEDULE_SHEET = "2025 schedule"

DEFAULT_BANKROLL = 50

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
                f"<div style='width:{width}px; height:{width}px; background:#e5e7eb; "
                f"display:flex; align-items:center; justify-content:center; border-radius:50%; "
                f"font-size:12px; color:#475569;'>{abbr or '?'}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f"<div style='width:{width}px; height:{width}px; background:#e5e7eb; "
            f"display:flex; align-items:center; justify-content:center; border-radius:50%; "
            f"font-size:12px; color:#475569;'>{abbr or '?'}</div>",
            unsafe_allow_html=True,
        )

def neon_text(text, abbr=None, size=24):
    color = TEAM_COLORS.get(abbr, "#39ff14") if abbr else "#39ff14"
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

# --- Set App Background ---
def set_background(image_path="Shield.png"):
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.85)),
                            url("data:image/png;base64,{b64}") no-repeat center center fixed;
                background-size: cover;
                color: white;
            }}
            /* Frosted glass cards */
            .card {{
                background: rgba(30,30,30,0.6);
                backdrop-filter: blur(14px);
                border-radius: 20px;
                padding: 20px;
                margin: 20px 0;
                box-shadow: 0 8px 25px rgba(0,0,0,0.4);
            }}
            @media (max-width: 768px) {{
                h1,h2,h3,h4,h5,h6 {{ font-size:90% !important; }}
                .card {{ padding:14px !important; margin:12px 0 !important; }}
                img {{ max-width:80px !important; height:auto !important; }}
                .stMarkdown p {{ font-size:14px !important; }}
            }}
            @media (max-width: 480px) {{
                .card {{ padding:10px !important; }}
                h1,h2,h3 {{ font-size:80% !important; }}
                img {{ max-width:60px !important; }}
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

set_background("Shield.png")

### ---------- ELO ----------
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

### ---------- SCOREBOARD HELPERS ----------
def _parse_utc_iso(ts: str):
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts.replace("Z", "+00:00")
        return datetime.datetime.fromisoformat(ts)
    except Exception:
        return None

def _fmt_sched_time(dt_utc, tz_name="US/Eastern"):
    if not dt_utc:
        return "Scheduled"
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=datetime.timezone.utc)
    tz = pytz.timezone(tz_name)
    dt_local = dt_utc.astimezone(tz)
    return "Scheduled " + dt_local.strftime("%a %I:%M %p").replace(" 0", " ")

@st.cache_data(ttl=30)
def fetch_nfl_scores():
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    try:
        resp = requests.get(url, timeout=8)
        if resp.status_code != 200:
            return []
        data = resp.json()
    except Exception:
        return []

    games = []
    for event in data.get("events", []):
        comp = event.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        away = next((t for t in competitors if t.get("homeAway") == "away"), None)
        home = next((t for t in competitors if t.get("homeAway") == "home"), None)
        if not away or not home:
            continue

        status = comp.get("status", {})
        stype = status.get("type", {})
        state = stype.get("state", "")

        if state == "in":
            game_status = f"Q{status.get('period', '')} {status.get('displayClock', '')}"
        elif state == "post":
            game_status = "Final"
        else:
            dt_utc = _parse_utc_iso(event.get("date", ""))
            game_status = _fmt_sched_time(dt_utc, tz_name="US/Eastern")

        games.append({
            "away": away,
            "home": home,
            "state": state,
            "status": game_status,
            "competition": comp,
        })

    return games

### ---------- INJURIES ----------
ESPN_TEAM_IDS = {
    "ARI":22,"ATL":1,"BAL":33,"BUF":2,"CAR":29,"CHI":3,"CIN":4,"CLE":5,"DAL":6,"DEN":7,"DET":8,"GB":9,
    "HOU":34,"IND":11,"JAX":30,"KC":12,"LV":13,"LAC":24,"LA":14,"MIA":15,"MIN":16,"NE":17,"NO":18,"NYG":19,"NYJ":20,"PHI":21,
    "PIT":23,"SF":25,"SEA":26,"TB":27,"TEN":10,"WAS":28
}

def fetch_nfl_injuries():
    """
    Fetch current NFL injury reports from ESPN's public API.
    Returns a pandas DataFrame with team, player, position, injury, and status.
    """
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
    resp = requests.get(url)

    if resp.status_code != 200:
        print("Error fetching injuries:", resp.status_code)
        return pd.DataFrame()

    data = resp.json()
    all_injuries = []

    for team in data.get("teams", []):
        team_name = team.get("team", {}).get("displayName", "")
        for injury in team.get("injuries", []):
            player = injury.get("athlete", {}).get("displayName", "")
            position = injury.get("athlete", {}).get("position", {}).get("abbreviation", "")
            injury_type = injury.get("injuryStatus", "")
            details = injury.get("details", "")
            status = injury.get("status", "")

            all_injuries.append({
                "team": team_name,
                "player": player,
                "pos": position,
                "injury": details if details else injury_type,
                "status": status
            })

    df = pd.DataFrame(all_injuries)
    return df


def injury_adjustment(players):
    penalty = 0
    for p in players:
        s = (p.get("status") or "").lower()
        pos = (p.get("position") or "").upper()
        if pos == "QB" and s in ["out", "doubtful"]:
            penalty -= 50
        elif pos in ["RB","WR","TE"] and s in ["out","doubtful"]:
            penalty -= 15
        elif s in ["out","doubtful"]:
            penalty -= 10
    return penalty

### ---------- WEATHER ----------
STADIUMS = {
   
    "Arizona Cardinals": {
        "stadium": "State Farm Stadium",
        "city": "Glendale",
        "state": "Arizona",
        "lat": 33.5276,
        "lon": -112.2626
    },
    "Atlanta Falcons": {
        "stadium": "Mercedes-Benz Stadium",
        "city": "Atlanta",
        "state": "Georgia",
        "lat": 33.7554,
        "lon": -84.4008
    },
    "Baltimore Ravens": {
        "stadium": "M&T Bank Stadium",
        "city": "Baltimore",
        "state": "Maryland",
        "lat": 39.2780,
        "lon": -76.6227
    },
    "Buffalo Bills": {
        "stadium": "Highmark Stadium",
        "city": "Orchard Park",
        "state": "New York",
        "lat": 42.7738,
        "lon": -78.7869
    },
    "Carolina Panthers": {
        "stadium": "Bank of America Stadium",
        "city": "Charlotte",
        "state": "North Carolina",
        "lat": 35.2251,
        "lon": -80.8531
    },
    "Chicago Bears": {
        "stadium": "Soldier Field",
        "city": "Chicago",
        "state": "Illinois",
        "lat": 41.8623,
        "lon": -87.6167
    },
    "Cincinnati Bengals": {
        "stadium": "Paycor Stadium",
        "city": "Cincinnati",
        "state": "Ohio",
        "lat": 39.0954,
        "lon": -84.5161
    },
    "Cleveland Browns": {
        "stadium": "Cleveland Browns Stadium",
        "city": "Cleveland",
        "state": "Ohio",
        "lat": 41.5061,
        "lon": -81.6995
    },
    "Dallas Cowboys": {
        "stadium": "AT&T Stadium",
        "city": "Arlington",
        "state": "Texas",
        "lat": 32.7473,
        "lon": -97.0945
    },
    "Denver Broncos": {
        "stadium": "Empower Field at Mile High",
        "city": "Denver",
        "state": "Colorado",
        "lat": 39.7439,
        "lon": -105.0201
    },
    "Detroit Lions": {
        "stadium": "Ford Field",
        "city": "Detroit",
        "state": "Michigan",
        "lat": 42.3400,
        "lon": -83.0456
    },
    "Green Bay Packers": {
        "stadium": "Lambeau Field",
        "city": "Green Bay",
        "state": "Wisconsin",
        "lat": 44.5013,
        "lon": -88.0622
    },
    "Houston Texans": {
        "stadium": "NRG Stadium",
        "city": "Houston",
        "state": "Texas",
        "lat": 29.6847,
        "lon": -95.4107
    },
    "Indianapolis Colts": {
        "stadium": "Lucas Oil Stadium",
        "city": "Indianapolis",
        "state": "Indiana",
        "lat": 39.7601,
        "lon": -86.1639
    },
    "Jacksonville Jaguars": {
        "stadium": "EverBank Stadium",
        "city": "Jacksonville",
        "state": "Florida",
        "lat": 30.3240,
        "lon": -81.6376
    },
    "Kansas City Chiefs": {
        "stadium": "GEHA Field at Arrowhead Stadium",
        "city": "Kansas City",
        "state": "Missouri",
        "lat": 39.0490,
        "lon": -94.4839
    },
    "Las Vegas Raiders": {
        "stadium": "Allegiant Stadium",
        "city": "Paradise",
        "state": "Nevada",
        "lat": 36.0909,
        "lon": -115.1830
    },
    "Los Angeles Chargers": {
        "stadium": "SoFi Stadium",
        "city": "Inglewood",
        "state": "California",
        "lat": 33.9535,
        "lon": -118.3387
    },
    "Los Angeles Rams": {
        "stadium": "SoFi Stadium",
        "city": "Inglewood",
        "state": "California",
        "lat": 33.9535,
        "lon": -118.3387
    },
    "Miami Dolphins": {
        "stadium": "Hard Rock Stadium",
        "city": "Miami Gardens",
        "state": "Florida",
        "lat": 25.9580,
        "lon": -80.2389
    },
    "Minnesota Vikings": {
        "stadium": "U.S. Bank Stadium",
        "city": "Minneapolis",
        "state": "Minnesota",
        "lat": 44.9737,
        "lon": -93.2570
    },
    "New England Patriots": {
        "stadium": "Gillette Stadium",
        "city": "Foxborough",
        "state": "Massachusetts",
        "lat": 42.0909,
        "lon": -71.2643
    },
    "New Orleans Saints": {
        "stadium": "Caesars Superdome",
        "city": "New Orleans",
        "state": "Louisiana",
        "lat": 29.9509,
        "lon": -90.0815
    },
    "New York Giants": {
        "stadium": "MetLife Stadium",
        "city": "East Rutherford",
        "state": "New Jersey",
        "lat": 40.8135,
        "lon": -74.0745
    },
    "New York Jets": {
        "stadium": "MetLife Stadium",
        "city": "East Rutherford",
        "state": "New Jersey",
        "lat": 40.8135,
        "lon": -74.0745
    },
    "Philadelphia Eagles": {
        "stadium": "Lincoln Financial Field",
        "city": "Philadelphia",
        "state": "Pennsylvania",
        "lat": 39.9008,
        "lon": -75.1675
    },
    "Pittsburgh Steelers": {
        "stadium": "Acrisure Stadium",
        "city": "Pittsburgh",
        "state": "Pennsylvania",
        "lat": 40.4468,
        "lon": -80.0158
    },
    "San Francisco 49ers": {
        "stadium": "Levi's Stadium",
        "city": "Santa Clara",
        "state": "California",
        "lat": 37.4030,
        "lon": -121.9700
    },
    "Seattle Seahawks": {
        "stadium": "Lumen Field",
        "city": "Seattle",
        "state": "Washington",
        "lat": 47.5952,
        "lon": -122.3316
    },
    "Tampa Bay Buccaneers": {
        "stadium": "Raymond James Stadium",
        "city": "Tampa",
        "state": "Florida",
        "lat": 27.9759,
        "lon": -82.5033
    },
    "Tennessee Titans": {
        "stadium": "Nissan Stadium",
        "city": "Nashville",
        "state": "Tennessee",
        "lat": 36.1665,
        "lon": -86.7713
    },
    "Washington Commanders": {
        "stadium": "Commanders Field",
        "city": "Landover",
        "state": "Maryland",
        "lat": 38.9077,
        "lon": -76.8645
    }
}

OWM_API_KEY = os.getenv("OWM_API_KEY", "")

def get_weather(team, kickoff_unix):
    if team not in STADIUMS or not OWM_API_KEY:
        return None
    lat, lon = STADIUMS[team]["lat"], STADIUMS[team]["lon"]
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OWM_API_KEY}&units=imperial"
    try:
        resp = requests.get(url, timeout=6); resp.raise_for_status(); data = resp.json()
    except Exception:
        return None
    forecasts = data.get("list", [])
    if not forecasts:
        return None
    closest = min(forecasts, key=lambda x: abs(int(x.get("dt",0)) - int(kickoff_unix)))
    dt_diff = abs(int(closest.get("dt",0)) - int(kickoff_unix))
    if dt_diff > 432000:
        return None
    try:
        return {
            "temp": closest["main"]["temp"],
            "wind_speed": closest["wind"]["speed"],
            "condition": closest["weather"][0]["main"]
        }
    except Exception:
        return None

def weather_adjustment(weather):
    if not weather:
        return 0
    pen = 0
    try:
        if weather.get("wind_speed", 0) > 20: pen -= 2
        if (weather.get("condition","").lower()) in ["rain","snow"]: pen -= 3
        if weather.get("temp", 100) < 25: pen -= 1
    except Exception:
        return 0
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

### ---------- NFL THEMED HEADERS ----------
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
                    align-items:center; justify-content:center; gap:16px;'>""" + logo_html + f"""
            <h1 style='color:white; margin:0; font-size:42px;'>{title}</h1>
            {logo_html}
        </div>
        """,
        unsafe_allow_html=True
    )

def nfl_subheader(text, icon="📊"):
    logo_html = f"<img src='data:image/png;base64,{NFL_LOGO_B64}' height='32' style='margin-right:8px;'/>" if NFL_LOGO_B64 else ""
    st.markdown(
        r"""
        <div style='background: linear-gradient(90deg, #d50a0a, #013369); 
                    padding: 12px; border-radius: 12px; text-align:center; display:flex; 
                    align-items:center; justify-content:center; gap:10px;'>""" + logo_html + f"""
            <h2 style='color:white; margin:0;'>{icon} {text}</h2>
            {logo_html}
        </div>
        """,
        unsafe_allow_html=True
    )

### ---------- KELLY BANKROLL MANAGEMENT ----------
def kelly_fraction(win_prob, odds_decimal):
    b = odds_decimal - 1 if odds_decimal else 0
    p = max(min(win_prob, 1), 0)
    q = 1 - p
    f = (b * p - q) / b if b > 0 else 0
    return max(f, 0)

### ---------- PREDICTION ACCURACY (detailed) ----------
@st.cache_data(ttl=3600)
def compute_detailed_accuracy(hist_df, elo_ratings):
    y_true, y_prob, correct, total = [], [], 0, 0
    per_team_stats = defaultdict(lambda: {"correct":0,"total":0})
    weekly_stats = defaultdict(lambda: {"correct":0,"total":0})
    home_stats = {"correct":0,"total":0}
    away_stats = {"correct":0,"total":0}

    for _, row in hist_df.iterrows():
        try:
            t1 = map_team_name(row.get("team1"))
            t2 = map_team_name(row.get("team2"))
            score1 = row.get("score1", 0) or 0
            score2 = row.get("score2", 0) or 0
            week = row.get("week", None)

            e1 = elo_ratings.get(t1, BASE_ELO)
            e2 = elo_ratings.get(t2, BASE_ELO)
            prob1 = expected_score(e1 + HOME_ADVANTAGE, e2)

            y_prob.append(prob1)
            y_true.append(1 if score1 > score2 else 0)

            predicted_winner = t1 if prob1 > 0.5 else t2
            actual_winner = t1 if score1 > score2 else t2
            if predicted_winner == actual_winner:
                correct += 1
            total += 1

            for team, won in [(t1, score1 > score2), (t2, score2 > score1)]:
                per_team_stats[team]["total"] += 1
                if (predicted_winner == team and won) or (predicted_winner != team and not won):
                    per_team_stats[team]["correct"] += 1

            if week is not None:
                weekly_stats[week]["total"] += 1
                if predicted_winner == actual_winner:
                    weekly_stats[week]["correct"] += 1

            home_team = t2
            away_team = t1
            home_stats["total"] += 1
            away_stats["total"] += 1
            if actual_winner == home_team:
                home_stats["correct"] += 1
            else:
                away_stats["correct"] += 1
        except Exception:
            # skip bad rows
            continue

    overall_accuracy = correct / total if total else 0
    brier = brier_score_loss(y_true, y_prob) if total else 1.0
    per_team_accuracy = {team: stats["correct"]/stats["total"] if stats["total"] else 0 for team, stats in per_team_stats.items()}
    weekly_accuracy = {week: stats["correct"]/stats["total"] if stats["total"] else 0 for week, stats in weekly_stats.items()}
    home_accuracy = home_stats["correct"]/home_stats["total"] if home_stats["total"] else 0
    away_accuracy = away_stats["correct"]/away_stats["total"] if away_stats["total"] else 0

    return {
        "overall_accuracy": overall_accuracy,
        "brier_score": brier,
        "per_team_accuracy": per_team_accuracy,
        "weekly_accuracy": weekly_accuracy,
        "home_accuracy": home_accuracy,
        "away_accuracy": away_accuracy
    }

### ---------- LOAD GAMES (cached) ----------
@st.cache_data(ttl=600)
def load_games(file=EXCEL_FILE):
    if os.path.exists(file):
        try:
            hist_df = pd.read_excel(file, sheet_name=HIST_SHEET)
        except Exception:
            hist_df = pd.DataFrame()
        try:
            sched_df = pd.read_excel(file, sheet_name=SCHEDULE_SHEET)
        except Exception:
            sched_df = pd.DataFrame()
        return hist_df, sched_df
    return pd.DataFrame(), pd.DataFrame()

# ---------- MAIN ----------
st.set_page_config(page_title="NFL Elo Projections", layout="wide")
nfl_header("NFL Elo Projections")

# Sidebar global settings
st.sidebar.header("Bankroll / Settings")
bankroll = st.sidebar.number_input("Bankroll ($)", min_value=1.0, value=float(DEFAULT_BANKROLL), step=1.0, format="%.2f")
st.sidebar.markdown("**Kelly stakes use the bankroll value above.**")

hist_df, sched_df = load_games()
ratings = run_elo_pipeline(hist_df) if not hist_df.empty else {}
acc_stats = compute_detailed_accuracy(hist_df, ratings) if not hist_df.empty else {
    "overall_accuracy":0,"brier_score":1.0,"per_team_accuracy":{}, "weekly_accuracy":{}, "home_accuracy":0, "away_accuracy":0
}

# Show some analytics in sidebar
with st.sidebar.expander("Prediction Accuracy"):
    st.metric("Overall Win %", f"{acc_stats['overall_accuracy']:.1%}")
    st.metric("Brier Score", f"{acc_stats['brier_score']:.4f}")
    st.metric("Home Win Accuracy", f"{acc_stats['home_accuracy']:.1%}")
    st.metric("Away Win Accuracy", f"{acc_stats['away_accuracy']:.1%}")

# Tabs
tabs = st.tabs(["Matchups", "Power Rankings", "Pick Winners", "Scoreboard", "Prediction Accuracy"])

# --- Matchups Tab ---
with tabs[0]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Matchups — Predictions & Kelly Stakes")
    if sched_df.empty:
        st.warning("Schedule not found in Excel.")
    else:
        week_series_num = pd.to_numeric(sched_df.get("week"), errors="coerce")
        available_weeks = sorted(set(week_series_num.dropna().astype(int).tolist()))
        if not available_weeks:
            st.warning("No valid weeks found in schedule.")
        else:
            selected_week = st.selectbox("Select Week", options=available_weeks, index=max(0,len(available_weeks)-1), key="week_matchups")
            mask = (week_series_num == selected_week)
            week_games = sched_df.loc[mask.fillna(False)]
            for _, row in week_games.iterrows():
                team_home = map_team_name(row.get("team2"))
                team_away = map_team_name(row.get("team1"))
                abbr_home, abbr_away = get_abbr(team_home), get_abbr(team_away)

                home_inj = fetch_injuries_espn(abbr_home) if abbr_home else []
                away_inj = fetch_injuries_espn(abbr_away) if abbr_away else []
                kickoff = default_kickoff_unix(row.get("date"))
                weather = get_weather(team_home, kickoff)

                adj_home = ratings.get(team_home, BASE_ELO) + injury_adjustment(home_inj) + weather_adjustment(weather)
                adj_away = ratings.get(team_away, BASE_ELO) + injury_adjustment(away_inj) + weather_adjustment(weather)

                win_prob_home = expected_score(adj_home + HOME_ADVANTAGE, adj_away)
                win_prob_away = 1 - win_prob_home

                col_odds1, col_odds2 = st.columns([1,1])
                with col_odds1:
                    odds_away = st.number_input(
                        f"{team_away} Odds (decimal)",
                        min_value=1.01, value=2.0, step=0.01,
                        key=f"odds_away_{team_away}_{team_home}"
                    )
                with col_odds2:
                    odds_home = st.number_input(
                        f"{team_home} Odds (decimal)",
                        min_value=1.01, value=2.0, step=0.01,
                        key=f"odds_home_{team_home}_{team_away}"
                    )

                kelly_home = kelly_fraction(win_prob_home, odds_home)
                kelly_away = kelly_fraction(win_prob_away, odds_away)
                stake_home = kelly_home * bankroll
                stake_away = kelly_away * bankroll

                # Projected score using season totals:
                NFL_AVG_TOTALS, overall_avg, alpha = {}, 44, 50
                if {"score1","score2","season"} <= set(hist_df.columns):
                    hist_df["total_points"] = (hist_df["score1"].fillna(0) + hist_df["score2"].fillna(0))
                    overall_avg = float(hist_df["total_points"].mean()) if len(hist_df) else 44.0
                    grouped = hist_df.groupby("season")["total_points"].agg(["mean","count"])
                    for s, r in grouped.iterrows():
                        NFL_AVG_TOTALS[int(s)] = (r["mean"]*r["count"] + overall_avg*alpha) / (r["count"] + alpha)
                season_val = row.get("season")
                try:
                    season_int = int(season_val) if pd.notna(season_val) else max(NFL_AVG_TOTALS.keys(), default=2025)
                except Exception:
                    season_int = max(NFL_AVG_TOTALS.keys(), default=2025)
                total_pts = NFL_AVG_TOTALS.get(season_int, overall_avg)
                proj_home = int(round(win_prob_home * total_pts))
                proj_away = int(round(win_prob_away * total_pts))

                st.markdown(
                    "<div style='background: rgba(255,255,255,0.12); backdrop-filter: blur(14px); "
                    "border-radius: 24px; padding: 25px; margin: 22px 0; box-shadow: 0 8px 25px rgba(0,0,0,0.25);'>",
                    unsafe_allow_html=True
                )

                col1, col_mid, col2 = st.columns([2, 3, 2])
                with col1:
                    safe_logo(abbr_away, 120)
                    st.markdown(f"<div style='text-align:center'>{neon_text(team_away, abbr_away, 28)}</div>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align:center; margin-top:6px;'>Win %: {win_prob_away:.1%}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align:center; margin-top:2px;'>Odds: {odds_away:.2f} — Kelly: {kelly_away:.2%} — Stake: ${stake_away:.2f}</p>", unsafe_allow_html=True)
                with col_mid:
                    st.markdown(f"<h1 style='text-align:center; margin:0;'>{proj_away} – {proj_home}</h1>", unsafe_allow_html=True)
                    st.markdown("<p style='text-align:center; margin:4px 0 0;'>Projected Score</p>", unsafe_allow_html=True)
                with col2:
                    safe_logo(abbr_home, 120)
                    st.markdown(f"<div style='text-align:center'>{neon_text(team_home, abbr_home, 28)}</div>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align:center; margin-top:6px;'>Win %: {win_prob_home:.1%}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align:center; margin-top:2px;'>Odds: {odds_home:.2f} — Kelly: {kelly_home:.2%} — Stake: ${stake_home:.2f}</p>", unsafe_allow_html=True)

                with st.expander("Weather Forecast 🌤️"):
                    if weather:
                        st.write(weather)
                    else:
                        st.caption("No forecast available.")

                with st.expander("Injuries 🩺"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**{team_away}**")
                        if away_inj:
                            for p in away_inj: st.write(p)
                        else:
                            st.caption("No reported injuries.")
                    with c2:
                        st.markdown(f"**{team_home}**")
                        if home_inj:
                            for p in home_inj: st.write(p)
                        else:
                            st.caption("No reported injuries.")
                st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Power Rankings Tab ---
with tabs[1]:
    nfl_subheader("Elo Power Rankings", "📊")
    adjusted_ratings = {}
    for team_full in NFL_FULL_NAMES.values():
        abbr = get_abbr(team_full)
        base = ratings.get(team_full, BASE_ELO)
        inj = fetch_injuries_espn(abbr) if abbr else []
        kickoff = default_kickoff_unix(datetime.datetime.now())
        weather = get_weather(team_full, kickoff)
        adj = base + injury_adjustment(inj) + weather_adjustment(weather)
        adjusted_ratings[team_full] = adj

    pr_df = (
        pd.DataFrame(sorted(adjusted_ratings.items(), key=lambda x: x[1], reverse=True),
                     columns=["Team", "Adj Elo"])
        .reset_index(drop=True)
    )
    pr_df.index = pr_df.index + 1
    pr_df.index.name = "Rank"

    for rank, row in pr_df.iterrows():
        team = row["Team"]
        abbr = get_abbr(team)
        elo_val = int(round(row["Adj Elo"]))
        c1, c2, c3 = st.columns([1, 2, 2])
        with c1:
            st.markdown(f"**#{rank}**")
        with c2:
            safe_logo(abbr, 50)
        with c3:
            st.markdown(f"{neon_text(team, abbr, 20)} – **{elo_val}**", unsafe_allow_html=True)
        st.markdown("---")

# --- Pick Winners Tab ---
with tabs[2]:
    nfl_subheader("Weekly Pick’em", "📝")
    week_series_num = pd.to_numeric(sched_df.get("week"), errors="coerce")
    available_weeks = sorted(set(week_series_num.dropna().astype(int).tolist()))
    if available_weeks:
        week = st.selectbox("Select Week", available_weeks, key="week_picks")
        games = sched_df.loc[(week_series_num == week).fillna(False)]
        picks = {}
        for _, row in games.iterrows():
            t_home, t_away = map_team_name(row.get("team2")), map_team_name(row.get("team1"))
            abbr_home, abbr_away = get_abbr(t_home), get_abbr(t_away)
            st.markdown("<div style='background:rgba(255,255,255,0.08); border-radius:18px; padding:16px; margin:12px 0;'>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns([3, 2, 3])
            with c1:
                safe_logo(abbr_away, 80)
                st.markdown(neon_text(t_away, abbr_away, 20), unsafe_allow_html=True)
            with c2:
                st.markdown("<h5 style='text-align:center'>Your Pick ➡️</h5>", unsafe_allow_html=True)
            with c3:
                safe_logo(abbr_home, 80)
                st.markdown(neon_text(t_home, abbr_home, 20), unsafe_allow_html=True)
            choice = st.radio("", [t_away, t_home], horizontal=True, key=f"pick_{t_home}_{t_away}")
            picks[f"{t_away} @ {t_home}"] = choice
            st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Save Picks to Excel (overwrites 'Picks' sheet)"):
            try:
                picks_df = pd.DataFrame([{"matchup": k, "pick": v} for k, v in picks.items()])
                with pd.ExcelWriter(EXCEL_FILE, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                    picks_df.to_excel(writer, sheet_name="Picks", index=False)
                st.success("Picks saved to Excel.")
            except Exception as e:
                st.error(f"Failed to save picks: {e}")
    else:
        st.info("Schedule not available for picks.")

# --- Scoreboard Tab ---
with tabs[3]:
    nfl_subheader("NFL Scoreboard", "🏟️")
    games = fetch_nfl_scores()
    if not games:
        st.info("No NFL games today or scheduled.")
    for game in games:
        away, home = game["away"], game["home"]
        state = game.get("state", "pre")
        comp = game.get("competition")
        situation = comp.get("situation", {}) if comp else {}

        status_obj = comp.get("status", {}) if comp else {}
        period = status_obj.get("period")
        clock = status_obj.get("displayClock", "")
        if state == "in":
            status_text = f"Q{period} {clock}"
        elif state == "post":
            status_text = "FINAL"
        else:
            status_text = game.get("status", "Scheduled")

        possession_id = situation.get("possession", {}).get("id")
        last_play = situation.get("lastPlay", {}).get("text", "")
        desc = situation.get("shortDownDistanceText")
        yard_line = situation.get("yardLine")
        drive_summary = f"{desc} on {yard_line}" if desc else None

        score_home = int(home.get("score", 0))
        score_away = int(away.get("score", 0))
        highlight_home = state == "post" and score_home > score_away
        highlight_away = state == "post" and score_away > score_home

        st.markdown(
            "<div style='background: #000000; backdrop-filter: blur(16px); border-radius:24px; "
            "padding:20px; margin:16px 0; box-shadow:0 10px 30px rgba(0,0,0,0.5); "
            "border:3px solid; border-image: linear-gradient(90deg, #d50a0a, #013369) 1;'>",
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns([3, 2, 3])
        with col1:
            try:
                logo_url = away['team'].get('logo')
                if logo_url:
                    st.image(logo_url, width=60)
            except Exception:
                pass
            team_abbr_away = get_abbr(away['team']['displayName'])
            st.markdown(f"<div style='text-align:center'>{neon_text(away['team']['displayName'], team_abbr_away, 22)}</div>", unsafe_allow_html=True)
            score_color_away = TEAM_COLORS.get(team_abbr_away, "#39ff14") if highlight_away else "#FFFFFF"
            st.markdown(
                f"<h2 style='text-align:center; color:{score_color_away}; text-shadow:0 0 10px {score_color_away};'>"
                f"{'🏈 ' if str(away['team'].get('id'))==str(possession_id) else ''}{score_away}</h2>",
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(f"<h3 style='text-align:center; color:#e5e7eb;'>{status_text}</h3>", unsafe_allow_html=True)

        with col3:
            try:
                logo_url = home['team'].get('logo')
                if logo_url:
                    st.image(logo_url, width=60)
            except Exception:
                pass
            team_abbr_home = get_abbr(home['team']['displayName'])
            st.markdown(f"<div style='text-align:center'>{neon_text(home['team']['displayName'], team_abbr_home, 22)}</div>", unsafe_allow_html=True)
            score_color_home = TEAM_COLORS.get(team_abbr_home, "#39ff14") if highlight_home else "#FFFFFF"
            st.markdown(
                f"<h2 style='text-align:center; color:{score_color_home}; text-shadow:0 0 10px {score_color_home};'>"
                f"{'🏈 ' if str(home['team'].get('id'))==str(possession_id) else ''}{score_home}</h2>",
                unsafe_allow_html=True
            )

        if drive_summary or last_play:
            st.markdown(
                "<div style='background: rgba(20,20,20,0.7); border-radius:12px; padding:8px; "
                "margin-top:10px; color:#e5e7eb; font-size:12px; text-align:center; text-shadow:0 0 4px #fff;'>",
                unsafe_allow_html=True
            )
            if drive_summary:
                st.markdown(f"📋 {drive_summary}", unsafe_allow_html=True)
            if last_play:
                st.markdown(f"📝 {last_play}", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# --- Prediction Accuracy Tab ---
with tabs[4]:
    st.header("Prediction Accuracy")
    st.markdown("Overall and breakdowns of Elo prediction performance on historical games.")

    st.subheader("Key Metrics")
    st.metric("Overall Win Accuracy", f"{acc_stats['overall_accuracy']:.1%}")
    st.metric("Brier Score", f"{acc_stats['brier_score']:.4f}")

    st.subheader("Home / Away Accuracy")
    st.write(f"Home accuracy: {acc_stats['home_accuracy']:.1%}")
    st.write(f"Away accuracy: {acc_stats['away_accuracy']:.1%}")

    st.subheader("Per-Team Accuracy")
    per_team = acc_stats.get("per_team_accuracy", {})
    if per_team:
        per_team_df = pd.DataFrame.from_dict(per_team, orient="index", columns=["Accuracy"]).sort_values("Accuracy", ascending=False)
        per_team_df.index.name = "Team"
        st.dataframe(per_team_df.style.format({"Accuracy":"{:.1%}"}))
    else:
        st.info("No per-team accuracy data available.")

    st.subheader("Weekly Accuracy Trend")
    weekly = acc_stats.get("weekly_accuracy", {})
    if weekly:
        weekly_df = pd.DataFrame.from_dict(weekly, orient="index", columns=["Accuracy"]).sort_index()
        weekly_df.index.name = "Week"
        st.line_chart(weekly_df)
    else:
        st.info("No weekly accuracy data available.")
