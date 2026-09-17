# app.py
# NFL Elo Projections App — Full rebuild with Kelly & Prediction Tracking (no Articles tab)
import streamlit as st
st.set_page_config(page_title="NFL Elo Projections", page_icon="🏈", layout="wide")

import pandas as pd
import numpy as np
from streamlit_autorefresh import st_autorefresh
from collections import defaultdict
from contextlib import contextmanager
import os, base64, requests, datetime, pytz, math, time, html
from pathlib import Path
from typing import Optional
from openpyxl import load_workbook, Workbook
from sklearn.metrics import brier_score_loss
try:
    import fcntl
except ImportError:
    fcntl = None

### ---------- CONFIG ----------
BASE_ELO = 1500
K = 20
HOME_ADVANTAGE = 65

APP_DIR = Path(__file__).resolve().parent
LOGOS_DIR = APP_DIR / "Logos"
SHIELD_IMAGE_PATH = APP_DIR / "Shield.png"
NFL_IMAGE_PATH = APP_DIR / "NFL.png"

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
    path = LOGOS_DIR / f"{abbr}.png"
    safe_abbr = html.escape(str(abbr or "?"))
    if abbr and path.exists():
        try:
            st.image(str(path), width=width)
        except Exception:
            st.markdown(
                f"<div style='width:{width}px; height:{width}px; background:#e5e7eb; "
                f"display:flex; align-items:center; justify-content:center; border-radius:50%; "
                f"font-size:12px; color:#475569;'>{safe_abbr}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f"<div style='width:{width}px; height:{width}px; background:#e5e7eb; "
            f"display:flex; align-items:center; justify-content:center; border-radius:50%; "
            f"font-size:12px; color:#475569;'>{safe_abbr}</div>",
            unsafe_allow_html=True,
        )

def neon_text(text, abbr=None, size=24):
    color = TEAM_COLORS.get(abbr, "#39ff14") if abbr else "#39ff14"
    safe_text = html.escape(str(text))
    return f"""
    <span style="
        color: #f8fafc;
        font-size: {size}px;
        font-weight: 800;
        letter-spacing: 0.02em;
        line-height: 1.15;
        -webkit-text-stroke: 1px {color};
        text-shadow:
            0 0 2px rgba(15, 23, 42, 0.95),
            0 0 8px {color},
            0 0 18px {color},
            0 0 30px {color};
    ">{safe_text}</span>
    """

# --- Set App Background ---
def set_background(image_path=SHIELD_IMAGE_PATH):
    image_path = Path(image_path)
    if image_path.exists():
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

set_background()

### ---------- ELO ----------
def expected_score(r1, r2):
    return 1 / (1 + 10 ** ((r2 - r1) / 400))

def regress_preseason(elo_ratings, reg=0.65, base=BASE_ELO):
    for t in list(elo_ratings.keys()):
        elo_ratings[t] = base + reg * (elo_ratings[t] - base)

def update_ratings(elo_ratings, team1, team2, score1, score2, home_team):
    """Update Elo ratings from a completed game result."""
    r1, r2 = elo_ratings[team1], elo_ratings[team2]
    if home_team == team1:
        r1 += HOME_ADVANTAGE
    elif home_team == team2:
        r2 += HOME_ADVANTAGE

    score1 = float(score1)
    score2 = float(score2)
    expected1 = expected_score(r1, r2)
    if score1 > score2:
        actual1 = 1.0
    elif score2 > score1:
        actual1 = 0.0
    else:
        actual1 = 0.5
    margin = max(abs(score1 - score2), 1.0)
    mov_mult = np.log(margin + 1) * (2.2 / ((r1 - r2) * 0.001 + 2.2))
    elo_ratings[team1] += K * mov_mult * (actual1 - expected1)
    elo_ratings[team2] += K * mov_mult * ((1 - actual1) - (1 - expected1))

def _normalize_status(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().replace("-", " ").replace("/", " ").split())

def _is_final_status(value) -> Optional[bool]:
    """Return True/False when status clearly indicates final/non-final, else None."""
    status = _normalize_status(value)
    if not status:
        return None
    tokens = set(status.split())
    if {"postponed", "cancelled", "canceled", "scheduled", "pregame", "halftime", "live"} & tokens:
        return False
    if status.startswith("q") or status in {"pre", "in"} or ("in" in tokens and "progress" in tokens):
        return False
    if "final" in tokens or {"complete", "completed"} & tokens or status == "post":
        return True
    return None

def _prepare_final_games(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce and return only completed historical games safe for Elo updates."""
    needed = {"season", "week", "team1", "team2", "score1", "score2"}
    if df is None or df.empty or not needed.issubset(set(df.columns)):
        return pd.DataFrame(columns=["season", "week", "team1", "team2", "score1", "score2", "home_team"])

    games = df.copy()
    for col in ["season", "week", "score1", "score2"]:
        games[col] = pd.to_numeric(games[col], errors="coerce")
    games = games.dropna(subset=["season", "week", "team1", "team2", "score1", "score2"])

    status_col = next((c for c in ["status", "game_status", "state"] if c in games.columns), None)
    if status_col:
        status_eval = games[status_col].apply(_is_final_status)
        games = games[status_eval != False]

    games["season"] = games["season"].astype(int)
    games["week"] = games["week"].astype(int)
    return games.sort_values(["season", "week"]).reset_index(drop=True)

def run_elo_pipeline(df):
    """Run Elo updates in chronological order on final games only."""
    games = _prepare_final_games(df)
    elo_ratings = defaultdict(lambda: BASE_ELO)
    prev_season = None
    for _, row in games.iterrows():
        season = int(row["season"])
        if prev_season is not None and season != prev_season:
            regress_preseason(elo_ratings)

        t1 = map_team_name(row.get("team1"))
        t2 = map_team_name(row.get("team2"))
        home_raw = row.get("home_team", None)
        home = map_team_name(home_raw) if pd.notna(home_raw) else None
        if home not in {t1, t2}:
            home = None
        update_ratings(elo_ratings, t1, t2, row["score1"], row["score2"], home)
        prev_season = season
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
INJURY_CACHE_TTL_SECONDS = 600

ESPN_TEAM_IDS = {
    "ARI":22,"ATL":1,"BAL":33,"BUF":2,"CAR":29,"CHI":3,"CIN":4,"CLE":5,"DAL":6,"DEN":7,"DET":8,"GB":9,
    "HOU":34,"IND":11,"JAX":30,"KC":12,"LV":13,"LAC":24,"LA":14,"MIA":15,"MIN":16,"NE":17,"NO":18,"NYG":19,"NYJ":20,"PHI":21,
    "PIT":23,"SF":25,"SEA":26,"TB":27,"TEN":10,"WAS":28
}

@st.cache_data(ttl=INJURY_CACHE_TTL_SECONDS)
def fetch_injuries_espn(team_abbr):
    team_id = ESPN_TEAM_IDS.get(team_abbr)
    if not team_id:
        return []
    url = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/{team_id}/injuries"
    try:
        r = requests.get(url, timeout=6); r.raise_for_status(); data = r.json()
    except Exception:
        return []
    players = []
    for e in data.get("entries", []):
        players.append({
            "name": e.get("athlete",{}).get("displayName"),
            "position": e.get("position",{}).get("abbreviation"),
            "status": e.get("status",{}).get("type","")
        })
    return players

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

@st.cache_data(ttl=600)
def _get_weather_cached(team: str, kickoff_unix: int, api_key: str):
    """Fetch cached weather near kickoff for the given home team."""
    if team not in STADIUMS or not api_key:
        return None
    try:
        kickoff_unix = int(kickoff_unix)
    except (TypeError, ValueError):
        return None
    lat, lon = STADIUMS[team]["lat"], STADIUMS[team]["lon"]
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=imperial"
    try:
        resp = requests.get(
            url,
            timeout=6,
            headers={"User-Agent": "DratingsClone/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    forecasts = data.get("list", [])
    if not isinstance(forecasts, list) or not forecasts:
        return None
    candidates = []
    for item in forecasts:
        try:
            dt_val = int(item.get("dt", 0))
        except (TypeError, ValueError, AttributeError):
            continue
        candidates.append((dt_val, item))
    if not candidates:
        return None
    dt_val, closest = min(candidates, key=lambda x: abs(x[0] - kickoff_unix))
    dt_diff = abs(dt_val - kickoff_unix)
    if dt_diff > 432000:
        return None
    try:
        return {
            "temp": float(closest["main"]["temp"]),
            "wind_speed": float(closest["wind"]["speed"]),
            "condition": str(closest["weather"][0]["main"]),
        }
    except (KeyError, TypeError, ValueError, IndexError):
        return None

def get_weather(team: str, kickoff_unix: int):
    """Public weather helper with cache isolation by API key."""
    return _get_weather_cached(team, kickoff_unix, OWM_API_KEY)

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
def load_local_logo(path=NFL_IMAGE_PATH):
    path = Path(path)
    if path.exists():
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

NFL_LOGO_B64 = load_local_logo()

def nfl_header(title):
    safe_title = html.escape(str(title))
    logo_html = f"<img src='data:image/png;base64,{NFL_LOGO_B64}' height='60'>" if NFL_LOGO_B64 else ""
    st.markdown(
        r"""
        <div style='background: linear-gradient(90deg, #013369, #d50a0a); 
                    padding: 20px; border-radius: 15px; text-align:center; display:flex; 
                    align-items:center; justify-content:center; gap:16px;'>""" + logo_html + f"""
            <h1 style='color:white; margin:0; font-size:42px;'>{safe_title}</h1>
            {logo_html}
        </div>
        """,
        unsafe_allow_html=True
    )

def nfl_subheader(text, icon="📊"):
    safe_text = html.escape(str(text))
    safe_icon = html.escape(str(icon))
    logo_html = f"<img src='data:image/png;base64,{NFL_LOGO_B64}' height='32' style='margin-right:8px;'/>" if NFL_LOGO_B64 else ""
    st.markdown(
        r"""
        <div style='background: linear-gradient(90deg, #d50a0a, #013369); 
                    padding: 12px; border-radius: 12px; text-align:center; display:flex; 
                    align-items:center; justify-content:center; gap:10px;'>""" + logo_html + f"""
            <h2 style='color:white; margin:0;'>{safe_icon} {safe_text}</h2>
            {logo_html}
        </div>
        """,
        unsafe_allow_html=True
    )

### ---------- KELLY BANKROLL MANAGEMENT ----------
def kelly_fraction(win_prob: float, odds_decimal: float, fraction: float = 0.25, max_fraction: float = 0.05) -> float:
    """Return a fractional Kelly stake fraction with safety cap."""
    try:
        p = max(min(float(win_prob), 1.0), 0.0)
        odds_decimal = float(odds_decimal)
        fraction = max(float(fraction), 0.0)
        max_fraction = max(float(max_fraction), 0.0)
    except (TypeError, ValueError):
        return 0.0
    b = odds_decimal - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - p
    full_kelly = ((b * p) - q) / b
    stake_fraction = max(full_kelly, 0.0) * fraction
    return min(stake_fraction, max_fraction)

def get_available_weeks(schedule_df: pd.DataFrame):
    """Return sorted available schedule weeks and aligned numeric week series."""
    if schedule_df is None or schedule_df.empty or "week" not in schedule_df.columns:
        if isinstance(schedule_df, pd.DataFrame):
            return [], pd.Series(index=schedule_df.index, dtype="float64")
        return [], pd.Series(dtype="float64")
    week_series = pd.to_numeric(schedule_df["week"], errors="coerce")
    weeks = sorted(set(week_series.dropna().astype(int).tolist()))
    return weeks, week_series

@st.cache_data(ttl=600)
def get_total_points_baselines(hist_df: pd.DataFrame, alpha: float = 50.0):
    """Compute per-season and global scoring baselines without mutating source data."""
    history = _prepare_final_games(hist_df)
    if history.empty:
        return {}, 44.0
    history["total_points"] = history["score1"] + history["score2"]
    overall_avg = float(history["total_points"].mean()) if len(history) else 44.0
    grouped = history.groupby("season")["total_points"].agg(["mean", "count"])
    season_avgs = {}
    for s, r in grouped.iterrows():
        season_avgs[int(s)] = (r["mean"] * r["count"] + overall_avg * alpha) / (r["count"] + alpha)
    return season_avgs, overall_avg

def normalize_matchup_key(away_team, home_team):
    return f"{map_team_name(away_team)} @ {map_team_name(home_team)}"

def normalize_matchup_value(matchup):
    if pd.isna(matchup):
        return None
    text = str(matchup).strip()
    parts = [p.strip() for p in text.split("@")]
    if len(parts) != 2:
        return text
    return normalize_matchup_key(parts[0], parts[1])

@st.cache_data(ttl=600)
def load_saved_picks(file=EXCEL_FILE):
    return _read_saved_picks_from_excel(file)

def _read_saved_picks_from_excel(file=EXCEL_FILE):
    columns = ["week", "matchup", "pick", "timestamp"]
    if not os.path.exists(file):
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_excel(file, sheet_name="Picks")
    except ValueError as e:
        if "Worksheet named 'Picks' not found" in str(e):
            return pd.DataFrame(columns=columns)
        raise
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df[columns]

def _write_picks_sheet(file, picks_df, columns):
    if os.path.exists(file):
        workbook = load_workbook(file)
        if "Picks" in workbook.sheetnames:
            sheet = workbook["Picks"]
        else:
            sheet = workbook.create_sheet("Picks")
    else:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Picks"

    if sheet.max_row and sheet.max_row > 0:
        sheet.delete_rows(1, sheet.max_row)
    for col_idx, col_name in enumerate(columns, start=1):
        sheet.cell(row=1, column=col_idx, value=col_name)
    for row_idx, row in enumerate(picks_df.itertuples(index=False, name=None), start=2):
        for col_idx, value in enumerate(row, start=1):
            sheet.cell(row=row_idx, column=col_idx, value=value)
    workbook.save(file)

@contextmanager
def picks_file_lock(file):
    lock_path = f"{file}.lock"
    if fcntl:
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        return

    lock_dir = f"{lock_path}.d"
    acquired = False
    try:
        for _ in range(100):
            try:
                os.mkdir(lock_dir)
                acquired = True
                break
            except FileExistsError:
                time.sleep(0.05)
        if not acquired:
            raise TimeoutError("Could not acquire picks save lock.")
        yield
    finally:
        if acquired and os.path.isdir(lock_dir):
            os.rmdir(lock_dir)

def save_week_picks(week, picks_dict, file=EXCEL_FILE):
    columns = ["week", "matchup", "pick", "timestamp"]
    try:
        week_int = int(week)
    except Exception:
        return False

    rows = []
    now_ts = datetime.datetime.now().isoformat(timespec="seconds")
    for matchup, pick in (picks_dict or {}).items():
        norm_matchup = normalize_matchup_value(matchup)
        norm_pick = map_team_name(pick)
        if not norm_matchup or not norm_pick:
            continue
        rows.append({
            "week": week_int,
            "matchup": norm_matchup,
            "pick": norm_pick,
            "timestamp": now_ts
        })

    if not rows:
        return False

    new_rows = pd.DataFrame(rows, columns=columns)

    def _save_once():
        existing = _read_saved_picks_from_excel(file).copy()
        if not existing.empty:
            existing["week"] = pd.to_numeric(existing["week"], errors="coerce")
            existing["matchup"] = existing["matchup"].apply(normalize_matchup_value)
            existing["pick"] = existing["pick"].apply(map_team_name)
            existing = existing.dropna(subset=["week", "matchup", "pick"])
            existing["week"] = existing["week"].astype(int)
            existing = existing[~(
                (existing["week"] == week_int) &
                (existing["matchup"].isin(new_rows["matchup"]))
            )]

        out = pd.concat([existing[columns], new_rows], ignore_index=True)
        out = out.drop_duplicates(subset=["week", "matchup"], keep="last")
        _write_picks_sheet(file, out, columns)

    with picks_file_lock(file):
        _save_once()

    load_saved_picks.clear()
    return True

def build_actual_results_by_week(hist_df):
    needed = {"week", "team1", "team2", "score1", "score2"}
    cols = ["week", "matchup", "winner", "is_final"]
    if hist_df is None or hist_df.empty or not needed.issubset(set(hist_df.columns)):
        return pd.DataFrame(columns=cols)

    rows = []
    for _, row in hist_df.iterrows():
        week = pd.to_numeric(row.get("week"), errors="coerce")
        if pd.isna(week):
            continue
        away_team = map_team_name(row.get("team1"))
        home_team = map_team_name(row.get("team2"))
        score1 = pd.to_numeric(row.get("score1"), errors="coerce")
        score2 = pd.to_numeric(row.get("score2"), errors="coerce")
        score_complete = pd.notna(score1) and pd.notna(score2)
        status_raw = row.get("status", row.get("game_status", row.get("state", None)))
        status_text = str(status_raw).strip().lower() if pd.notna(status_raw) else ""
        normalized_status = " ".join(status_text.replace("-", " ").replace("/", " ").split())
        status_tokens = set(normalized_status.split()) if normalized_status else set()
        if "postponed" in status_tokens:
            is_final = False
        elif "final" in status_tokens or {"complete", "completed"} & status_tokens or normalized_status == "post":
            is_final = True
        elif (
            normalized_status.startswith("q")
            or "live" in status_tokens
            or "halftime" in status_tokens
            or "scheduled" in status_tokens
            or "pregame" in status_tokens
            or "pre" in status_tokens
            or ("in" in status_tokens and "progress" in status_tokens)
        ):
            is_final = False
        else:
            is_final = score_complete
        winner = None
        if is_final:
            if score1 > score2:
                winner = away_team
            elif score2 > score1:
                winner = home_team
            else:
                winner = "TIE"
        rows.append({
            "week": int(week),
            "matchup": normalize_matchup_key(away_team, home_team),
            "winner": winner,
            "is_final": bool(is_final)
        })

    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).drop_duplicates(subset=["week", "matchup"], keep="last")

def grade_picks(saved_picks_df, results_df):
    graded_cols = ["week", "matchup", "pick", "timestamp", "winner", "is_final", "status", "result"]
    if saved_picks_df is None or saved_picks_df.empty:
        return pd.DataFrame(columns=graded_cols)

    picks = saved_picks_df.copy()
    for col in ["week", "matchup", "pick", "timestamp"]:
        if col not in picks.columns:
            picks[col] = np.nan

    picks["week"] = pd.to_numeric(picks["week"], errors="coerce")
    picks = picks.dropna(subset=["week", "matchup", "pick"])
    if picks.empty:
        return pd.DataFrame(columns=graded_cols)

    picks["week"] = picks["week"].astype(int)
    picks["matchup"] = picks["matchup"].apply(normalize_matchup_value)
    picks["pick"] = picks["pick"].apply(map_team_name)

    if results_df is None or results_df.empty:
        merged = picks.copy()
        merged["winner"] = None
        merged["is_final"] = False
    else:
        results = results_df.copy()
        results["week"] = pd.to_numeric(results["week"], errors="coerce")
        results = results.dropna(subset=["week", "matchup"])
        results["week"] = results["week"].astype(int)
        results["matchup"] = results["matchup"].apply(normalize_matchup_value)
        merged = picks.merge(results[["week", "matchup", "winner", "is_final"]], on=["week", "matchup"], how="left")
        merged["is_final"] = merged["is_final"].fillna(False)

    merged["status"] = np.where(
        merged["is_final"] != True,
        "pending",
        np.where(
            merged["winner"] == "TIE",
            "tie/push",
            np.where(merged["pick"] == merged["winner"], "correct", "wrong")
        ),
    )
    merged["result"] = np.where(
        merged["status"] == "correct",
        "✅ Correct",
        np.where(
            merged["status"] == "wrong",
            "❌ Wrong",
            np.where(merged["status"] == "tie/push", "➖ Tie/Push", "⏳ Pending"),
        ),
    )
    return merged[graded_cols]

### ---------- PREDICTION ACCURACY (detailed) ----------
@st.cache_data(ttl=3600)
def compute_detailed_accuracy(hist_df: pd.DataFrame, elo_ratings=None):
    """Evaluate prediction accuracy sequentially with pregame Elo probabilities."""
    games = _prepare_final_games(hist_df)
    if games.empty:
        return {
            "overall_accuracy": 0,
            "brier_score": 1.0,
            "per_team_accuracy": {},
            "weekly_accuracy": {},
            "home_accuracy": 0,
            "away_accuracy": 0
        }

    ratings = defaultdict(lambda: BASE_ELO)
    y_true, y_prob, correct, total = [], [], 0, 0
    per_team_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    weekly_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    home_stats = {"correct": 0, "total": 0}
    away_stats = {"correct": 0, "total": 0}
    prev_season = None

    for _, row in games.iterrows():
        t1 = map_team_name(row.get("team1"))
        t2 = map_team_name(row.get("team2"))
        home_raw = row.get("home_team", None)
        home_team = map_team_name(home_raw) if pd.notna(home_raw) else None
        if home_team == t1:
            away_team = t2
        elif home_team == t2:
            away_team = t1
        else:
            home_team = None
            away_team = None
        score1 = float(row["score1"])
        score2 = float(row["score2"])
        week = int(row["week"])
        season = int(row["season"])

        if prev_season is not None and season != prev_season:
            regress_preseason(ratings)

        e1 = ratings[t1] + (HOME_ADVANTAGE if home_team == t1 else 0)
        e2 = ratings[t2] + (HOME_ADVANTAGE if home_team == t2 else 0)
        prob1 = expected_score(e1, e2)

        update_ratings(ratings, t1, t2, score1, score2, home_team)
        prev_season = season

        if score1 == score2:
            continue

        actual_team1_won = score1 > score2
        predicted_team1_won = prob1 >= 0.5
        predicted_winner = t1 if predicted_team1_won else t2
        actual_winner = t1 if actual_team1_won else t2

        y_prob.append(prob1)
        y_true.append(1 if actual_team1_won else 0)
        is_correct = predicted_winner == actual_winner
        correct += int(is_correct)
        total += 1

        for team in (t1, t2):
            per_team_stats[team]["total"] += 1
            if (predicted_winner == team) == (actual_winner == team):
                per_team_stats[team]["correct"] += 1

        weekly_stats[week]["total"] += 1
        weekly_stats[week]["correct"] += int(is_correct)

        if predicted_winner == home_team:
            home_stats["total"] += 1
            home_stats["correct"] += int(actual_winner == home_team)
        elif predicted_winner == away_team:
            away_stats["total"] += 1
            away_stats["correct"] += int(actual_winner == away_team)

    overall_accuracy = correct / total if total else 0
    brier = brier_score_loss(y_true, y_prob) if y_true else 1.0
    per_team_accuracy = {team: stats["correct"] / stats["total"] if stats["total"] else 0 for team, stats in per_team_stats.items()}
    weekly_accuracy = {week: stats["correct"] / stats["total"] if stats["total"] else 0 for week, stats in weekly_stats.items()}
    home_accuracy = home_stats["correct"] / home_stats["total"] if home_stats["total"] else 0
    away_accuracy = away_stats["correct"] / away_stats["total"] if away_stats["total"] else 0

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
nfl_header("NFL Elo Projections")

st_autorefresh(interval=INJURY_CACHE_TTL_SECONDS * 1000, key="injury_data_autorefresh")

# Sidebar global settings
st.sidebar.header("Bankroll / Settings")
bankroll = st.sidebar.number_input("Bankroll ($)", min_value=1.0, value=float(DEFAULT_BANKROLL), step=1.0, format="%.2f")
st.sidebar.markdown("**Kelly stakes use the bankroll value above.**")
if st.sidebar.button("Refresh Injury Data", use_container_width=True):
    fetch_injuries_espn.clear()
    st.rerun()
st.sidebar.caption("Injury data refreshes automatically about every 10 minutes.")

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
        available_weeks, week_series_num = get_available_weeks(sched_df)
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
                NFL_AVG_TOTALS, overall_avg = get_total_points_baselines(hist_df)
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
    available_weeks, week_series_num = get_available_weeks(sched_df)
    if available_weeks:
        week = st.selectbox("Select Week", available_weeks, key="week_picks")
        games = sched_df.loc[(week_series_num == week).fillna(False)]
        try:
            saved_picks_df = load_saved_picks()
        except Exception:
            saved_picks_df = pd.DataFrame(columns=["week", "matchup", "pick", "timestamp"])
            st.warning("Could not read saved picks. You can still make picks and save again.")
        existing_week_picks = {}
        if not saved_picks_df.empty:
            week_saved = saved_picks_df[pd.to_numeric(saved_picks_df["week"], errors="coerce") == int(week)].copy()
            if not week_saved.empty:
                week_saved["timestamp_dt"] = pd.to_datetime(week_saved["timestamp"], errors="coerce")
                week_saved["matchup"] = week_saved["matchup"].apply(normalize_matchup_value)
                week_saved["pick"] = week_saved["pick"].apply(map_team_name)
                week_saved = week_saved.sort_values("timestamp_dt").drop_duplicates(subset=["matchup"], keep="last")
                existing_week_picks = dict(zip(week_saved["matchup"], week_saved["pick"]))
        picks = {}
        for _, row in games.iterrows():
            t_home, t_away = map_team_name(row.get("team2")), map_team_name(row.get("team1"))
            abbr_home, abbr_away = get_abbr(t_home), get_abbr(t_away)
            matchup_key = normalize_matchup_key(t_away, t_home)
            default_pick = existing_week_picks.get(matchup_key)
            options = [t_away, t_home]
            default_index = options.index(default_pick) if default_pick in options else 0
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
            choice = st.radio("", options, index=default_index, horizontal=True, key=f"pick_{week}_{t_home}_{t_away}")
            picks[matchup_key] = map_team_name(choice)
            st.markdown("</div>", unsafe_allow_html=True)

        if st.button(f"Save Picks for Week {week}"):
            try:
                if save_week_picks(week, picks):
                    st.success(f"Picks saved for Week {week}. Re-save anytime to update your picks.")
                else:
                    st.warning("No valid picks to save for this week.")
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
        safe_status_text = html.escape(str(status_text))

        possession_id = situation.get("possession", {}).get("id")
        last_play = situation.get("lastPlay", {}).get("text", "")
        desc = situation.get("shortDownDistanceText")
        yard_line = situation.get("yardLine")
        drive_summary = f"{desc} on {yard_line}" if desc else None
        safe_drive_summary = html.escape(str(drive_summary)) if drive_summary else None
        safe_last_play = html.escape(str(last_play)) if last_play else None

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
            st.markdown(f"<h3 style='text-align:center; color:#e5e7eb;'>{safe_status_text}</h3>", unsafe_allow_html=True)

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
            if safe_drive_summary:
                st.markdown(f"📋 {safe_drive_summary}", unsafe_allow_html=True)
            if safe_last_play:
                st.markdown(f"📝 {safe_last_play}", unsafe_allow_html=True)
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

    st.subheader("Your Pick Results")
    try:
        saved_picks = load_saved_picks()
    except Exception:
        saved_picks = pd.DataFrame(columns=["week", "matchup", "pick", "timestamp"])
        st.warning("Could not read saved picks for review.")
    actual_results = build_actual_results_by_week(hist_df)
    graded_picks = grade_picks(saved_picks, actual_results)

    if graded_picks.empty:
        st.info("No saved picks yet.")
    else:
        saved_weeks = sorted(graded_picks["week"].dropna().astype(int).unique().tolist())
        review_week = st.selectbox(
            "Select Week to Review",
            options=saved_weeks,
            index=max(0, len(saved_weeks) - 1),
            key="review_pick_week"
        )
        week_results = graded_picks[graded_picks["week"] == review_week].copy()

        week_wins = int((week_results["status"] == "correct").sum())
        week_losses = int((week_results["status"] == "wrong").sum())
        week_pending = int((week_results["status"] == "pending").sum())
        week_pushes = int((week_results["status"] == "tie/push").sum())

        st.markdown(
            f"**Week {review_week} Record:** {week_wins}-{week_losses}"
            + (f" (Pending: {week_pending})" if week_pending else "")
            + (f" (Pushes: {week_pushes})" if week_pushes else "")
        )

        season_final = graded_picks[graded_picks["status"].isin(["correct", "wrong"])]
        season_wins = int((season_final["status"] == "correct").sum())
        season_losses = int((season_final["status"] == "wrong").sum())
        st.markdown(f"**Season Record (Finalized):** {season_wins}-{season_losses}")

        display_df = week_results[["matchup", "pick", "winner", "result"]].copy()
        display_df["winner"] = display_df["winner"].fillna("Pending")
        st.dataframe(display_df, use_container_width=True)
