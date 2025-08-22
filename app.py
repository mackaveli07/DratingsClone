import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
import os
import requests
import datetime
import pytz
import io

### ---------- CONFIG ----------
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

### ---------- HELPERS ----------
def map_team_name(name):
    if not name:
        return "Unknown"
    name = str(name).strip()
    key = name.upper()
    if key in NFL_FULL_NAMES:
        return NFL_FULL_NAMES[key]
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
            f"<div style='width:{width}px; height:{width}px; background:#e5e7eb; display:flex; align-items:center; justify-content:center; border-radius:50%; font-size:12px; color:#475569;'>{abbr or '?'}" + "</div>",
            unsafe_allow_html=True,
        )

### ---------- ELO FUNCTIONS ----------
def expected_score(r1, r2):
    return 1 / (1 + 10 ** ((r2 - r1) / 400))

def regress_preseason(elo_ratings, reg=0.65, base=BASE_ELO):
    for t in elo_ratings:
        elo_ratings[t] = base + reg * (elo_ratings[t] - base)

def update_ratings(elo_ratings, team1, team2, score1, score2, home_team):
    r1, r2 = elo_ratings[team1], elo_ratings[team2]

    if home_team == team1:
        r1 += HOME_ADVANTAGE
    elif home_team == team2:
        r2 += HOME_ADVANTAGE

    expected1 = expected_score(r1, r2)
    actual1 = 1 if score1 > score2 else 0

    margin = abs(score1 - score2)
    if margin == 0:
        margin = 1
    mov_mult = np.log(margin + 1) * (2.2 / ((r1 - r2) * 0.001 + 2.2))

    elo_ratings[team1] += K * mov_mult * (actual1 - expected1)
    elo_ratings[team2] += K * mov_mult * ((1 - actual1) - expected_score(r2, r1))

def run_elo_pipeline(df):
    elo_ratings = defaultdict(lambda: BASE_ELO)
    if {"season", "week"} <= set(df.columns):
        df = df.sort_values(["season", "week"])  # historical order
        seasons = df["season"].dropna().unique().tolist()
        for i, s in enumerate(seasons):
            if i > 0:
                regress_preseason(elo_ratings)
            games = df[df["season"] == s]
            for _, row in games.iterrows():
                team1, team2 = map_team_name(row.get("team1")), map_team_name(row.get("team2"))
                score1, score2 = row.get("score1", 0), row.get("score2", 0)
                home_team = map_team_name(row.get("home_team", team2))
                update_ratings(elo_ratings, team1, team2, score1, score2, home_team)
    return dict(elo_ratings)

### ---------- ESPN INJURIES ----------
ESPN_TEAM_IDS = {
    "ARI": 22, "ATL": 1, "BAL": 33, "BUF": 2,
    "CAR": 29, "CHI": 3, "CIN": 4, "CLE": 5,
    "DAL": 6, "DEN": 7, "DET": 8, "GB": 9,
    "HOU": 34, "IND": 11, "JAX": 30, "KC": 12,
    "LV": 13, "LAC": 24, "LA": 14, "MIA": 15,
    "MIN": 16, "NE": 17, "NO": 18, "NYG": 19,
    "NYJ": 20, "PHI": 21, "PIT": 23, "SF": 25,
    "SEA": 26, "TB": 27, "TEN": 10, "WAS": 28
}

@st.cache_data(show_spinner=False, ttl=900)
def fetch_injuries_espn(team_abbr):
    """Fetch team injury report from ESPN. Cached for 15 minutes."""
    team_id = ESPN_TEAM_IDS.get(team_abbr)
    if not team_id:
        return []
    url = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/{team_id}/injuries"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    players = []
    for entry in data.get("entries", []):
        athlete = entry.get("athlete", {})
        position = (entry.get("position") or {}).get("abbreviation")
        status = (entry.get("status") or {}).get("type", "")
        players.append({
            "name": athlete.get("displayName"),
            "position": position,
            "status": status
        })
    return players

def injury_adjustment(players):
    """Return Elo penalty (negative number) from a list of injury dicts."""
    penalty = 0
    for p in players:
        status = (p.get("status") or "").lower()
        pos = (p.get("position") or "").upper()
        if pos == "QB" and status in ["out", "doubtful"]:
            penalty -= 50
        elif pos in ["RB", "WR", "TE"] and status in ["out", "doubtful"]:
            penalty -= 15
        elif status in ["out", "doubtful"]:
            penalty -= 10
    return penalty

### ---------- WEATHER ----------
STADIUMS = {
    "Arizona Cardinals": {"lat": 33.5277, "lon": -112.2626},
    "Atlanta Falcons": {"lat": 33.7554, "lon": -84.4007},
    "Baltimore Ravens": {"lat": 39.2780, "lon": -76.6227},
    "Buffalo Bills": {"lat": 42.7738, "lon": -78.7868},
    "Carolina Panthers": {"lat": 35.2251, "lon": -80.8526},
    "Chicago Bears": {"lat": 41.8625, "lon": -87.6166},
    "Cincinnati Bengals": {"lat": 39.0954, "lon": -84.5160},
    "Cleveland Browns": {"lat": 41.5061, "lon": -81.6995},
    "Dallas Cowboys": {"lat": 32.7473, "lon": -97.0945},
    "Denver Broncos": {"lat": 39.7439, "lon": -105.0201},
    "Detroit Lions": {"lat": 42.3390, "lon": -83.0456},
    "Green Bay Packers": {"lat": 44.5013, "lon": -88.0622},
    "Houston Texans": {"lat": 29.6847, "lon": -95.4107},
    "Indianapolis Colts": {"lat": 39.7601, "lon": -86.1639},
    "Jacksonville Jaguars": {"lat": 30.3239, "lon": -81.6374},
    "Kansas City Chiefs": {"lat": 39.0490, "lon": -94.4840},
    "Las Vegas Raiders": {"lat": 36.0908, "lon": -115.1830},
    "Los Angeles Chargers": {"lat": 33.9535, "lon": -118.3392},
    "Los Angeles Rams": {"lat": 33.9535, "lon": -118.3392},
    "Miami Dolphins": {"lat": 25.9580, "lon": -80.2389},
    "Minnesota Vikings": {"lat": 44.9738, "lon": -93.2581},
    "New England Patriots": {"lat": 42.0909, "lon": -71.2643},
    "New Orleans Saints": {"lat": 29.9511, "lon": -90.0812},
    "New York Giants": {"lat": 40.8128, "lon": -74.0745},
    "New York Jets": {"lat": 40.8128, "lon": -74.0745},
    "Philadelphia Eagles": {"lat": 39.9008, "lon": -75.1675},
    "Pittsburgh Steelers": {"lat": 40.4467, "lon": -80.0158},
    "San Francisco 49ers": {"lat": 37.4030, "lon": -121.9700},
    "Seattle Seahawks": {"lat": 47.5952, "lon": -122.3316},
    "Tampa Bay Buccaneers": {"lat": 27.9759, "lon": -82.5033},
    "Tennessee Titans": {"lat": 36.1665, "lon": -86.7713},
    "Washington Commanders": {"lat": 38.9077, "lon": -76.8645},
}

# Prefer secrets or env vars for keys
OWM_API_KEY = st.secrets.get("OWM_API_KEY") or os.getenv("OWM_API_KEY") or ""

@st.cache_data(show_spinner=False, ttl=1800)
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

    # Pick the forecast closest to kickoff
    closest = min(forecasts, key=lambda x: abs(int(x.get("dt", 0)) - int(kickoff_unix)))
    dt_diff = abs(int(closest.get("dt", 0)) - int(kickoff_unix))

    # Only accept if within 5 days (432,000 seconds)
    if dt_diff > 432000:
        return None

    return {
        "temp": closest.get("main", {}).get("temp"),
        "wind_speed": closest.get("wind", {}).get("speed"),
        "condition": (closest.get("weather") or [{}])[0].get("main", "")
    }


def weather_adjustment(weather):
    """Return point adjustment to total score based on weather."""
    if not weather:
        return 0
    penalty = 0
    try:
        if (weather.get("wind_speed") or 0) > 20:
            penalty -= 2
        if (weather.get("condition", "").lower() in ["rain", "snow"]):
            penalty -= 3
        if (weather.get("temp") or 100) < 25:
            penalty -= 1
    except Exception:
        return 0
    return penalty

def default_kickoff_unix(game_date):
    """Default kickoff to 1:00 PM EST for the given date (YYYY-MM-DD)."""
    if isinstance(game_date, str):
        try:
            game_date = datetime.datetime.strptime(game_date, "%Y-%m-%d")
        except Exception:
            return 0
    elif not isinstance(game_date, datetime.datetime):
        return 0
    est = pytz.timezone("US/Eastern")
    kickoff = est.localize(datetime.datetime(
        game_date.year, game_date.month, game_date.day, 13, 0, 0
    ))
    return int(kickoff.timestamp())

### ---------- LOAD DATA ----------
st.set_page_config(page_title="NFL Elo Projections", layout="wide")
st.title("🏈 NFL Elo Projections (Elo + Injuries + Weather)")

try:
    hist_df = pd.read_excel(EXCEL_FILE, sheet_name=HIST_SHEET)
    sched_df = pd.read_excel(EXCEL_FILE, sheet_name=SCHEDULE_SHEET)
except Exception as e:
    st.error(f"Error loading Excel file or sheets: {e}")
    st.stop()

NFL_AVG_TOTALS = {}
overall_avg = 44
alpha = 50
if {"score1", "score2", "season"} <= set(hist_df.columns):
    hist_df["total_points"] = hist_df["score1"] + hist_df["score2"]
    overall_avg = float(hist_df["total_points"].mean())
    grouped = hist_df.groupby("season")["total_points"].agg(["mean", "count"])
    for season, row in grouped.iterrows():
        season_avg, n = row["mean"], row["count"]
        blended = (season_avg * n + overall_avg * alpha) / (n + alpha)
        NFL_AVG_TOTALS[season] = float(blended)

ratings = run_elo_pipeline(hist_df)

# Column names in schedule
HOME_COL = "team2"  # home team
AWAY_COL = "team1"  # away team

# ---------- Tabs ----------
tabs = st.tabs(["Matchups", "Power Rankings", "Pick Winners"]) 

### ---- Matchups Tab ----
with tabs[0]:
    available_weeks = sorted(sched_df['week'].dropna().unique().astype(int).tolist())
    selected_week = st.selectbox("Select Week", available_weeks, index=max(0, len(available_weeks)-1))
    week_games = sched_df[sched_df['week'] == selected_week]

    for _, row in week_games.iterrows():
        team_home, team_away = map_team_name(row[HOME_COL]), map_team_name(row[AWAY_COL])
        elo_home, elo_away = ratings.get(team_home, BASE_ELO), ratings.get(team_away, BASE_ELO)

        # --- Injury Adjustments ---
        home_abbr = get_abbr(team_home)
        away_abbr = get_abbr(team_away)
        inj_home = fetch_injuries_espn(home_abbr)
        inj_away = fetch_injuries_espn(away_abbr)

        elo_home_adj = elo_home + injury_adjustment(inj_home)
        elo_away_adj = elo_away + injury_adjustment(inj_away)

        # --- Win Probabilities (injury-adjusted Elo) ---
        prob_home = expected_score(elo_home_adj + HOME_ADVANTAGE, elo_away_adj)
        prob_away = 1 - prob_home

        # --- Weather Adjustment for totals ---
        game_date = row.get("date")
        kickoff_unix = default_kickoff_unix(game_date)
        weather = get_weather(team_home, kickoff_unix) if kickoff_unix else None

        avg_total = NFL_AVG_TOTALS.get(row.get("season"), overall_avg)
        if weather:
            avg_total += weather_adjustment(weather)

        elo_diff = (elo_home_adj + HOME_ADVANTAGE) - elo_away_adj
        spread_home = elo_diff / 25
        predicted_home_score = round((avg_total / 2) + (spread_home / 2), 1)
        predicted_away_score = round((avg_total / 2) - (spread_home / 2), 1)

        # ----- Card -----
        with st.container():
            st.markdown(
                """
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(12px); border-radius: 20px; padding: 20px; margin: 16px 0; box-shadow: 0 4px 30px rgba(0,0,0,0.1);">
                """,
                unsafe_allow_html=True,
            )
            col1, col_mid, col2 = st.columns([2, 3, 2])

            with col1:
                if away_abbr:
                    safe_logo(away_abbr, width=90)
                st.markdown(f"<h4 style='text-align:center'>{team_away}</h4>", unsafe_allow_html=True)

            with col_mid:
                st.markdown(
                    f"<h2 style='text-align:center; margin:0;'>{predicted_away_score} – {predicted_home_score}</h2>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<p style='text-align:center; font-size:14px; color:#6b7280;'>Projected Score</p>",
                    unsafe_allow_html=True,
                )

            with col2:
                if home_abbr:
                    safe_logo(home_abbr, width=90)
                st.markdown(f"<h4 style='text-align:center'>{team_home}</h4>", unsafe_allow_html=True)

            st.markdown(
                f"""
                <div style=\"display:flex; justify-content:space-between; margin-top:15px;\">
                    <div style=\"flex:1; text-align:center; color:#ef4444;\">
                        <b>{prob_away*100:.1f}%</b><br><span style=\"font-size:13px;\">Win Prob</span>
                    </div>
                    <div style=\"flex:1; text-align:center; color:#2563eb;\">
                        <b>{prob_home*100:.1f}%</b><br><span style=\"font-size:13px;\">Win Prob</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # --- Injury Report ---
            with st.expander("Injury Report"):
                if inj_home or inj_away:
                    st.markdown(f"**{team_home}**:")
                    for p in inj_home[:6]:
                        st.write(f"- {p['name']} ({p['position']}): {p['status']}")
                    st.markdown(f"**{team_away}**:")
                    for p in inj_away[:6]:
                        st.write(f"- {p['name']} ({p['position']}): {p['status']}")
                else:
                    st.write("No major injuries reported.")

            # --- Weather Report ---
            with st.expander("Weather Forecast"):
                if weather:
                    st.write(f"🌡️ Temp: {weather['temp']}°F")
                    st.write(f"💨 Wind: {weather['wind_speed']} mph")
                    st.write(f"🌦️ Condition: {weather['condition']}")
                else:
                    if not OWM_API_KEY:
                        st.info("Add your OpenWeatherMap key to st.secrets or the OWM_API_KEY env var to enable forecasts.")
                    else:
                        st.write("No weather data available.")

            st.markdown("</div>", unsafe_allow_html=True)

### ---- Power Rankings Tab ----
with tabs[1]:
    st.subheader("📊 Elo Power Rankings (with current injury adjustments)")

    rows = []
    for abbr, full in NFL_FULL_NAMES.items():
        base = ratings.get(full, BASE_ELO)
        inj = fetch_injuries_espn(abbr)
        pen = injury_adjustment(inj)
        rows.append({
            "Team": full,
            "Abbr": abbr,
            "Base Elo": round(base, 1),
            "Injury Penalty": pen,
            "Adjusted Elo": round(base + pen, 1),
        })
    pr_df = pd.DataFrame(rows).sort_values("Adjusted Elo", ascending=False).reset_index(drop=True)

    st.dataframe(
        pr_df,
        use_container_width=True,
        hide_index=True,
    )

### ---- Pick Winners Tab ----
with tabs[2]:
    st.subheader("✅ Pick Winners (injury + weather aware)")
    week_for_picks = st.selectbox(
        "Week", sorted(sched_df['week'].dropna().unique().astype(int).tolist()),
        index=max(0, len(available_weeks)-1), key="pick_week"
    )
    games = sched_df[sched_df['week'] == week_for_picks]

    picks = []
    for _, row in games.iterrows():
        team_home, team_away = map_team_name(row[HOME_COL]), map_team_name(row[AWAY_COL])
        abbr_home, abbr_away = get_abbr(team_home), get_abbr(team_away)
        elo_home, elo_away = ratings.get(team_home, BASE_ELO), ratings.get(team_away, BASE_ELO)
        # Injuries
        inj_home = fetch_injuries_espn(abbr_home)
        inj_away = fetch_injuries_espn(abbr_away)
        elo_home_adj = elo_home + injury_adjustment(inj_home)
        elo_away_adj = elo_away + injury_adjustment(inj_away)
        # Probabilities
        prob_home = expected_score(elo_home_adj + HOME_ADVANTAGE, elo_away_adj)
        prob_away = 1 - prob_home
        # Weather totals
        kickoff_unix = default_kickoff_unix(row.get("date"))
        weather = get_weather(team_home, kickoff_unix) if kickoff_unix else None
        avg_total = NFL_AVG_TOTALS.get(row.get("season"), overall_avg)
        if weather:
            avg_total += weather_adjustment(weather)
        elo_diff = (elo_home_adj + HOME_ADVANTAGE) - elo_away_adj
        spread_home = elo_diff / 25
        pred_home = round((avg_total / 2) + (spread_home / 2), 1)
        pred_away = round((avg_total / 2) - (spread_home / 2), 1)

        default_pick = team_home if prob_home >= 0.5 else team_away
        pick = st.selectbox(
            f"{team_away} @ {team_home}", [team_home, team_away], index=0 if default_pick==team_home else 1
        )
        picks.append({
            "Week": week_for_picks,
            "Away": team_away,
            "Home": team_home,
            "Home Win %": round(prob_home*100, 1),
            "Away Win %": round(prob_away*100, 1),
            "Pred Home": pred_home,
            "Pred Away": pred_away,
            "Your Pick": pick,
        })

    if picks:
        picks_df = pd.DataFrame(picks)
        st.dataframe(picks_df, use_container_width=True, hide_index=True)
        csv = picks_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Picks (CSV)",
            data=csv,
            file_name=f"picks_week_{week_for_picks}.csv",
            mime="text/csv",
        )

# ---- Footer tip ----
st.caption("Tip: Add your OpenWeatherMap key to `.streamlit/secrets.toml` as `OWM_API_KEY = '...''` for live forecasts. ESPN injuries are auto-fetched and cached for 15 minutes.")
