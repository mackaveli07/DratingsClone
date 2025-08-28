# NFL Elo Projections App — Neon + Adjustments + Projections + Scoreboard
import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
import os, base64, requests, datetime, pytz, math

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
    actual1 = 1 if score1 > score2 else 0
    margin = abs(score1 - score2) or 1
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
            "competition": comp,   # full data for possession + lastPlay
        })

    return games

# --- The rest of the file continues with injuries, weather, headers, model utils, and tabs (Matchups, Power Rankings, Pick Winners, Scoreboard) ---


### ---------- INJURIES ----------
ESPN_TEAM_IDS = {
    "ARI":22,"ATL":1,"BAL":33,"BUF":2,"CAR":29,"CHI":3,"CIN":4,"CLE":5,"DAL":6,"DEN":7,"DET":8,"GB":9,
    "HOU":34,"IND":11,"JAX":30,"KC":12,"LV":13,"LAC":24,"LA":14,"MIA":15,"MIN":16,"NE":17,"NO":18,"NYG":19,"NYJ":20,"PHI":21,
    "PIT":23,"SF":25,"SEA":26,"TB":27,"TEN":10,"WAS":28
}

def fetch_injuries_espn(team_abbr):
    team_id = ESPN_TEAM_IDS.get(team_abbr)
    if not team_id:
        return []
    url = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/{team_id}/injuries"
    try:
        r = requests.get(url, timeout=6); r.raise_for_status()
        data = r.json()
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
# A small stadium set; add more as needed
STADIUMS = {
    "Buffalo Bills":{"lat":42.7738,"lon":-78.7868},
    "Green Bay Packers":{"lat":44.5013,"lon":-88.0622},
    "Chicago Bears":{"lat":41.8625,"lon":-87.6166},
    "Kansas City Chiefs":{"lat":39.0490,"lon":-94.4840},
    "New England Patriots":{"lat":42.0909,"lon":-71.2643},
    "Philadelphia Eagles":{"lat":39.9008,"lon":-75.1675}
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
    if dt_diff > 432000:  # > 5 days off
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
                    align-items:center; justify-content:center; gap:16px;'>
            """ + logo_html + f"""
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
                    align-items:center; justify-content:center; gap:10px;'>
        """ + logo_html + f"""
            <h2 style='color:white; margin:0;'>{icon} {text}</h2>
            {logo_html}
        </div>
        """,
        unsafe_allow_html=True
    )

### ---------- MAIN ----------
st.set_page_config(page_title="NFL Elo Projections", layout="wide")
nfl_header("NFL Elo Projections")

# Load data
try:
    hist_df = pd.read_excel(EXCEL_FILE, sheet_name=HIST_SHEET)
    sched_df = pd.read_excel(EXCEL_FILE, sheet_name=SCHEDULE_SHEET)
except Exception as e:
    st.error(f"Error loading Excel: {e}")
    st.stop()

# Compute smoothed NFL average totals by season (for projections)
NFL_AVG_TOTALS, overall_avg, alpha = {}, 44, 50
if {"score1","score2","season"} <= set(hist_df.columns):
    hist_df["total_points"] = (hist_df["score1"].fillna(0) + hist_df["score2"].fillna(0))
    overall_avg = float(hist_df["total_points"].mean()) if len(hist_df) else 44.0
    grouped = hist_df.groupby("season")["total_points"].agg(["mean","count"])
    for s, r in grouped.iterrows():
        NFL_AVG_TOTALS[int(s)] = (r["mean"]*r["count"] + overall_avg*alpha) / (r["count"] + alpha)

ratings = run_elo_pipeline(hist_df)
HOME_COL, AWAY_COL = "team2", "team1"

tabs = st.tabs(["Matchups", "Power Rankings", "Pick Winners", "Scoreboard"])

# --- Matchups Tab ---
with tabs[0]:
    # Safe week handling (avoid NaN -> int crash)
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

        # Fetch adjustments
        home_inj = fetch_injuries_espn(abbr_home) if abbr_home else []
        away_inj = fetch_injuries_espn(abbr_away) if abbr_away else []
        kickoff = default_kickoff_unix(row.get("date"))
        # Weather at HOME stadium; apply to BOTH teams (shared conditions)
        weather = get_weather(team_home, kickoff)

        # SILENT adjusted Elos for projection only
        adj_home = ratings.get(team_home, BASE_ELO) + injury_adjustment(home_inj) + weather_adjustment(weather)
        adj_away = ratings.get(team_away, BASE_ELO) + injury_adjustment(away_inj) + weather_adjustment(weather)

        # Win probabilities (include home advantage at projection time)
        win_prob_home = expected_score(adj_home + HOME_ADVANTAGE, adj_away)
        win_prob_away = 1 - win_prob_home

        # Projected score split using smoothed season total
        season_val = row.get("season")
        try:
            season_int = int(season_val) if pd.notna(season_val) else max(NFL_AVG_TOTALS.keys(), default=2025)
        except Exception:
            season_int = max(NFL_AVG_TOTALS.keys(), default=2025)
        total_pts = NFL_AVG_TOTALS.get(season_int, overall_avg)
        proj_home = int(round(win_prob_home * total_pts))
        proj_away = int(round(win_prob_away * total_pts))

        # Card
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
        with col_mid:
            st.markdown(f"<h1 style='text-align:center; margin:0;'>{proj_away} – {proj_home}</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align:center; margin:4px 0 0;'>Projected Score</p>", unsafe_allow_html=True)
        with col2:
            safe_logo(abbr_home, 120)
            st.markdown(f"<div style='text-align:center'>{neon_text(team_home, abbr_home, 28)}</div>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center; margin-top:6px;'>Win %: {win_prob_home:.1%}</p>", unsafe_allow_html=True)

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

with tabs[1]:
    nfl_subheader("Elo Power Rankings", "📊")

    # Build rankings with current adjustments
    adjusted_ratings = {}
    for team_full in NFL_FULL_NAMES.values():
        abbr = get_abbr(team_full)
        base = ratings.get(team_full, BASE_ELO)

        # Fetch adjustments
        inj = fetch_injuries_espn(abbr) if abbr else []
        kickoff = default_kickoff_unix(datetime.datetime.now())  # today baseline
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

    # Render table with logos + neon
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
            st.markdown(
                f"{neon_text(team, abbr, 20)} – **{elo_val}**",
                unsafe_allow_html=True
            )
        st.markdown("---")
# --- Pick Winners Tab ---
with tabs[2]:
    nfl_subheader("Weekly Pick’em", "📝")
    week_series_num = pd.to_numeric(sched_df.get("week"), errors="coerce")
    available_weeks = sorted(set(week_series_num.dropna().astype(int).tolist()))
    week = st.selectbox("Select Week", available_weeks, key="week_picks")
    games = sched_df.loc[(week_series_num == week).fillna(False)]
    picks = {}
    for _, row in games.iterrows():
        t_home, t_away = map_team_name(row.get(HOME_COL)), map_team_name(row.get(AWAY_COL))
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


# --- Scoreboard Tab ---
with tabs[3]:
    nfl_subheader("NFL Scoreboard", "🏟️")

    # Inject shared CSS
    st.markdown("""
    <style>
    .score-card {
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        text-align: center;
    }
    .team-block {
        flex: 1;
        text-align: center;
    }
    .team-score {
        font-size: 40px;
        font-weight: bold;
        text-shadow: 0 0 6px black;
    }
    .live-pill {
        background: #dc2626;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
    }
    .final-pill {
        background: #16a34a;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
    }
    .scheduled-pill {
        background: #2563eb;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
    }
    .info-box {
        background: rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 6px 12px;
        margin-top: 10px;
        font-size: 14px;
        font-style: italic;
    }
    .status-box {
        text-align: center;
        font-size: 16px;
        margin-top: 6px;
        color: #e5e7eb;
    }
    </style>
    """, unsafe_allow_html=True)

    games = fetch_nfl_scores()
    if not games:
        st.info("No NFL games today or scheduled.")

    for game in games:
        away, home = game["away"], game["home"]
        state, status_text = game.get("state","pre"), game.get("status","")
        comp = game.get("competition")
        situation = comp.get("situation", {}) if comp else {}

        possession_id = situation.get("possession", {}).get("id")
        last_play = situation.get("lastPlay", {}).get("text","")
        down = situation.get("down")
        distance = situation.get("distance")
        yard_line = situation.get("yardLine")
        desc = situation.get("shortDownDistanceText")

        # Build drive summary text
        drive_summary = None
        if down and distance:
            drive_summary = f"{desc} on {yard_line or '??'}"
        elif desc:
            drive_summary = desc

        # Game clock & quarter
        status_obj = comp.get("status", {}) if comp else {}
        period = status_obj.get("period")
        clock = status_obj.get("displayClock", "")
        clock_text = None
        if state == "in" and period:
            clock_text = f"Q{period} – {clock}"

        # Status pill
        if state == "in":
            pill_html = "<span class='live-pill'>LIVE</span>"
        elif state == "post":
            pill_html = "<span class='final-pill'>FINAL</span>"
        else:
            pill_html = f"<span class='scheduled-pill'>{status_text}</span>"

        # Highlight winner if final
        highlight_home = state == "post" and int(home.get("score",0)) > int(away.get("score",0))
        highlight_away = state == "post" and int(away.get("score",0)) > int(home.get("score",0))

        st.markdown(f"""
        <div class="score-card">
            <div style="display:flex; align-items:center; justify-content:space-between;">
                <div class="team-block">
                    <img src="{away['team']['logo']}" width="80" />
                    <div>{neon_text(away['team']['displayName'], away['team']['abbreviation'], 26)}</div>
                    <div class="team-score" style="color:{'#16a34a' if highlight_away else 'white'};">
                        {'🏈 ' if str(away['team'].get('id')) == str(possession_id) else ''}{away.get('score','0')}
                    </div>
                </div>

                <div style="flex:0.7; text-align:center;">
                    {pill_html}
                    {f"<div class='status-box'>{clock_text}</div>" if clock_text else ""}
                </div>

                <div class="team-block">
                    <img src="{home['team']['logo']}" width="80" />
                    <div>{neon_text(home['team']['displayName'], home['team']['abbreviation'], 26)}</div>
                    <div class="team-score" style="color:{'#16a34a' if highlight_home else 'white'};">
                        {'🏈 ' if str(home['team'].get('id')) == str(possession_id) else ''}{home.get('score','0')}
                    </div>
                </div>
            </div>
            {f'<div class="info-box">📋 {drive_summary}</div>' if drive_summary else ''}
            {f'<div class="info-box">📝 {last_play}</div>' if last_play else ''}
        </div>
        """, unsafe_allow_html=True)
