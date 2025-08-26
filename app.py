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

### ---------- BACKGROUND ----------
def set_background(image_file="Shield.png"):
    if os.path.exists(image_file):
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

### ---------- HELPERS ----------
def map_team_name(name):
    if not name: return "Unknown"
    name = str(name).strip()
    if name.upper() in NFL_FULL_NAMES: return NFL_FULL_NAMES[name.upper()]
    for full in NFL_FULL_NAMES.values():
        if name.lower() == full.lower(): return full
    return name

def get_abbr(team_full):
    for abbr, full in NFL_FULL_NAMES.items():
        if full == team_full: return abbr
    return None

def safe_logo(abbr, width=64):
    path = f"Logos/{abbr}.png"
    if abbr and os.path.exists(path):
        st.image(path, width=width)
    else:
        url = f"https://a.espncdn.com/i/teamlogos/nfl/500/{abbr.lower()}.png"
        st.image(url, width=width)

### ---------- ELO ----------
def expected_score(r1, r2):
    return 1 / (1 + 10 ** ((r2 - r1) / 400))

def regress_preseason(elo_ratings, reg=0.65, base=BASE_ELO):
    for t in elo_ratings:
        elo_ratings[t] = base + reg * (elo_ratings[t] - base)

def update_ratings(elo_ratings, team1, team2, score1, score2, home_team):
    r1, r2 = elo_ratings[team1], elo_ratings[team2]
    if home_team == team1: r1 += HOME_ADVANTAGE
    elif home_team == team2: r2 += HOME_ADVANTAGE
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
            if i > 0: regress_preseason(elo_ratings)
            for _, row in df[df["season"]==s].iterrows():
                t1, t2 = map_team_name(row.get("team1")), map_team_name(row.get("team2"))
                update_ratings(elo_ratings, t1, t2, row.get("score1",0), row.get("score2",0), map_team_name(row.get("home_team",t2)))
    return dict(elo_ratings)

### ---------- INJURIES ----------
ESPN_TEAM_IDS = { "ARI":22,"ATL":1,"BAL":33,"BUF":2,"CAR":29,"CHI":3,"CIN":4,"CLE":5,"DAL":6,"DEN":7,"DET":8,"GB":9,
"HOU":34,"IND":11,"JAX":30,"KC":12,"LV":13,"LAC":24,"LA":14,"MIA":15,"MIN":16,"NE":17,"NO":18,"NYG":19,"NYJ":20,"PHI":21,
"PIT":23,"SF":25,"SEA":26,"TB":27,"TEN":10,"WAS":28 }

def fetch_injuries_espn(team_abbr):
    team_id = ESPN_TEAM_IDS.get(team_abbr)
    if not team_id: return []
    url = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/{team_id}/injuries"
    try:
        r = requests.get(url, timeout=5); r.raise_for_status()
        data = r.json()
    except Exception: return []
    players = []
    for e in data.get("entries", []):
        players.append({
            "name": e.get("athlete",{}).get("displayName"),
            "position": e.get("position",{}).get("abbreviation"),
            "status": e.get("status",{}).get("type","")
        })
    return players

def injury_adjustment(players):
    penalty=0
    for p in players:
        s=(p.get("status") or "").lower(); pos=(p.get("position") or "").upper()
        if pos=="QB" and s in ["out","doubtful"]: penalty-=50
        elif pos in ["RB","WR","TE"] and s in ["out","doubtful"]: penalty-=15
        elif s in ["out","doubtful"]: penalty-=10
    return penalty

### ---------- SCOREBOARD ----------
@st.cache_data(ttl=30)
def fetch_nfl_scores():
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    games = []
    for event in data.get("events", []):
        comp = event.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue
        away = next((t for t in competitors if t["homeAway"] == "away"), None)
        home = next((t for t in competitors if t["homeAway"] == "home"), None)
        if not away or not home:
            continue
        status = comp.get("status", {})
        situation = comp.get("situation", {})
        games.append({
            "away": away,
            "home": home,
            "quarter": f"Q{status.get('period', 'N/A')}",
            "clock": status.get("displayClock", ""),
            "possession": situation.get("possession", {}).get("displayName", "")
        })
    return games

### ---------- MAIN ----------
st.set_page_config(page_title="NFL Elo Projections", layout="wide")
set_background("Shield.png")

st.markdown("""<h1 style='text-align:center; color:white;'>NFL Elo Projections</h1>""", unsafe_allow_html=True)

try:
    hist_df=pd.read_excel(EXCEL_FILE,sheet_name=HIST_SHEET)
    sched_df=pd.read_excel(EXCEL_FILE,sheet_name=SCHEDULE_SHEET)
except Exception as e:
    st.error(f"Error loading Excel: {e}"); st.stop()

ratings=run_elo_pipeline(hist_df)
HOME_COL,AWAY_COL="team2","team1"

# Tabs
tabs = st.tabs(["Matchups","Power Rankings","Pick Winners","Scoreboard"])

# --- Matchups Tab ---
with tabs[0]:
    st.markdown("<h2 style='color:white;'>Upcoming Matchups</h2>", unsafe_allow_html=True)
    week = st.selectbox("Select Week", sorted(sched_df['week'].dropna().unique().astype(int)), key="week_matchups")
    games = sched_df[sched_df['week'] == week]
    for _, row in games.iterrows():
        t_home, t_away = map_team_name(row[HOME_COL]), map_team_name(row[AWAY_COL])
        abbr_home, abbr_away = get_abbr(t_home), get_abbr(t_away)

        r_home, r_away = ratings.get(t_home, BASE_ELO), ratings.get(t_away, BASE_ELO)
        p_home = expected_score(r_home, r_away)
        p_away = 1 - p_home

        st.markdown("<div style='background:rgba(255,255,255,0.08); border-radius:18px; padding:16px; margin:12px 0;'>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([3,2,3])
        with col1:
            safe_logo(abbr_away, 80)
            st.markdown(f"<h4 style='color:white;'>{t_away}</h4>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:white;'>Elo: {r_away:.0f}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:white;'>Win %: {p_away*100:.1f}%</p>", unsafe_allow_html=True)
        with col2:
            st.markdown("<h4 style='text-align:center; color:white;'>@</h4>", unsafe_allow_html=True)
        with col3:
            safe_logo(abbr_home, 80)
            st.markdown(f"<h4 style='color:white;'>{t_home}</h4>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:white;'>Elo: {r_home:.0f}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:white;'>Win %: {p_home*100:.1f}%</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# --- Power Rankings Tab ---
with tabs[1]:
    st.markdown("<h2 style='color:white;'>Elo Power Rankings</h2>", unsafe_allow_html=True)
    rows = []
    for abbr, full in NFL_FULL_NAMES.items():
        base = ratings.get(full, BASE_ELO)
        inj = fetch_injuries_espn(abbr)
        pen = injury_adjustment(inj)
        adj = base + pen
        rows.append({"Team": full,"Abbr": abbr,"Adjusted Elo": round(adj, 1)})
    pr_df = pd.DataFrame(rows).sort_values("Adjusted Elo", ascending=False).reset_index(drop=True)
    st.dataframe(pr_df)

# --- Pick Winners Tab ---
with tabs[2]:
    st.markdown("<h2 style='color:white;'>Weekly Pick’em</h2>", unsafe_allow_html=True)
    week=st.selectbox("Select Week",sorted(sched_df['week'].dropna().unique().astype(int)),key="week_picks")
    games=sched_df[sched_df['week']==week]; picks={}
    for _,row in games.iterrows():
        t_home,t_away=map_team_name(row[HOME_COL]),map_team_name(row[AWAY_COL])
        abbr_home,abbr_away=get_abbr(t_home),get_abbr(t_away)
        col1,col2,col3=st.columns([3,2,3])
        with col1: safe_logo(abbr_away,80); st.markdown(f"<h5 style='color:white;'>{t_away}</h5>",unsafe_allow_html=True)
        with col2: st.markdown("<h5 style='text-align:center; color:white;'>Your Pick ➡️</h5>",unsafe_allow_html=True)
        with col3: safe_logo(abbr_home,80); st.markdown(f"<h5 style='color:white;'>{t_home}</h5>",unsafe_allow_html=True)
        choice=st.radio("",[t_away,t_home],horizontal=True,key=f"pick_{t_home}_{t_away}"); picks[f"{t_away} @ {t_home}"]=choice
    if st.button("💾 Save Picks"):
        try:
            df=pd.DataFrame([{"Game":g,"Pick":p} for g,p in picks.items()])
            with pd.ExcelWriter(EXCEL_FILE,mode="a",engine="openpyxl",if_sheet_exists="replace") as w:
                df.to_excel(w,sheet_name=PICKS_SHEET,index=False)
            st.success("✅ Picks saved!")
        except Exception as e: st.error(f"Error saving picks: {e}")

# --- Scoreboard Tab ---
with tabs[3]:
    st.markdown("<h2 style='text-align:center; color:white;'>Live NFL Scoreboard</h2>", unsafe_allow_html=True)
    games = fetch_nfl_scores()
    if not games:
        st.info("No NFL games available.")
    for game in games:
        away, home = game["away"], game["home"]
        quarter, clock, poss = game["quarter"], game["clock"], game["possession"]
        col1, col2, col3 = st.columns([3,2,3])
        with col1:
            st.image(away['team']['logo'], width=100)
            st.markdown(f"<h3 style='color:white;'>{away['team']['displayName']}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:36px; color:white;'>{away.get('score','0')}</p>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<p style='text-align:center; color:white;'><strong>{quarter} {clock}</strong></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center; color:white;'>Possession: {poss}</p>", unsafe_allow_html=True)
        with col3:
            st.image(home['team']['logo'], width=100)
            st.markdown(f"<h3 style='color:white;'>{home['team']['displayName']}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:36px; color:white;'>{home.get('score','0')}</p>", unsafe_allow_html=True)
