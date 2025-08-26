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
        st.markdown(
            f"<div style='width:{width}px; height:{width}px; background:#e5e7eb; "
            f"display:flex; align-items:center; justify-content:center; border-radius:50%; "
            f"font-size:12px; color:#475569;'>{abbr or '?'}</div>",
            unsafe_allow_html=True,
        )

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

# ---------- SCOREBOARD HELPERS ----------
@st.cache_data(ttl=5)
def fetch_nfl_scores():
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    resp = requests.get(url)
    if resp.status_code != 200:
        return []
    data = resp.json()
    games = []
    today = datetime.date.today().isoformat()
    for event in data.get("events", []):
        if event.get("date", "").split("T")[0] != today:
            continue
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
        info = {
            "quarter": f"Q{status.get('period', 'N/A')}",
            "clock": status.get("displayClock", ""),
            "possession": situation.get("possession", {}).get("displayName", "")
        }
        games.append({
            "away": away,
            "home": home,
            "info": info
        })
    return games


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

### ---------- WEATHER ----------
STADIUMS={"Buffalo Bills":{"lat":42.7738,"lon":-78.7868},"Green Bay Packers":{"lat":44.5013,"lon":-88.0622},
"Chicago Bears":{"lat":41.8625,"lon":-87.6166},"Kansas City Chiefs":{"lat":39.0490,"lon":-94.4840},
"New England Patriots":{"lat":42.0909,"lon":-71.2643},"Philadelphia Eagles":{"lat":39.9008,"lon":-75.1675}}

OWM_API_KEY = os.getenv("OWM_API_KEY","")

def get_weather(team,kickoff_unix):
    if team not in STADIUMS or not OWM_API_KEY: return None
    lat,lon=STADIUMS[team]["lat"],STADIUMS[team]["lon"]
    url=f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OWM_API_KEY}&units=imperial"
    try:
        resp=requests.get(url,timeout=6); resp.raise_for_status(); data=resp.json()
    except Exception: return None
    forecasts=data.get("list",[]); 
    if not forecasts: return None
    closest=min(forecasts,key=lambda x:abs(int(x.get("dt",0))-int(kickoff_unix)))
    dt_diff=abs(int(closest.get("dt",0))-int(kickoff_unix))
    if dt_diff>432000: return None
    return {"temp":closest["main"]["temp"],"wind_speed":closest["wind"]["speed"],"condition":closest["weather"][0]["main"]}

def weather_adjustment(weather):
    pen=0
    if weather["wind_speed"]>20: pen-=2
    if weather["condition"].lower() in ["rain","snow"]: pen-=3
    if weather["temp"]<25: pen-=1
    return pen

def default_kickoff_unix(game_date):
    if isinstance(game_date,str):
        try: game_date=datetime.datetime.strptime(game_date,"%Y-%m-%d")
        except: return 0
    elif not isinstance(game_date,datetime.datetime): return 0
    est=pytz.timezone("US/Eastern")
    kickoff=est.localize(datetime.datetime(game_date.year,game_date.month,game_date.day,13,0,0))
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
st.set_page_config(page_title="NFL Elo Projections",layout="wide")
nfl_header("NFL Elo Projections")




try:
    hist_df=pd.read_excel(EXCEL_FILE,sheet_name=HIST_SHEET)
    sched_df=pd.read_excel(EXCEL_FILE,sheet_name=SCHEDULE_SHEET)
except Exception as e:
    st.error(f"Error loading Excel: {e}"); st.stop()

NFL_AVG_TOTALS={}; overall_avg=44; alpha=50
if {"score1","score2","season"} <= set(hist_df.columns):
    hist_df["total_points"]=hist_df["score1"]+hist_df["score2"]
    overall_avg=hist_df["total_points"].mean()
    grouped=hist_df.groupby("season")["total_points"].agg(["mean","count"])
    for s,r in grouped.iterrows():
        NFL_AVG_TOTALS[s]=(r["mean"]*r["count"]+overall_avg*alpha)/(r["count"]+alpha)

ratings=run_elo_pipeline(hist_df)
HOME_COL,AWAY_COL="team2","team1"

tabs=st.tabs(["Matchups","Power Rankings","Pick Winners","Scoreboard"])


# --- Matchups Tab ---
with tabs[0]:
    available_weeks = sorted(sched_df['week'].dropna().unique().astype(int).tolist())
    selected_week = st.selectbox("Select Week", available_weeks, index=max(0, len(available_weeks)-1), key="week_matchups")
    week_games = sched_df[sched_df['week'] == selected_week]

    for _, row in week_games.iterrows():
        team_home, team_away = map_team_name(row[HOME_COL]), map_team_name(row[AWAY_COL])
        elo_home, elo_away = ratings.get(team_home, BASE_ELO), ratings.get(team_away, BASE_ELO)

        inj_home, inj_away = fetch_injuries_espn(get_abbr(team_home)), fetch_injuries_espn(get_abbr(team_away))
        elo_home_adj, elo_away_adj = elo_home + injury_adjustment(inj_home), elo_away + injury_adjustment(inj_away)

        prob_home = expected_score(elo_home_adj + HOME_ADVANTAGE, elo_away_adj)
        prob_away = 1 - prob_home

        kickoff_unix = default_kickoff_unix(row.get("date"))
        weather = get_weather(team_home, kickoff_unix) if kickoff_unix else None

        avg_total = NFL_AVG_TOTALS.get(row.get("season"), overall_avg)
        if weather: avg_total += weather_adjustment(weather)

        elo_diff = (elo_home_adj + HOME_ADVANTAGE) - elo_away_adj
        spread_home = elo_diff / 25
        predicted_home_score = round((avg_total / 2) + (spread_home / 2), 1)
        predicted_away_score = round((avg_total / 2) - (spread_home / 2), 1)

        # Card container
        st.markdown("<div style='background: rgba(255,255,255,0.12); backdrop-filter: blur(14px); border-radius: 24px; padding: 25px; margin: 22px 0; box-shadow: 0 8px 25px rgba(0,0,0,0.25);'>", unsafe_allow_html=True)

        col1, col_mid, col2 = st.columns([2, 3, 2])
        with col1:
            safe_logo(get_abbr(team_away), 120)
            st.markdown(f"<h4 style='text-align:center'>{team_away}</h4>", unsafe_allow_html=True)
        with col_mid:
            st.markdown(f"<h1 style='text-align:center; margin:0;'>{predicted_away_score} – {predicted_home_score}</h1>", unsafe_allow_html=True)
            prob_html = f"<div style='width:100%; background:#e5e7eb; border-radius:12px; overflow:hidden; height:20px; margin:15px 0;'><div style='width:{prob_away*100:.1f}%; background:#ef4444; height:100%; float:left;'></div><div style='width:{prob_home*100:.1f}%; background:#2563eb; height:100%; float:right;'></div></div><p style='text-align:center; font-size:14px; color:#6b7280;'>{team_away} {prob_away*100:.1f}% | {prob_home*100:.1f}% {team_home}</p>"
            st.markdown(prob_html, unsafe_allow_html=True)
        with col2:
            safe_logo(get_abbr(team_home), 120)
            st.markdown(f"<h4 style='text-align:center'>{team_home}</h4>", unsafe_allow_html=True)

        with st.expander("🩺 Injury Report"):
            if inj_home or inj_away:
                st.markdown(f"**{team_home}**:")
                for p in inj_home[:6]: st.write(f"- {p['name']} ({p['position']}): {p['status']}")
                st.markdown(f"**{team_away}**:")
                for p in inj_away[:6]: st.write(f"- {p['name']} ({p['position']}): {p['status']}")
            else:
                st.write("No major injuries reported.")

        with st.expander("🌦️ Weather Forecast"):
            if weather:
                st.write(f"🌡️ Temp: {weather['temp']}°F")
                st.write(f"💨 Wind: {weather['wind_speed']} mph")
                st.write(f"🌦️ Condition: {weather['condition']}")
            else:
                st.write("No weather data available (forecast only within 5 days).")

        st.markdown("</div>", unsafe_allow_html=True)

# --- Power Rankings Tab ---
with tabs[1]:
    nfl_subheader("Elo Power Rankings (with injury adjustments)", "📊")
    rows = []
    for abbr, full in NFL_FULL_NAMES.items():
        base = ratings.get(full, BASE_ELO); inj = fetch_injuries_espn(abbr); pen = injury_adjustment(inj); adj = base + pen
        rows.append({"Team": full,"Abbr": abbr,"Adjusted Elo": round(adj, 1)})
    pr_df = pd.DataFrame(rows).sort_values("Adjusted Elo", ascending=False).reset_index(drop=True)

    def render_row(row):
        abbr = row["Abbr"]; logo_path = f"Logos/{abbr}.png"
        logo_html = (f"<img src='data:image/png;base64,{base64.b64encode(open(logo_path,'rb').read()).decode()}' style='height:28px; vertical-align:middle; margin-right:8px;' />" if os.path.exists(logo_path) else f"<span style='margin-right:8px; font-weight:600;'>{abbr}</span>")
        bar_width = (row["Adjusted Elo"] - 1300) / 4; bar_width = max(0, min(bar_width, 100))
        bar_html = f"<div style='background:#e5e7eb; border-radius:8px; width:100%; height:14px; overflow:hidden;'><div style='width:{bar_width:.1f}%; background:#2563eb; height:100%;'></div></div>"
        return f"<div style='display:flex; align-items:center; justify-content:space-between; padding:6px 0;'><div style='display:flex; align-items:center;'>{logo_html}<span style='font-weight:600;'>{row['Team']}</span></div><div style='flex:1; margin:0 16px;'>{bar_html}</div><div style='width:60px; text-align:right; font-weight:600;'>{row['Adjusted Elo']}</div></div>"

    leaderboard_html = "<div style='background:rgba(255,255,255,0.08); border-radius:18px; padding:18px;'>"
    for _, r in pr_df.iterrows(): leaderboard_html += render_row(r)
    leaderboard_html += "</div>"
    st.markdown(leaderboard_html, unsafe_allow_html=True)

# --- Pick Winners Tab ---
with tabs[2]:
    nfl_subheader("Weekly Pick’em", "📝")
    week=st.selectbox("Select Week",sorted(sched_df['week'].dropna().unique().astype(int)),key="week_picks")
    games=sched_df[sched_df['week']==week]; picks={}
    for _,row in games.iterrows():
        t_home,t_away=map_team_name(row[HOME_COL]),map_team_name(row[AWAY_COL])
        abbr_home,abbr_away=get_abbr(t_home),get_abbr(t_away)
        st.markdown("<div style='background:rgba(255,255,255,0.08); border-radius:18px; padding:16px; margin:12px 0;'>",unsafe_allow_html=True)
        c1,c2,c3=st.columns([3,2,3])
        with c1: safe_logo(abbr_away,80); st.markdown(f"<h5>{t_away}</h5>",unsafe_allow_html=True)
        with c2: st.markdown("<h5 style='text-align:center'>Your Pick ➡️</h5>",unsafe_allow_html=True)
        with c3: safe_logo(abbr_home,80); st.markdown(f"<h5>{t_home}</h5>",unsafe_allow_html=True)
        choice=st.radio("",[t_away,t_home],horizontal=True,key=f"pick_{t_home}_{t_away}"); picks[f"{t_away} @ {t_home}"]=choice
        st.markdown("</div>",unsafe_allow_html=True)
    if st.button("💾 Save Picks"):
        try:
            df=pd.DataFrame([{"Game":g,"Pick":p} for g,p in picks.items()])
            with pd.ExcelWriter(EXCEL_FILE,mode="a",engine="openpyxl",if_sheet_exists="replace") as w:
                df.to_excel(w,sheet_name=PICKS_SHEET,index=False)
            st.success("✅ Picks saved!")
        except Exception as e: st.error(f"Error saving picks: {e}")

# --- Scoreboard Tab ---
with tabs[3]:
    nfl_subheader("Live NFL Scoreboard", "🏟️")
    games = fetch_nfl_scores()
    if not games:
        st.info("No NFL games today.")
    for game in games:
        away, home, info = game["away"], game["home"], game["info"]
        col1, col2, col3 = st.columns([3,2,3])
        with col1:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #013369, #d50a0a); border-radius: 10px; padding: 10px; text-align:center;'>
                    <h3>{away['team']['displayName']}</h3>
                    <img src="{away['team']['logo']}" width="100" />
                    <p style='font-size: 36px; margin: 10px 0;'>{away.get('score', '0')}</p>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div style='text-align:center;'>
                    <p><strong>{info['quarter']} {info['clock']}</strong></p>
                    <p>🟢 Possession: {info['possession']}</p>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #d50a0a, #013369); border-radius: 10px; padding: 10px; text-align:center;'>
                    <h3>{home['team']['displayName']}</h3>
                    <img src="{home['team']['logo']}" width="100" />
                    <p style='font-size: 36px; margin: 10px 0;'>{home.get('score', '0')}</p>
                </div>
            """, unsafe_allow_html=True)
