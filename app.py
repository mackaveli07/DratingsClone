# app_scraper.py
import streamlit as st
import pandas as pd
import numpy as np
import time
import re
import random
from collections import defaultdict
from difflib import get_close_matches

# ---------- SCRAPER DEPENDENCIES ----------
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

# Selenium / undetected-chromedriver
try:
    import undetected_chromedriver as uc
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    SELENIUM_OK = True
except Exception:
    SELENIUM_OK = False

# ---------- CONFIG ----------
SPORT_KEY = "americanfootball_nfl"   # for labeling only
BASE_ELO = 1500
K = 20
HOME_ADVANTAGE = 65

EXCEL_FILE = "games.xlsx"
HIST_SHEET = "games"
SCHEDULE_SHEET = "2025 schedule"

TEAM_LOGOS = {
   "Arizona Cardinals": "https://upload.wikimedia.org/wikipedia/commons/4/47/Arizona_Cardinals_logo.svg",
    "Atlanta Falcons": "https://upload.wikimedia.org/wikipedia/commons/5/5e/Atlanta_Falcons_logo.svg",
    "Baltimore Ravens": "https://upload.wikimedia.org/wikipedia/commons/2/2c/Baltimore_Ravens_logo.svg",
    "Buffalo Bills": "https://upload.wikimedia.org/wikipedia/commons/1/1a/Buffalo_Bills_logo.svg",
    "Carolina Panthers": "https://upload.wikimedia.org/wikipedia/commons/3/3b/Carolina_Panthers_logo.svg",
    "Chicago Bears": "https://upload.wikimedia.org/wikipedia/commons/5/5f/Chicago_Bears_logo.svg",
    "Cincinnati Bengals": "https://upload.wikimedia.org/wikipedia/commons/8/8f/Cincinnati_Bengals_logo.svg",
    "Cleveland Browns": "https://upload.wikimedia.org/wikipedia/commons/4/4f/Cleveland_Browns_logo.svg",
    "Dallas Cowboys": "https://upload.wikimedia.org/wikipedia/commons/9/97/Dallas_Cowboys_logo.svg",
    "Denver Broncos": "https://upload.wikimedia.org/wikipedia/commons/4/44/Denver_Broncos_logo.svg",
    "Detroit Lions": "https://upload.wikimedia.org/wikipedia/commons/2/2f/Detroit_Lions_logo.svg",
    "Green Bay Packers": "https://upload.wikimedia.org/wikipedia/commons/5/5c/Green_Bay_Packers_logo.svg",
    "Houston Texans": "https://upload.wikimedia.org/wikipedia/commons/2/2f/Houston_Texans_logo.svg",
    "Indianapolis Colts": "https://upload.wikimedia.org/wikipedia/commons/5/5d/Indianapolis_Colts_logo.svg",
    "Jacksonville Jaguars": "https://upload.wikimedia.org/wikipedia/commons/0/0c/Jacksonville_Jaguars_logo.svg",
    "Kansas City Chiefs": "https://upload.wikimedia.org/wikipedia/commons/6/6e/Kansas_City_Chiefs_logo.svg",
    "Las Vegas Raiders": "https://upload.wikimedia.org/wikipedia/commons/5/5c/Las_Vegas_Raiders_logo.svg",
    "Los Angeles Chargers": "https://upload.wikimedia.org/wikipedia/commons/2/2d/Los_Angeles_Chargers_logo.svg",
    "Los Angeles Rams": "https://upload.wikimedia.org/wikipedia/commons/5/5a/Los_Angeles_Rams_logo.svg",
    "Miami Dolphins": "https://upload.wikimedia.org/wikipedia/commons/2/2b/Miami_Dolphins_logo.svg",
    "Minnesota Vikings": "https://upload.wikimedia.org/wikipedia/commons/6/6b/Minnesota_Vikings_logo.svg",
    "New England Patriots": "https://upload.wikimedia.org/wikipedia/commons/2/2d/New_England_Patriots_logo.svg",
    "New Orleans Saints": "https://upload.wikimedia.org/wikipedia/commons/4/4f/New_Orleans_Saints_logo.svg",
    "New York Giants": "https://upload.wikimedia.org/wikipedia/commons/4/4c/New_York_Giants_logo.svg",
    "New York Jets": "https://upload.wikimedia.org/wikipedia/commons/2/25/New_York_Jets_logo.svg",
    "Philadelphia Eagles": "https://upload.wikimedia.org/wikipedia/commons/3/3e/Philadelphia_Eagles_logo.svg",
    "Pittsburgh Steelers": "https://upload.wikimedia.org/wikipedia/commons/0/0c/Pittsburgh_Steelers_logo.svg",
    "San Francisco 49ers": "https://upload.wikimedia.org/wikipedia/commons/9/99/San_Francisco_49ers_logo.svg",
    "Seattle Seahawks": "https://upload.wikimedia.org/wikipedia/commons/5/5f/Seattle_Seahawks_logo.svg",
    "Tampa Bay Buccaneers": "https://upload.wikimedia.org/wikipedia/commons/4/4a/Tampa_Bay_Buccaneers_logo.svg",
    "Tennessee Titans": "https://upload.wikimedia.org/wikipedia/commons/2/2d/Tennessee_Titans_logo.svg",
    "Washington Commanders": "https://upload.wikimedia.org/wikipedia/commons/0/0e/Washington_Commanders_logo.svg",
}

# ---------- ELO ----------
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
    required = {"season", "week", "team1", "team2", "score1", "score2", "home_team"}
    if not required.issubset(df.columns):
        raise ValueError("Historical games sheet missing required columns.")
    grouped = df.groupby(["season", "week"])
    for (_, _), games in grouped:
        for _, row in games.iterrows():
            update_ratings(elo_ratings, row.team1, row.team2, row.score1, row.score2, row.home_team)
    return dict(elo_ratings)

# ---------- HELPERS ----------
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
    if prob is None:
        return "N/A"
    prob = max(min(prob, 0.999), 0.001)
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

def spread_to_probability(spread):
    b = 0.23
    return 1 / (1 + np.exp(-b * (-spread)))

def format_edge_badge(edge):
    if edge is None:
        return ""
    if edge > 0.05:
        return f'<div style="display:inline-block;padding:4px 8px;border-radius:6px;background:#d4f9d4;color:#0b6623;font-weight:700">VALUE +{edge:.1%}</div>'
    if edge < -0.05:
        return f'<div style="display:inline-block;padding:4px 8px;border-radius:6px;background:#ffd6d6;color:#8b0000;font-weight:700">NEG -{edge:.1%}</div>'
    return f'<div style="display:inline-block;padding:4px 8px;border-radius:6px;background:#efefef;color:#333;font-weight:700">Close {edge:.1%}</div>'

def fuzzy_find_team_in_odds(team_name, odds_index_keys):
    name = str(team_name).lower()
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
    matches = get_close_matches(name, candidates, n=1, cutoff=0.6)
    if matches:
        best = matches[0]
        for k in odds_index_keys:
            if best in k:
                return k
    return None

# ---------- CARD CSS ----------
CARD_CSS = """
<style>
.matchup-card{border-radius:12px;padding:14px;margin-bottom:14px;box-shadow:0 6px 22px rgba(0,0,0,0.10);
background:linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,248,248,0.98));}
.team-block{display:flex;align-items:center;gap:12px;}
.team-name{font-weight:800;font-size:18px;}
.small-muted{color:#6b7280;font-size:12px;}
.prob-bar{height:10px;border-radius:6px;overflow:hidden;background:#eee;margin-top:6px;}
.prob-fill{height:10px;}
.ml-badge{font-weight:700;padding:6px 8px;border-radius:6px;background:#f3f4f6;}
.bookmaker{font-size:11px;color:#666;margin-left:8px;}
.pill{display:inline-block;padding:2px 8px;border-radius:999px;background:#eef2ff;color:#334155;font-weight:700;font-size:11px;margin-left:6px;}
</style>
"""

# ---------- SCRAPER CORE ----------
USER_AGENTS = [
    # a few rotating UAs
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
]

@contextmanager
def make_driver(headless=True):
    if not SELENIUM_OK:
        raise RuntimeError("Selenium/undetected-chromedriver not available. Install packages and restart.")
    opts = uc.ChromeOptions()
    ua = random.choice(USER_AGENTS)
    opts.add_argument(f"--user-agent={ua}")
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1280,1800")
    try:
        driver = uc.Chrome(options=opts, use_subprocess=True)
        yield driver
    finally:
        try:
            driver.quit()
        except Exception:
            pass

def _to_moneyline_string(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    try:
        xi = int(x)
        if xi > 0:
            return f"+{xi}"
        else:
            return f"{xi}"
    except Exception:
        return "N/A"

def _to_spread_string(point: Optional[float]) -> str:
    if point is None:
        return "N/A"
    try:
        # show sign explicitly for favorite (negative) / dog (positive)
        if point > 0:
            return f"+{point:.1f}".rstrip('0').rstrip('.')
        elif point < 0:
            return f"{point:.1f}".rstrip('0').rstrip('.')
        else:
            return "PK"
    except Exception:
        return "N/A"

def _normalize_pair(home: str, away: str, home_ml: Optional[int], away_ml: Optional[int],
                    home_sp: Optional[float], away_sp: Optional[float],
                    book_title: str) -> Dict[frozenset, Dict[str, Any]]:
    # force consistent signs: favorite = negative spread, underdog = positive
    if isinstance(home_sp, (int, float)) and isinstance(away_sp, (int, float)):
        # If both present but signs same, flip using who’s favored by ML as tiebreaker
        if np.sign(home_sp) == np.sign(away_sp):
            # derive from MLs
            if (isinstance(home_ml, int) and isinstance(away_ml, int)):
                if home_ml < away_ml:  # home favored
                    home_sp = -abs(home_sp)
                    away_sp = abs(away_sp)
                elif away_ml < home_ml:  # away favored
                    home_sp = abs(home_sp)
                    away_sp = -abs(away_sp)
        else:
            # ensure opposite signs and magnitude parity
            m = max(abs(home_sp), abs(away_sp))
            if home_sp < 0:
                home_sp = -m
                away_sp = m
            else:
                home_sp = m
                away_sp = -m
    key = frozenset([home.lower(), away.lower()])
    return {
        key: {
            "moneyline": {
                home.lower(): _to_moneyline_string(home_ml),
                away.lower(): _to_moneyline_string(away_ml),
            },
            "spread": {
                home.lower(): _to_spread_string(home_sp),
                away.lower(): _to_spread_string(away_sp),
            },
            "bookmaker": book_title
        }
    }

def _clean_team_name(txt: str) -> str:
    return re.sub(r"\s+", " ", txt).strip()

# --------- FanDuel scraper (experimental selectors) ----------
def scrape_fanduel_nfl(headless=True, timeout=18) -> Dict[frozenset, Dict[str, Any]]:
    """
    Attempts to scrape NFL Game Lines (Moneyline + Spread) from FanDuel main NFL page.
    This is brittle by nature—selectors can change. We handle failures gracefully.
    """
    url = "https://sportsbook.fanduel.com/football/nfl?tab=game-lines"
    odds: Dict[frozenset, Dict[str, Any]] = {}
    if not SELENIUM_OK:
        return odds
    try:
        with make_driver(headless=headless) as d:
            d.get(url)
            WebDriverWait(d, timeout).until(lambda drv: drv.execute_script("return document.readyState") == "complete")
            # Wait for event cards to render (heuristic)
            WebDriverWait(d, timeout).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "[data-testid*='event-row'],[class*='event'],[class*='EventRow']")))

            event_nodes = d.find_elements(By.CSS_SELECTOR, "[data-testid*='event-row'],[class*='event'],[class*='EventRow']")
            for node in event_nodes:
                try:
                    text = node.text
                    # Heuristic parse: find two team names (lines), their ML and Spread bits.
                    # We look for patterns like +120 / -140 and spreads like -2.5 / +2.5
                    # Extract team names from the first 2 non-empty lines that aren't prices.
                    lines = [ln for ln in text.splitlines() if ln.strip()]
                    # crude filter to avoid headers
                    teams = []
                    for ln in lines:
                        if re.search(r"[+-]?\d+(\.\d+)?", ln):  # likely price or spread line
                            continue
                        if len(ln.split()) >= 2 and len(teams) < 2:
                            teams.append(_clean_team_name(ln))
                        if len(teams) == 2:
                            break
                    if len(teams) != 2:
                        continue

                    # Find moneylines (first two +/- numbers that look like ML between -2000..+2000)
                    ml_candidates = [int(x) for x in re.findall(r"(?<!\d)(?:\+|-)\d{2,4}(?!\d)", text)]
                    ml_candidates = [x for x in ml_candidates if -2000 <= x <= 2000]
                    home_ml: Optional[int] = None
                    away_ml: Optional[int] = None
                    # try to pick two MLs
                    if len(ml_candidates) >= 2:
                        away_ml, home_ml = ml_candidates[:2]  # display order often away, then home

                    # Find spreads: e.g., -2.5, +3.5 (limit reasonable range)
                    sp_candidates = [float(x) for x in re.findall(r"(?<!\d)(?:\+|-)\d+(?:\.\d)?(?!\d)", text)]
                    sp_candidates = [x for x in sp_candidates if -40.0 <= x <= 40.0]
                    home_sp: Optional[float] = None
                    away_sp: Optional[float] = None
                    if len(sp_candidates) >= 2:
                        # Often shown as away, then home; we normalize signs later
                        away_sp, home_sp = sp_candidates[:2]

                    # Determine which is home / away:
                    # FanDuel often lists in Away vs Home order in many widgets; we’ll assume teams[0]=Away, teams[1]=Home
                    away = teams[0]
                    home = teams[1]

                    normalized = _normalize_pair(
                        home=home, away=away,
                        home_ml=home_ml, away_ml=away_ml,
                        home_sp=home_sp, away_sp=away_sp,
                        book_title="FanDuel"
                    )
                    odds.update(normalized)
                except Exception:
                    continue
    except Exception:
        # swallow and return what we got
        pass
    return odds

# --------- DraftKings scraper (experimental selectors) ----------
def scrape_dk_nfl(headless=True, timeout=18) -> Dict[frozenset, Dict[str, Any]]:
    """
    Attempts to scrape NFL Game Lines (Moneyline + Spread) from DraftKings league page.
    """
    url = "https://sportsbook.draftkings.com/leagues/football/nfl"
    odds: Dict[frozenset, Dict[str, Any]] = {}
    if not SELENIUM_OK:
        return odds
    try:
        with make_driver(headless=headless) as d:
            d.get(url)
            WebDriverWait(d, timeout).until(lambda drv: drv.execute_script("return document.readyState") == "complete")
            # Wait for game containers (heuristic)
            WebDriverWait(d, timeout).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "[data-event], [class*='event'], [class*='Event']")))
            event_nodes = d.find_elements(By.CSS_SELECTOR, "[data-event], [class*='event'], [class*='Event']")

            for node in event_nodes:
                try:
                    text = node.text
                    lines = [ln for ln in text.splitlines() if ln.strip()]
                    # Extract two team names (non-price lines)
                    teams = []
                    for ln in lines:
                        if re.search(r"[+-]?\d+(\.\d+)?", ln):
                            continue
                        if len(ln.split()) >= 2 and len(teams) < 2:
                            teams.append(_clean_team_name(ln))
                        if len(teams) == 2:
                            break
                    if len(teams) != 2:
                        continue

                    # Moneylines
                    ml_candidates = [int(x) for x in re.findall(r"(?<!\d)(?:\+|-)\d{2,4}(?!\d)", text)]
                    ml_candidates = [x for x in ml_candidates if -2000 <= x <= 2000]
                    home_ml = away_ml = None
                    if len(ml_candidates) >= 2:
                        away_ml, home_ml = ml_candidates[:2]

                    # Spreads
                    sp_candidates = [float(x) for x in re.findall(r"(?<!\d)(?:\+|-)\d+(?:\.\d)?(?!\d)", text)]
                    sp_candidates = [x for x in sp_candidates if -40.0 <= x <= 40.0]
                    home_sp = away_sp = None
                    if len(sp_candidates) >= 2:
                        away_sp, home_sp = sp_candidates[:2]

                    # Assume teams[0]=Away, teams[1]=Home (common on DK list displays)
                    away = teams[0]
                    home = teams[1]

                    normalized = _normalize_pair(
                        home=home, away=away,
                        home_ml=home_ml, away_ml=away_ml,
                        home_sp=home_sp, away_sp=away_sp,
                        book_title="DraftKings"
                    )
                    odds.update(normalized)
                except Exception:
                    continue
    except Exception:
        pass
    return odds

# --------- Mock odds (for offline/dev fallback) ----------
def mock_odds_from_schedule(sched_df: pd.DataFrame) -> Dict[frozenset, Dict[str, Any]]:
    odds = {}
    for _, r in sched_df.iterrows():
        t1 = str(r["team1"])
        t2 = str(r["team2"])
        home = str(r.get("home_team", t1))
        away = t2 if home == t1 else t1

        # fabricate a plausible line
        fav = random.choice([home, away])
        dog = away if fav == home else home
        fav_ml = -random.choice([110, 120, 130, 140, 150, 160])
        dog_ml = +random.choice([100, 110, 120, 130, 140, 150])
        fav_sp = -random.choice([1.5, 2.5, 3.5, 4.5, 6.5])
        dog_sp = -fav_sp

        odds[frozenset([home.lower(), away.lower()])] = {
            "moneyline": {
                home.lower(): _to_moneyline_string(fav_ml if fav == home else dog_ml),
                away.lower(): _to_moneyline_string(dog_ml if fav == home else fav_ml),
            },
            "spread": {
                home.lower(): _to_spread_string(fav_sp if fav == home else -fav_sp),
                away.lower(): _to_spread_string(dog_sp if fav == home else -dog_sp),
            },
            "bookmaker": "MockBook"
        }
    return odds

# --------- Combine / choose source ----------
@st.cache_data(ttl=120, show_spinner=False)
def fetch_odds(source: str, headless: bool, sched_df: pd.DataFrame) -> Dict[frozenset, Dict[str, Any]]:
    """
    source: "FanDuel", "DraftKings", "Both (prefer best ML)", or "Mock"
    """
    out: Dict[frozenset, Dict[str, Any]] = {}
    if source == "Mock" or not SELENIUM_OK:
        return mock_odds_from_schedule(sched_df)

    fd = dk = {}
    if source in ("FanDuel", "Both (prefer best ML)"):
        fd = scrape_fanduel_nfl(headless=headless)
    if source in ("DraftKings", "Both (prefer best ML)"):
        dk = scrape_dk_nfl(headless=headless)

    if source == "FanDuel":
        return fd
    if source == "DraftKings":
        return dk

    # Merge: prefer best ML (most favorable price for each team)
    keys = set(fd.keys()) | set(dk.keys())
    for k in keys:
        fd_row = fd.get(k)
        dk_row = dk.get(k)
        if fd_row and not dk_row:
            out[k] = fd_row
        elif dk_row and not fd_row:
            out[k] = dk_row
        elif fd_row and dk_row:
            # combine: choose better ML per team; spread pick from the book with better absolute price delta
            teams = list(k)
            t0, t1 = teams[0], teams[1]
            ml = {}
            sp = {}
            # choose ML with better value for bettor (higher + for dog, lower abs - for fav)
            for t in (t0, t1):
                fd_ml = fd_row["moneyline"].get(t, "N/A")
                dk_ml = dk_row["moneyline"].get(t, "N/A")
                def _ml_to_int(s):
                    try:
                        return int(str(s))
                    except:
                        return None
                a = _ml_to_int(fd_ml)
                b = _ml_to_int(dk_ml)
                best = fd_ml
                if a is None and b is not None:
                    best = dk_ml
                elif b is None and a is not None:
                    best = fd_ml
                elif a is not None and b is not None:
                    # choose the one that yields lower implied hold for bettor
                    # for positive numbers: prefer larger; for negative numbers: prefer closer to zero
                    if (a >= 0 and b >= 0 and b > a) or (a < 0 and b < 0 and abs(b) < abs(a)) or (a < 0 <= b):
                        best = dk_ml
                    else:
                        best = fd_ml
                ml[t] = best

            # pick spreads from FanDuel by default; if missing, use DK
            for t in (t0, t1):
                sp[t] = fd_row["spread"].get(t, dk_row["spread"].get(t, "N/A"))

            out[k] = {
                "moneyline": ml,
                "spread": sp,
                "bookmaker": "FD/DK (best ML)"
            }
    return out

# ---------- UI RENDER ----------
def render_matchup_card(team_home, team_away, logos, odds_book,
                        prob_home, prob_away,
                        pred_spread_home, pred_spread_away,
                        live_ml_home, live_ml_away,
                        live_spread_home, live_spread_away):
    implied_home = moneyline_to_probability(live_ml_home)
    implied_away = moneyline_to_probability(live_ml_away)
    edge_home = None if implied_home is None else (prob_home - implied_home)
    edge_away = None if implied_away is None else (prob_away - implied_away)

    st.markdown(CARD_CSS, unsafe_allow_html=True)
    st.markdown("<div class='matchup-card'>", unsafe_allow_html=True)

    cols = st.columns([1,1])
    # Left side = Away
    with cols[0]:
        logo_url = logos.get(team_away.lower(), "")
        st.markdown(
            f"<div class='team-block'><img src='{logo_url}' width='56' style='border-radius:6px'/>"
            f"<div><div class='team-name'>{team_away}</div>"
            f"<div class='small-muted'>ML: <span class='ml-badge'>{live_ml_away}</span> | Spread: <strong>{live_spread_away}</strong>"
            f"<span class='pill'>Pred: {pred_spread_away}</span></div>"
            f"</div></div>", unsafe_allow_html=True
        )
        pct = prob_away if prob_away is not None else 0.5
        fill_color = "#16a34a" if edge_away and edge_away > 0.05 else ("#ef4444" if edge_away and edge_away < -0.05 else "#3b82f6")
        st.markdown(f"<div class='prob-bar'><div class='prob-fill' style='width:{pct*100:.1f}%; background:{fill_color}'></div></div>", unsafe_allow_html=True)
        if prob_away is not None:
            st.markdown(f"<div class='small-muted'>{(prob_away*100):.1f}% win probability</div>", unsafe_allow_html=True)

    # Right side = Home
    with cols[1]:
        logo_url2 = logos.get(team_home.lower(), "")
        st.markdown(
            f"<div class='team-block' style='justify-content:flex-end'>"
            f"<div><div class='team-name' style='text-align:right'>{team_home}</div>"
            f"<div class='small-muted' style='text-align:right'>ML: <span class='ml-badge'>{live_ml_home}</span> | Spread: <strong>{live_spread_home}</strong>"
            f"<span class='pill'>Pred: {pred_spread_home}</span></div>"
            f"</div> <img src='{logo_url2}' width='56' style='border-radius:6px'/></div>", unsafe_allow_html=True
        )
        pct2 = prob_home if prob_home is not None else 0.5
        fill_color2 = "#16a34a" if edge_home and edge_home > 0.05 else ("#ef4444" if edge_home and edge_home < -0.05 else "#3b82f6")
        st.markdown(f"<div class='prob-bar'><div class='prob-fill' style='width:{pct2*100:.1f}%; background:{fill_color2}'></div></div>", unsafe_allow_html=True)
        if prob_home is not None:
            st.markdown(f"<div class='small-muted' style='text-align:right'>{(prob_home*100):.1f}% win probability</div>", unsafe_allow_html=True)

    edge_html = ""
    if edge_home is not None:
        edge_html += f"Home edge: {format_edge_badge(edge_home)}&nbsp;&nbsp;"
    if edge_away is not None:
        edge_html += f"Away edge: {format_edge_badge(edge_away)}"
    if edge_html:
        st.markdown(f"<div style='margin-top:8px'>{edge_html} <span class='bookmaker'>Bookmaker: {odds_book}</span></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("")

# ---------- STREAMLIT APP ----------
st.set_page_config(page_title="NFL Elo — FanDuel & DraftKings Scraper", layout="wide")
st.title("🏈 NFL Elo Betting Dashboard — FanDuel & DraftKings")
st.caption("Elo predictions + live odds scraped from FanDuel/DraftKings (with mock fallback).")

# Load Excel
try:
    hist_df = pd.read_excel(EXCEL_FILE, sheet_name=HIST_SHEET)
    sched_df = pd.read_excel(EXCEL_FILE, sheet_name=SCHEDULE_SHEET)
except FileNotFoundError:
    st.error(f"Excel file '{EXCEL_FILE}' not found. Put it in the app folder or update EXCEL_FILE path.")
    st.stop()
except Exception as e:
    st.error(f"Error reading Excel file: {e}")
    st.stop()

# Validate schedule columns
if 'week' not in sched_df.columns:
    st.error("Schedule sheet must contain a 'week' column.")
    st.stop()

# Build Elo ratings
try:
    ratings = run_elo_pipeline(hist_df)
except Exception as e:
    st.error(f"Error computing Elo: {e}")
    st.stop()

# Sidebar
st.sidebar.header("Odds Source")
source = st.sidebar.selectbox("Choose source", ["Both (prefer best ML)", "FanDuel", "DraftKings", "Mock"], index=0)
headless = st.sidebar.checkbox("Headless browser", value=True)
st.sidebar.caption("If scraping fails due to site changes or bot checks, switch to Mock for testing.")

# Week dropdown
available_weeks = sorted(sched_df['week'].dropna().unique().astype(int).tolist())
if not available_weeks:
    st.error("No weeks found in schedule sheet.")
    st.stop()

default_index = len(available_weeks) - 1 if len(available_weeks) > 0 else 0
selected_week = st.selectbox("Select Week to view", available_weeks, index=default_index)

st.markdown("---")

# Filter schedule for selected week
week_games = sched_df[sched_df['week'] == selected_week].copy()
if week_games.empty:
    st.info(f"No games found for week {selected_week}.")
    st.stop()

# Fetch odds via scrapers (or mock)
with st.spinner(f"Fetching odds from {source}..."):
    try:
        odds_index = fetch_odds(source, headless, week_games)
    except Exception as e:
        st.warning(f"Fetching odds failed: {e}. Using Mock odds for display.")
        odds_index = mock_odds_from_schedule(week_games)

if not odds_index:
    st.warning("No odds found from the selected source. Try switching source, disabling headless, or using Mock.")
    odds_index = mock_odds_from_schedule(week_games)

# Render matchup cards
for _, row in week_games.iterrows():
    team1 = str(row['team1'])
    team2 = str(row['team2'])
    home_team = str(row.get('home_team', team1))

    # Elo probs (from home/away perspective)
    r1 = ratings.get(team1, BASE_ELO) + (HOME_ADVANTAGE if home_team == team1 else 0)
    r2 = ratings.get(team2, BASE_ELO) + (HOME_ADVANTAGE if home_team == team2 else 0)
    prob_team1 = expected_score(r1, r2)
    prob_team2 = 1 - prob_team1

    # Predicted spreads for each team (negative favorite, positive dog)
    # predicted_spread here is from team1 perspective; then mirror to team2
    team1_is_fav = prob_team1 > prob_team2
    sp_team1 = probability_to_spread(prob_team1, team_is_favorite=team1_is_fav)
    sp_team2 = -sp_team1

    # Live odds defaults
    live_ml_home = live_ml_away = "N/A"
    live_spread_home = live_spread_away = "N/A"
    bookmaker_title = "N/A"

    match_key = fuzzy_find_team_in_odds(team1, odds_index.keys()) or fuzzy_find_team_in_odds(team2, odds_index.keys())
    if match_key:
        entry = odds_index.get(match_key, {})
        bookmaker_title = entry.get("bookmaker", "N/A")
        ml = entry.get("moneyline", {}) or {}
        sp = entry.get("spread", {}) or {}

        live_ml_team1 = ml.get(team1.lower(), next(iter(ml.values()), "N/A"))
        live_ml_team2 = ml.get(team2.lower(), next(iter(ml.values()), "N/A"))
        live_spread_team1 = sp.get(team1.lower(), next(iter(sp.values()), "N/A"))
        live_spread_team2 = sp.get(team2.lower(), next(iter(sp.values()), "N/A"))

        # assign to home/away slots for display (home is on the RIGHT)
        if home_team == team1:
            live_ml_home, live_ml_away = live_ml_team1, live_ml_team2
            live_spread_home, live_spread_away = live_spread_team1, live_spread_team2
            prob_home, prob_away = prob_team1, prob_team2
            pred_spread_home, pred_spread_away = sp_team1, sp_team2
            home_name, away_name = team1, team2
        else:
            live_ml_home, live_ml_away = live_ml_team2, live_ml_team1
            live_spread_home, live_spread_away = live_spread_team2, live_spread_team1
            prob_home, prob_away = prob_team2, prob_team1
            pred_spread_home, pred_spread_away = sp_team2, sp_team1
            home_name, away_name = team2, team1
    else:
        # If we couldn't map, still show Elo and labels
        if home_team == team1:
            prob_home, prob_away = prob_team1, prob_team2
            pred_spread_home, pred_spread_away = sp_team1, sp_team2
            home_name, away_name = team1, team2
        else:
            prob_home, prob_away = prob_team2, prob_team1
            pred_spread_home, pred_spread_away = sp_team2, sp_team1
            home_name, away_name = team2, team1

    # Format predicted spreads to show explicit sign per team
    def fmt_pred(v):
        return f"{v:+.1f}"

    # Render card
    render_matchup_card(
        team_home=home_name,
        team_away=away_name,
        logos=TEAM_LOGOS,
        odds_book=bookmaker_title,
        prob_home=prob_home,
        prob_away=prob_away,
        pred_spread_home=fmt_pred(pred_spread_home),
        pred_spread_away=fmt_pred(pred_spread_away),
        live_ml_home=live_ml_home,
        live_ml_away=live_ml_away,
        live_spread_home=live_spread_home,
        live_spread_away=live_spread_away,
    )

st.markdown("---")
st.caption("Note: sportsbook HTML changes frequently; if scraping fails, use Mock or try toggling headless.")
