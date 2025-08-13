# app.py
import streamlit as st
import pandas as pd
import numpy as np
import time
import re
from collections import defaultdict
from difflib import get_close_matches

# ---------- Optional faster fuzzy matching ----------
try:
    import Levenshtein  # type: ignore
    has_lev = True
except Exception:
    has_lev = False

# ---------- Selenium / scraping deps ----------
import undetected_chromedriver as uc
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException
from bs4 import BeautifulSoup

### ---------- CONFIG ----------
BASE_ELO = 1500
K = 20
HOME_ADVANTAGE = 65

EXCEL_FILE = "games.xlsx"
HIST_SHEET = "games"
SCHEDULE_SHEET = "2025 schedule"

# Minimal team logo map — extend for all teams you care about (use full lowercase keys)
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
    # Ensure expected columns exist
    if not {"season", "week", "team1", "team2", "score1", "score2", "home_team"}.issubset(df.columns):
        raise ValueError("Historical games sheet missing required columns.")
    grouped = df.groupby(["season", "week"])
    for (season, week), games in grouped:
        for _, row in games.iterrows():
            update_ratings(elo_ratings, row.team1, row.team2, row.score1, row.score2, row.home_team)
    return dict(elo_ratings)

### ---------- HELPER: Scraper utilities ----------
def init_driver():
    """Headless Chrome for Streamlit Cloud (uses packages.txt installs)."""
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.binary_location = "/usr/bin/chromium"
    try:
        driver = uc.Chrome(options=options, driver_executable_path="/usr/bin/chromedriver")
    except TypeError:
        # older uc versions use 'executable_path'
        driver = uc.Chrome(options=options, executable_path="/usr/bin/chromedriver")
    return driver

def _clean_ml(text):
    """Normalize moneyline like '+120', '-145' -> string; remove extraneous chars."""
    if text is None:
        return "N/A"
    t = text.strip()
    # Pick first +/- number pattern
    m = re.search(r'([+-]\s*\d{2,4})', t.replace(" ", ""))
    if m:
        return m.group(1).replace(" ", "")
    # DraftKings sometimes shows as e.g. "EVEN"
    if re.search(r'EVEN|EV|Even', t):
        return "+100"
    # fallback: digits only, assume positive
    m2 = re.search(r'(\d{2,4})', t)
    if m2:
        return f"+{m2.group(1)}"
    return "N/A"

def _clean_spread(text):
    """Normalize spread like '-3.5', '+2', 'PK' -> signed string with decimal if needed."""
    if text is None:
        return "N/A"
    t = text.strip().upper()
    if "PK" in t or "PICK" in t:
        return "0"
    m = re.search(r'([+-]?\d+(\.\d+)?)', t.replace(" ", ""))
    if m:
        # keep sign if present (favorite negative, dog positive)
        return m.group(1)
    return "N/A"

def _pairwise(lst, size=2):
    return [lst[i:i+size] for i in range(0, len(lst), size)]

### ---------- SCRAPERS (ML + SPREAD) ----------
def scrape_fanduel_nfl():
    """
    Returns odds_index like:
    {
      frozenset(["team a", "team b"]): {
        "moneyline": { "team a": "-120", "team b": "+105" },
        "spread":    { "team a": "-2.5", "team b": "+2.5" },
        "bookmaker": "FanDuel"
      }
    }
    """
    url = "https://sportsbook.fanduel.com/sports/navigation/football/nfl"
    odds_index = {}
    driver = None
    try:
        driver = init_driver()
        driver.get(url)
        time.sleep(8)  # allow JS to render
        soup = BeautifulSoup(driver.page_source, "lxml")

        # FanDuel often renders cards with data-test-id attributes
        # We find event blocks then extract participant names + ML + Spread in order.
        events = soup.select('[data-test-id="event"]')
        for ev in events:
            names = [n.get_text(strip=True).lower() for n in ev.select('[data-test-id="participant"]')]
            # Moneyline prices:
            ml_nodes = [p.get_text(strip=True) for p in ev.select('[data-test-id="price"]')]
            # Spread lines:
            sp_nodes = [s.get_text(strip=True) for s in ev.select('[data-test-id="spread"]')]

            if len(names) != 2:
                continue

            # ML: try to pair first two prices (home/away order not guaranteed)
            mls = ["N/A", "N/A"]
            if len(ml_nodes) >= 2:
                mls = [_clean_ml(ml_nodes[0]), _clean_ml(ml_nodes[1])]

            # Spread: grab first two spreads for the two teams
            sps = ["N/A", "N/A"]
            if len(sp_nodes) >= 2:
                sps = [_clean_spread(sp_nodes[0]), _clean_spread(sp_nodes[1])]

            odds_index[frozenset(names)] = {
                "moneyline": {names[0]: mls[0], names[1]: mls[1]},
                "spread":    {names[0]: sps[0], names[1]: sps[1]},
                "bookmaker": "FanDuel"
            }
    except WebDriverException as e:
        st.error(f"FanDuel scraper error: {e}")
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass
    return odds_index

def scrape_dk_nfl():
    url = "https://sportsbook.draftkings.com/leagues/football/nfl"
    odds_index = {}
    driver = None
    try:
        driver = init_driver()
        driver.get(url)
        time.sleep(8)
        soup = BeautifulSoup(driver.page_source, "lxml")

        # DK: rows per game; names in .event-cell__name; ML in .sportsbook-odds;
        # spread numbers in .sportsbook-outcome-cell__line
        game_cards = soup.select('.event-card, .sportsbook-event-accordion__wrapper, .event-cell')
        for card in game_cards:
            names = [n.get_text(strip=True).lower() for n in card.select('.event-cell__name, .sportsbook-event-accordion__title, .event-cell__name-text')]
            names = [n for n in names if n]  # clean empties
            if len(names) < 2:
                continue
            names = names[:2]

            ml_nodes = [m.get_text(strip=True) for m in card.select('.sportsbook-odds')]
            sp_nodes = [s.get_text(strip=True) for s in card.select('.sportsbook-outcome-cell__line')]

            mls = ["N/A", "N/A"]
            if len(ml_nodes) >= 2:
                mls = [_clean_ml(ml_nodes[0]), _clean_ml(ml_nodes[1])]

            sps = ["N/A", "N/A"]
            if len(sp_nodes) >= 2:
                sps = [_clean_spread(sp_nodes[0]), _clean_spread(sp_nodes[1])]

            odds_index[frozenset(names)] = {
                "moneyline": {names[0]: mls[0], names[1]: mls[1]},
                "spread":    {names[0]: sps[0], names[1]: sps[1]},
                "bookmaker": "DraftKings"
            }
    except WebDriverException as e:
        st.error(f"DraftKings scraper error: {e}")
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass
    return odds_index

def scrape_betonline_nfl():
    url = "https://www.betonline.ag/sportsbook/football/nfl"
    odds_index = {}
    driver = None
    try:
        driver = init_driver()
        driver.get(url)
        time.sleep(8)
        soup = BeautifulSoup(driver.page_source, "lxml")

        # BetOnline desktop markup: rows often under market tables; team names and odds near each other.
        # We'll look for event containers that have two team names, then capture ML & spread nearby.
        # Team names often in ".opp" or ".teamName" or within table rows.
        # We'll parse table-ish structures first.
        # Fallback to a generic approach if selectors change.
        # 1) Try common table rows:
        rows = soup.select("div.eventHolder, div.event, tr, .couponRow")
        for blk in rows:
            # find two team names in this block
            team_nodes = blk.select(".teamName, .opp, .seln-name, .participant, .team-name")
            teams = [t.get_text(strip=True).lower() for t in team_nodes if t.get_text(strip=True)]
            if len(teams) < 2:
                # alt: some blocks have teams split in separate lines:
                ttxt = " ".join([t.get_text(" ", strip=True).lower() for t in team_nodes])
                # attempt to find "team a at team b" patterns – skip; continue to next block
                continue
            teams = teams[:2]

            # moneyline odds in elements like ".moneyLine", "td.odds", or text containing +/-
            ml_nodes = blk.select(".moneyLine, .odds, .price, .priceAmerican, .seln-odds")
            ml_vals = [n.get_text(strip=True) for n in ml_nodes if n.get_text(strip=True)]
            ml_vals = [_clean_ml(x) for x in ml_vals]
            # try first two odds
            mls = ["N/A", "N/A"]
            if len(ml_vals) >= 2:
                mls = [ml_vals[0], ml_vals[1]]

            # spread odds in ".pointSpread", or text with +/- number near "Spread" labels
            sp_nodes = blk.select(".pointSpread, .line, .hcap, .handicap, .seln-spread, .spread")
            sp_vals = [n.get_text(strip=True) for n in sp_nodes if n.get_text(strip=True)]
            sp_vals = [_clean_spread(x) for x in sp_vals]
            sps = ["N/A", "N/A"]
            if len(sp_vals) >= 2:
                sps = [sp_vals[0], sp_vals[1]]

            odds_index[frozenset(teams)] = {
                "moneyline": {teams[0]: mls[0], teams[1]: mls[1]},
                "spread":    {teams[0]: sps[0], teams[1]: sps[1]},
                "bookmaker": "BetOnline"
            }
    except WebDriverException as e:
        st.error(f"BetOnline scraper error: {e}")
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass
    return odds_index

### ---------- ODDS HELPERS ----------
def moneyline_to_probability(ml):
    try:
        if ml in [None, "N/A", ""]:
            return None
        s = str(ml).strip()
        if s.startswith('+'):
            val = int(s.replace('+',''))
            return 100.0 / (val + 100.0)
        if s.startswith('-'):
            val = int(s.replace('-',''))
            return val / (val + 100.0)
        # raw integer string
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
    if prob >= 0.5:
        return f"-{round(100 * prob / (1 - prob))}"
    else:
        return f"+{round(100 * (1 - prob) / prob)}"

def probability_to_spread(prob, team_is_favorite=True):
    # logistic bridge between prob and point spread; tuned b
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
    name = team_name.lower()
    # try direct membership
    for key in odds_index_keys:
        for tk in key:
            if name == tk:
                return key
    # try substring
    for key in odds_index_keys:
        combined = " ".join(list(key))
        if name in combined:
            return key
    # use Levenshtein or difflib
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
.matchup-card{border-radius:14px;padding:14px;margin-bottom:14px;box-shadow:0 8px 28px rgba(0,0,0,0.12);background:linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,248,248,0.98));}
.team-block{display:flex;align-items:center;gap:12px;}
.team-name{font-weight:800;font-size:18px;}
.small-muted{color:#6b7280;font-size:12px;}
.prob-bar{height:10px;border-radius:6px;overflow:hidden;background:#eee;margin-top:6px;}
.prob-fill{height:10px;}
.ml-badge{font-weight:800;padding:6px 8px;border-radius:8px;background:#f3f4f6;}
.bookmaker{font-size:11px;color:#666;margin-left:8px;}
</style>
"""

def render_matchup_card(team_home, team_away, logos, odds_book, prob_home, prob_away,
                        pred_spread_home, pred_spread_away, live_ml_home, live_ml_away,
                        live_spread_home, live_spread_away):
    implied_home = moneyline_to_probability(live_ml_home)
    implied_away = moneyline_to_probability(live_ml_away)
    edge_home = None if implied_home is None else (prob_home - implied_home)
    edge_away = None if implied_away is None else (prob_away - implied_away)

    st.markdown(CARD_CSS, unsafe_allow_html=True)
    st.markdown("<div class='matchup-card'>", unsafe_allow_html=True)

    cols = st.columns([1,1])
    # Left / Away (kept as your earlier layout)
    with cols[0]:
        logo_url = logos.get(team_away.lower(), "")
        st.markdown(
            f"<div class='team-block'><img src='{logo_url}' width='56' style='border-radius:6px'/>"
            f" <div><div class='team-name'>{team_away}</div>"
            f"<div class='small-muted'>ML: <span class='ml-badge'>{live_ml_away}</span> | "
            f"Spread: <strong>{live_spread_away}</strong> | Pred: <strong>{pred_spread_away:+}</strong></div></div></div>",
            unsafe_allow_html=True
        )
        pct = prob_away if prob_away is not None else 0.5
        fill = "#16a34a" if edge_away and edge_away > 0.05 else ("#ef4444" if edge_away and edge_away < -0.05 else "#3b82f6")
        st.markdown(f"<div class='prob-bar'><div class='prob-fill' style='width:{pct*100:.1f}%; background:{fill}'></div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>{(prob_away*100):.1f}% win probability</div>" if prob_away is not None else "", unsafe_allow_html=True)

    # Right / Home
    with cols[1]:
        logo_url2 = logos.get(team_home.lower(), "")
        st.markdown(
            f"<div class='team-block' style='justify-content:flex-end'>"
            f"<div><div class='team-name' style='text-align:right'>{team_home}</div>"
            f"<div class='small-muted' style='text-align:right'>ML: <span class='ml-badge'>{live_ml_home}</span> | "
            f"Spread: <strong>{live_spread_home}</strong> | Pred: <strong>{pred_spread_home:+}</strong></div></div> "
            f"<img src='{logo_url2}' width='56' style='border-radius:6px'/></div>",
            unsafe_allow_html=True
        )
        pct2 = prob_home if prob_home is not None else 0.5
        fill2 = "#16a34a" if edge_home and edge_home > 0.05 else ("#ef4444" if edge_home and edge_home < -0.05 else "#3b82f6")
        st.markdown(f"<div class='prob-bar'><div class='prob-fill' style='width:{pct2*100:.1f}%; background:{fill2}'></div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted' style='text-align:right'>{(prob_home*100):.1f}% win probability</div>" if prob_home is not None else "", unsafe_allow_html=True)

    st.markdown(f"<div style='margin-top:8px'><strong>Bookmaker:</strong> <span class='bookmaker'>{odds_book}</span></div>", unsafe_allow_html=True)

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
st.set_page_config(page_title="NFL Elo — Matchup Cards (Scraper Edition)", layout="wide")
st.title("🏈 NFL Elo Betting Dashboard (Scraper Edition)")
st.caption("Elo predictions + live odds via headless scrapers — value hints highlighted")

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

# Sidebar controls
st.sidebar.header("Controls")
odds_source = st.sidebar.selectbox("Odds Source", ["FanDuel", "DraftKings", "BetOnline"])

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
else:
    # Fetch odds via selected scraper
    odds_index = {}
    try:
        if odds_source == "FanDuel":
            odds_index = scrape_fanduel_nfl()
        elif odds_source == "DraftKings":
            odds_index = scrape_dk_nfl()
        else:
            odds_index = scrape_betonline_nfl()
    except Exception as e:
        st.error(f"Could not fetch odds from {odds_source}: {e}")
        odds_index = {}

    # Render matchup cards for the selected week
    for _, row in week_games.iterrows():
        team1 = row['team1']
        team2 = row['team2']
        home_team = row.get('home_team', team1)

        # Predicted probabilities using Elo + home adv
        r1 = ratings.get(team1, BASE_ELO) + (HOME_ADVANTAGE if home_team == team1 else 0)
        r2 = ratings.get(team2, BASE_ELO) + (HOME_ADVANTAGE if home_team == team2 else 0)
        # Probability for each team
        prob_team1 = expected_score(r1, r2)
        prob_team2 = 1 - prob_team1

        # Predicted spread from each team's perspective (favorite negative, dog positive)
        # We'll define it relative to each team (not strictly "home" here)
        pred_spread_team1 = probability_to_spread(prob_team1, team_is_favorite=(prob_team1 > prob_team2))
        pred_spread_team2 = -pred_spread_team1

        # Default live values
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

                # Try exact lowercased key, else fallback to "first available"
                live_ml_team1 = ml.get(team1.lower(), next(iter(ml.values()), "N/A"))
                live_ml_team2 = ml.get(team2.lower(), next(iter(ml.values()), "N/A"))
                live_spread_team1 = sp.get(team1.lower(), next(iter(sp.values()), "N/A"))
                live_spread_team2 = sp.get(team2.lower(), next(iter(sp.values()), "N/A"))

        # Render the card (home team shown on right column in this layout)
        # We pass predicted spreads from each team's perspective,
        # but the card displays Left = Away, Right = Home (as in your visual).
        if home_team.lower() == team1.lower():
            # team1 is home, team2 away -> right = team1
            render_matchup_card(
                team_home=team1,
                team_away=team2,
                logos=TEAM_LOGOS,
                odds_book=bookmaker_title,
                prob_home=prob_team1,
                prob_away=prob_team2,
                pred_spread_home=pred_spread_team1,
                pred_spread_away=pred_spread_team2,
                live_ml_home=live_ml_team1,
                live_ml_away=live_ml_team2,
                live_spread_home=live_spread_team1,
                live_spread_away=live_spread_team2,
            )
        else:
            # team2 is home -> right = team2
            render_matchup_card(
                team_home=team2,
                team_away=team1,
                logos=TEAM_LOGOS,
                odds_book=bookmaker_title,
                prob_home=prob_team2,
                prob_away=prob_team1,
                pred_spread_home=pred_spread_team2,
                pred_spread_away=pred_spread_team1,
                live_ml_home=live_ml_team2,
                live_ml_away=live_ml_team1,
                live_spread_home=live_spread_team2,
                live_spread_away=live_spread_team1,
            )

st.markdown("---")
st.caption("Tip: add more team logos to TEAM_LOGOS for a nicer UI. Scraper CSS can change; tweak selectors if a book updates their page.")
