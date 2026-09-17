# config.py
from pathlib import Path

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
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    # ...
}

TEAM_COLORS = {
    "ARI": "#97233F",
    "ATL": "#A71930",
    # ...
}

TEAM_NAME_FIXES = {
    "Clevland Browns": "Cleveland Browns",
    "NY Jets": "New York Jets",
    # ...
}
