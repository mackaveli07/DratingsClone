# injuries.py
import requests
import streamlit as st

from config import ESPN_TEAM_IDS

INJURY_CACHE_TTL_SECONDS = 600


@st.cache_data(ttl=INJURY_CACHE_TTL_SECONDS)
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
    for entry in data.get("entries", []):
        players.append({
            "name": entry.get("athlete", {}).get("displayName"),
            "position": entry.get("position", {}).get("abbreviation"),
            "status": entry.get("status", {}).get("type", ""),
        })
    return players


def injury_adjustment(players):
    penalty = 0
    for player in players or []:
        status = (player.get("status") or "").lower()
        position = (player.get("position") or "").upper()
        if position == "QB" and status in ["out", "doubtful"]:
            penalty -= 50
        elif position in ["RB", "WR", "TE"] and status in ["out", "doubtful"]:
            penalty -= 15
        elif status in ["out", "doubtful"]:
            penalty -= 10
    return penalty


__all__ = ["INJURY_CACHE_TTL_SECONDS", "fetch_injuries_espn", "injury_adjustment"]
