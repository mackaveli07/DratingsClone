# team_utils.py
from config import NFL_FULL_NAMES, TEAM_NAME_FIXES


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


__all__ = ["map_team_name", "get_abbr"]
