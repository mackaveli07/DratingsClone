# team_utils.py
from config import NFL_FULL_NAMES, TEAM_NAME_FIXES


def map_team_name(name):
    if not name:
        return "Unknown"
    value = str(name).strip()
    if value in TEAM_NAME_FIXES:
        value = TEAM_NAME_FIXES[value]
    if value.upper() in NFL_FULL_NAMES:
        return NFL_FULL_NAMES[value.upper()]
    for full_name in NFL_FULL_NAMES.values():
        if value.lower() == full_name.lower():
            return full_name
    return value


def get_abbr(team_full):
    for abbr, full_name in NFL_FULL_NAMES.items():
        if full_name == team_full:
            return abbr
    return None


__all__ = ["map_team_name", "get_abbr"]
