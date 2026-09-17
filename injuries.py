# weather.py
import datetime
import os

import pytz
import requests

from config import STADIUMS

OWM_API_KEY = os.getenv("OWM_API_KEY", "")


@staticmethod
def _get_weather_cached(team: str, kickoff_unix: int, api_key: str):
    """Fetch cached weather near kickoff for the given home team."""
    if team not in STADIUMS or not api_key:
        return None
    try:
        kickoff_unix = int(kickoff_unix)
    except (TypeError, ValueError):
        return None
    lat, lon = STADIUMS[team]["lat"], STADIUMS[team]["lon"]
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=imperial"
    try:
        resp = requests.get(url, timeout=6, headers={"User-Agent": "DratingsClone/1.0"})
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    forecasts = data.get("list", [])
    if not isinstance(forecasts, list) or not forecasts:
        return None

    candidates = []
    for item in forecasts:
        try:
            dt_val = int(item.get("dt", 0))
        except (TypeError, ValueError, AttributeError):
            continue
        candidates.append((dt_val, item))
    if not candidates:
        return None

    dt_val, closest = min(candidates, key=lambda x: abs(x[0] - kickoff_unix))
    dt_diff = abs(dt_val - kickoff_unix)
    if dt_diff > 432000:
        return None

    try:
        return {
            "temp": float(closest["main"]["temp"]),
            "wind_speed": float(closest["wind"]["speed"]),
            "condition": str(closest["weather"][0]["main"]),
        }
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def get_weather(team: str, kickoff_unix: int):
    return _get_weather_cached(team, kickoff_unix, OWM_API_KEY)


def weather_adjustment(weather):
    if not weather:
        return 0
    pen = 0
    try:
        if weather.get("wind_speed", 0) > 20:
            pen -= 2
        if weather.get("condition", "").lower() in ["rain", "snow"]:
            pen -= 3
        if weather.get("temp", 100) < 25:
            pen -= 1
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


__all__ = ["OWM_API_KEY", "_get_weather_cached", "get_weather", "weather_adjustment", "default_kickoff_unix"]
