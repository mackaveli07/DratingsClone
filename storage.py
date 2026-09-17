# predictions.py
import pandas as pd
import streamlit as st

from config import BASE_ELO, DEFAULT_BANKROLL, HOME_ADVANTAGE
from elo import expected_score
from injuries import injury_adjustment
from team_utils import get_abbr, map_team_name
from weather import get_weather, weather_adjustment


def kelly_fraction(win_prob: float, odds_decimal: float, fraction: float = 0.25, max_fraction: float = 0.05) -> float:
    try:
        p = max(min(float(win_prob), 1.0), 0.0)
        odds_decimal = float(odds_decimal)
        fraction = max(float(fraction), 0.0)
        max_fraction = max(float(max_fraction), 0.0)
    except (TypeError, ValueError):
        return 0.0

    b = odds_decimal - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - p
    full_kelly = ((b * p) - q) / b
    stake_fraction = max(full_kelly, 0.0) * fraction
    return min(stake_fraction, max_fraction)


def get_available_weeks(schedule_df: pd.DataFrame):
    if schedule_df is None or schedule_df.empty or "week" not in schedule_df.columns:
        if isinstance(schedule_df, pd.DataFrame):
            return [], pd.Series(index=schedule_df.index, dtype="float64")
        return [], pd.Series(dtype="float64")

    week_series = pd.to_numeric(schedule_df["week"], errors="coerce")
    weeks = sorted(set(week_series.dropna().astype(int).tolist()))
    return weeks, week_series


@st.cache_data(ttl=600)
def get_total_points_baselines(hist_df: pd.DataFrame, alpha: float = 50.0):
    from elo import _prepare_final_games

    history = _prepare_final_games(hist_df)
    if history.empty:
        return {}, 44.0

    history["total_points"] = history["score1"] + history["score2"]
    overall_avg = float(history["total_points"].mean()) if len(history) else 44.0
    grouped = history.groupby("season")["total_points"].agg(["mean", "count"])
    season_avgs = {}
    for season, row in grouped.iterrows():
        season_avgs[int(season)] = (row["mean"] * row["count"] + overall_avg * alpha) / (row["count"] + alpha)
    return season_avgs, overall_avg


def normalize_matchup_key(away_team, home_team):
    return f"{map_team_name(away_team)} @ {map_team_name(home_team)}"


def normalize_matchup_value(matchup):
    if pd.isna(matchup):
        return None
    text = str(matchup).strip()
    parts = [p.strip() for p in text.split("@")]
    if len(parts) != 2:
        return text
    return normalize_matchup_key(parts[0], parts[1])


def compute_matchup_prediction(
    team_away,
    team_home,
    ratings,
    odds_away=2.0,
    odds_home=2.0,
    kickoff_ts=None,
    away_injuries=None,
    home_injuries=None,
    weather_away=None,
    weather_home=None,
):
    away = map_team_name(team_away)
    home = map_team_name(team_home)
    away_abbr = get_abbr(away)
    home_abbr = get_abbr(home)

    if kickoff_ts is None:
        kickoff_ts = 0

    if away_injuries is None:
        away_injuries = fetch_injuries_espn(away_abbr) if away_abbr else []
    if home_injuries is None:
        home_injuries = fetch_injuries_espn(home_abbr) if home_abbr else []

    if weather_away is None:
        weather_away = get_weather(away, kickoff_ts)
    if weather_home is None:
        weather_home = get_weather(home, kickoff_ts)

    adj_away = ratings.get(away, BASE_ELO) + injury_adjustment(away_injuries) + weather_adjustment(weather_away)
    adj_home = ratings.get(home, BASE_ELO) + injury_adjustment(home_injuries) + weather_adjustment(weather_home)

    win_prob_home = expected_score(adj_home + HOME_ADVANTAGE, adj_away)
    win_prob_away = 1.0 - win_prob_home

    kelly_home = kelly_fraction(win_prob_home, odds_home)
    kelly_away = kelly_fraction(win_prob_away, odds_away)

    return {
        "away": away,
        "home": home,
        "away_abbr": away_abbr,
        "home_abbr": home_abbr,
        "adj_away": adj_away,
        "adj_home": adj_home,
        "win_prob_away": win_prob_away,
        "win_prob_home": win_prob_home,
        "kelly_away": kelly_away,
        "kelly_home": kelly_home,
        "stake_away": kelly_away * DEFAULT_BANKROLL,
        "stake_home": kelly_home * DEFAULT_BANKROLL,
        "proj_away": int(round(win_prob_away * 44.0)),
        "proj_home": int(round(win_prob_home * 44.0)),
        "weather_away": weather_away,
        "weather_home": weather_home,
        "away_injuries": away_injuries,
        "home_injuries": home_injuries,
    }


__all__ = [
    "kelly_fraction",
    "get_available_weeks",
    "get_total_points_baselines",
    "normalize_matchup_key",
    "normalize_matchup_value",
    "compute_matchup_prediction",
]
