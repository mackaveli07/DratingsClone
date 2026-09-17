# elo.py
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from config import BASE_ELO, HOME_ADVANTAGE, K
from team_utils import map_team_name


def expected_score(r1, r2):
    return 1 / (1 + 10 ** ((r2 - r1) / 400))


def regress_preseason(elo_ratings, reg=0.65, base=BASE_ELO):
    for team in list(elo_ratings.keys()):
        elo_ratings[team] = base + reg * (elo_ratings[team] - base)


def update_ratings(elo_ratings, team1, team2, score1, score2, home_team):
    r1, r2 = elo_ratings[team1], elo_ratings[team2]
    if home_team == team1:
        r1 += HOME_ADVANTAGE
    elif home_team == team2:
        r2 += HOME_ADVANTAGE

    score1 = float(score1)
    score2 = float(score2)
    expected1 = expected_score(r1, r2)
    if score1 > score2:
        actual1 = 1.0
    elif score2 > score1:
        actual1 = 0.0
    else:
        actual1 = 0.5

    margin = max(abs(score1 - score2), 1.0)
    mov_mult = np.log(margin + 1) * (2.2 / ((r1 - r2) * 0.001 + 2.2))
    elo_ratings[team1] += K * mov_mult * (actual1 - expected1)
    elo_ratings[team2] += K * mov_mult * ((1 - actual1) - (1 - expected1))


def _normalize_status(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().replace("-", " ").replace("/", " ").split())


def _is_final_status(value) -> Optional[bool]:
    status = _normalize_status(value)
    if not status:
        return None
    tokens = set(status.split())
    if {"postponed", "cancelled", "canceled", "scheduled", "pregame", "halftime", "live"} & tokens:
        return False
    if status.startswith("q") or status in {"pre", "in"} or ("in" in tokens and "progress" in tokens):
        return False
    if "final" in tokens or {"complete", "completed"} & tokens or status == "post":
        return True
    return False


def _prepare_final_games(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"season", "week", "team1", "team2", "score1", "score2"}
    if df is None or df.empty or not needed.issubset(set(df.columns)):
        return pd.DataFrame(columns=["season", "week", "team1", "team2", "score1", "score2", "home_team"])

    games = df.copy()
    for col in ["season", "week", "score1", "score2"]:
        games[col] = pd.to_numeric(games[col], errors="coerce")
    games = games.dropna(subset=["season", "week", "team1", "team2", "score1", "score2"])

    status_col = next((c for c in ["status", "game_status", "state"] if c in games.columns), None)
    if status_col:
        status_eval = games[status_col].apply(_is_final_status)
        games = games[status_eval != False]

    games["season"] = games["season"].astype(int)
    games["week"] = games["week"].astype(int)
    return games.sort_values(["season", "week"]).reset_index(drop=True)


def run_elo_pipeline(df):
    games = _prepare_final_games(df)
    elo_ratings = defaultdict(lambda: BASE_ELO)
    has_home_col = "home_team" in games.columns
    prev_season = None

    for _, row in games.iterrows():
        season = int(row["season"])
        if prev_season is not None and season != prev_season:
            regress_preseason(elo_ratings)

        t1 = map_team_name(row.get("team1"))
        t2 = map_team_name(row.get("team2"))
        home_raw = row.get("home_team", None)
        if pd.notna(home_raw):
            home = map_team_name(home_raw)
        elif not has_home_col:
            home = t2
        else:
            home = None
        if home not in {t1, t2}:
            home = None

        update_ratings(elo_ratings, t1, t2, row["score1"], row["score2"], home)
        prev_season = season

    return dict(elo_ratings)


def compute_detailed_accuracy(hist_df: pd.DataFrame, elo_ratings=None):
    games = _prepare_final_games(hist_df)
    if games.empty:
        return {
            "overall_accuracy": 0,
            "brier_score": 1.0,
            "per_team_accuracy": {},
            "weekly_accuracy": {},
            "home_accuracy": 0,
            "away_accuracy": 0,
        }

    ratings = defaultdict(lambda: BASE_ELO)
    has_home_col = "home_team" in games.columns
    y_true, y_prob, correct, total = [], [], 0, 0
    per_team_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    weekly_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    home_stats = {"correct": 0, "total": 0}
    away_stats = {"correct": 0, "total": 0}
    prev_season = None

    for _, row in games.iterrows():
        t1 = map_team_name(row.get("team1"))
        t2 = map_team_name(row.get("team2"))
        home_raw = row.get("home_team", None)
        if pd.notna(home_raw):
            home_team = map_team_name(home_raw)
        elif not has_home_col:
            home_team = t2
        else:
            home_team = None
        if home_team == t1:
            away_team = t2
        elif home_team == t2:
            away_team = t1
        else:
            home_team = None
            away_team = None

        score1 = float(row["score1"])
        score2 = float(row["score2"])
        week = int(row["week"])
        season = int(row["season"])

        if prev_season is not None and season != prev_season:
            regress_preseason(ratings)

        e1 = ratings[t1] + (HOME_ADVANTAGE if home_team == t1 else 0)
        e2 = ratings[t2] + (HOME_ADVANTAGE if home_team == t2 else 0)
        prob1 = expected_score(e1, e2)

        update_ratings(ratings, t1, t2, score1, score2, home_team)
        prev_season = season

        if score1 == score2:
            continue

        actual_team1_won = score1 > score2
        predicted_team1_won = prob1 >= 0.5
        predicted_winner = t1 if predicted_team1_won else t2
        actual_winner = t1 if actual_team1_won else t2

        y_prob.append(prob1)
        y_true.append(1 if actual_team1_won else 0)
        is_correct = predicted_winner == actual_winner
        correct += int(is_correct)
        total += 1

        for team in (t1, t2):
            per_team_stats[team]["total"] += 1
            if (predicted_winner == team) == (actual_winner == team):
                per_team_stats[team]["correct"] += 1

        weekly_stats[week]["total"] += 1
        weekly_stats[week]["correct"] += int(is_correct)

        if predicted_winner == home_team:
            home_stats["total"] += 1
            home_stats["correct"] += int(actual_winner == home_team)
        elif predicted_winner == away_team:
            away_stats["total"] += 1
            away_stats["correct"] += int(actual_winner == away_team)

    overall_accuracy = correct / total if total else 0
    brier = brier_score_loss(y_true, y_prob) if y_true else 1.0
    per_team_accuracy = {team: stats["correct"] / stats["total"] if stats["total"] else 0 for team, stats in per_team_stats.items()}
    weekly_accuracy = {week: stats["correct"] / stats["total"] if stats["total"] else 0 for week, stats in weekly_stats.items()}
    home_accuracy = home_stats["correct"] / home_stats["total"] if home_stats["total"] else 0
    away_accuracy = away_stats["correct"] / away_stats["total"] if away_stats["total"] else 0

    return {
        "overall_accuracy": overall_accuracy,
        "brier_score": brier,
        "per_team_accuracy": per_team_accuracy,
        "weekly_accuracy": weekly_accuracy,
        "home_accuracy": home_accuracy,
        "away_accuracy": away_accuracy,
    }


__all__ = [
    "expected_score",
    "regress_preseason",
    "update_ratings",
    "_normalize_status",
    "_is_final_status",
    "_prepare_final_games",
    "run_elo_pipeline",
    "compute_detailed_accuracy",
]
