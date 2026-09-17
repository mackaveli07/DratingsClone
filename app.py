# storage.py
import logging
import os
import tempfile
import time
from contextlib import contextmanager
from typing import Mapping

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook

from config import EXCEL_FILE, HIST_SHEET, NFL_FULL_NAMES, PICKS_SHEET, SCHEDULE_SHEET
from team_utils import map_team_name

try:
    import fcntl
except ImportError:
    fcntl = None

logger = logging.getLogger(__name__)
PICKS_COLUMNS = ["week", "matchup", "pick", "timestamp"]
VALID_TEAM_NAMES = frozenset(NFL_FULL_NAMES.values())


def _read_excel_sheet(file: str | os.PathLike[str], sheet_name: str) -> pd.DataFrame:
    try:
        return pd.read_excel(file, sheet_name=sheet_name)
    except ValueError as exc:
        if f"Worksheet named '{sheet_name}' not found" in str(exc):
            logger.warning("Worksheet %r not found in %s", sheet_name, file)
            return pd.DataFrame()
        raise


def _normalize_team_name(name: object) -> str | None:
    mapped = map_team_name(name)
    if mapped in VALID_TEAM_NAMES:
        return mapped
    return None


def normalize_matchup(matchup: object) -> str | None:
    if pd.isna(matchup):
        return None

    parts = [part.strip() for part in str(matchup).split("@")]
    if len(parts) != 2:
        return None

    away_team = _normalize_team_name(parts[0])
    home_team = _normalize_team_name(parts[1])
    if not away_team or not home_team:
        return None

    return f"{away_team} @ {home_team}"


def _normalize_saved_picks_frame(df: pd.DataFrame, *, drop_invalid: bool) -> pd.DataFrame:
    normalized = df.copy()
    for col in PICKS_COLUMNS:
        if col not in normalized.columns:
            normalized[col] = np.nan

    normalized = normalized[PICKS_COLUMNS]
    normalized["matchup"] = normalized["matchup"].apply(normalize_matchup)
    normalized["pick"] = normalized["pick"].apply(_normalize_team_name)

    if not drop_invalid:
        return normalized

    normalized["week"] = pd.to_numeric(normalized["week"], errors="coerce")
    normalized = normalized.dropna(subset=["week", "matchup", "pick"])
    if normalized.empty:
        return pd.DataFrame(columns=PICKS_COLUMNS)

    normalized["week"] = normalized["week"].astype(int)
    return normalized[PICKS_COLUMNS]


def _is_final_status(status: object, score_complete: bool) -> bool:
    if pd.isna(status):
        return score_complete

    normalized_status = " ".join(str(status).strip().lower().replace("-", " ").replace("/", " ").split())
    if not normalized_status:
        return score_complete

    status_tokens = set(normalized_status.split())
    if "postponed" in status_tokens:
        return False
    if (
        "final" in status_tokens
        or normalized_status == "post"
        or "complete" in status_tokens
        or "completed" in status_tokens
    ):
        return score_complete
    if (
        normalized_status.startswith("q")
        or "quarter" in status_tokens
        or "quarters" in status_tokens
        or "live" in status_tokens
        or "halftime" in status_tokens
        or "scheduled" in status_tokens
        or "pregame" in status_tokens
        or "pre" in status_tokens
        or ("in" in status_tokens and "progress" in status_tokens)
    ):
        return False
    return score_complete


def _atomic_workbook_save(workbook: Workbook, file: str | os.PathLike[str]) -> None:
    file_path = os.fspath(file)
    directory = os.path.dirname(os.path.abspath(file_path)) or "."
    os.makedirs(directory, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(file_path)}.",
        suffix=os.path.splitext(file_path)[1] or ".xlsx",
        dir=directory,
    )
    os.close(fd)
    try:
        workbook.save(temp_path)
        os.replace(temp_path, file_path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def load_games(file: str | os.PathLike[str] = EXCEL_FILE) -> tuple[pd.DataFrame, pd.DataFrame]:
    if os.path.exists(file):
        hist_df = _read_excel_sheet(file, HIST_SHEET)
        sched_df = _read_excel_sheet(file, SCHEDULE_SHEET)
        return hist_df, sched_df
    return pd.DataFrame(), pd.DataFrame()


def load_saved_picks(file: str | os.PathLike[str] = EXCEL_FILE) -> pd.DataFrame:
    if not os.path.exists(file):
        return pd.DataFrame(columns=PICKS_COLUMNS)
    try:
        df = pd.read_excel(file, sheet_name=PICKS_SHEET)
    except ValueError as exc:
        if f"Worksheet named '{PICKS_SHEET}' not found" in str(exc):
            return pd.DataFrame(columns=PICKS_COLUMNS)
        raise
    return _normalize_saved_picks_frame(df, drop_invalid=False)


def _read_saved_picks_from_excel(file: str | os.PathLike[str] = EXCEL_FILE) -> pd.DataFrame:
    return load_saved_picks(file)


def _write_picks_sheet(
    file: str | os.PathLike[str], picks_df: pd.DataFrame, columns: list[str]
) -> None:
    file_path = os.fspath(file)
    os.makedirs(os.path.dirname(os.path.abspath(file_path)) or ".", exist_ok=True)
    if os.path.exists(file_path):
        workbook = load_workbook(file_path)
        if PICKS_SHEET in workbook.sheetnames:
            sheet = workbook[PICKS_SHEET]
        else:
            sheet = workbook.create_sheet(PICKS_SHEET)
    else:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = PICKS_SHEET

    if sheet.max_row and sheet.max_row > 0:
        sheet.delete_rows(1, sheet.max_row)
    for col_idx, col_name in enumerate(columns, start=1):
        sheet.cell(row=1, column=col_idx, value=col_name)
    for row_idx, row in enumerate(picks_df.itertuples(index=False, name=None), start=2):
        for col_idx, value in enumerate(row, start=1):
            sheet.cell(row=row_idx, column=col_idx, value=value)
    _atomic_workbook_save(workbook, file_path)


@contextmanager
def picks_file_lock(file: str | os.PathLike[str]):
    lock_path = f"{file}.lock"
    if fcntl:
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        return

    lock_dir = f"{lock_path}.d"
    acquired = False
    try:
        for _ in range(100):
            try:
                os.mkdir(lock_dir)
                acquired = True
                break
            except FileExistsError:
                time.sleep(0.05)
        if not acquired:
            raise TimeoutError("Could not acquire picks save lock.")
        yield
    finally:
        if acquired and os.path.isdir(lock_dir):
            os.rmdir(lock_dir)


def save_week_picks(
    week: object, picks_dict: Mapping[object, object] | None, file: str | os.PathLike[str] = EXCEL_FILE
) -> bool:
    try:
        week_int = int(week)
    except (TypeError, ValueError):
        return False

    rows = []
    now_ts = pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S")
    if picks_dict is None:
        picks_items = {}
    elif hasattr(picks_dict, "items"):
        picks_items = picks_dict
    else:
        return False

    for matchup, pick in picks_items.items():
        norm_matchup = normalize_matchup(matchup)
        norm_pick = _normalize_team_name(pick)
        if not norm_matchup or not norm_pick:
            continue
        rows.append({"week": week_int, "matchup": norm_matchup, "pick": norm_pick, "timestamp": now_ts})

    if not rows:
        return False

    new_rows = pd.DataFrame(rows, columns=PICKS_COLUMNS)

    def _save_once():
        existing = _normalize_saved_picks_frame(_read_saved_picks_from_excel(file), drop_invalid=True)
        if not existing.empty:
            existing = existing[~((existing["week"] == week_int) & (existing["matchup"].isin(new_rows["matchup"])))]

        out = pd.concat([existing[PICKS_COLUMNS], new_rows], ignore_index=True)
        out = out.drop_duplicates(subset=["week", "matchup"], keep="last")
        _write_picks_sheet(file, out, PICKS_COLUMNS)

    os.makedirs(os.path.dirname(os.path.abspath(os.fspath(file))) or ".", exist_ok=True)
    with picks_file_lock(file):
        _save_once()

    return True


def build_actual_results_by_week(hist_df: pd.DataFrame) -> pd.DataFrame:
    needed = {"week", "team1", "team2", "score1", "score2"}
    cols = ["week", "matchup", "winner", "is_final"]
    if hist_df is None or hist_df.empty or not needed.issubset(set(hist_df.columns)):
        return pd.DataFrame(columns=cols)

    rows = []
    for _, row in hist_df.iterrows():
        week = pd.to_numeric(row.get("week"), errors="coerce")
        if pd.isna(week):
            continue
        away_team = _normalize_team_name(row.get("team1"))
        home_team = _normalize_team_name(row.get("team2"))
        if not away_team or not home_team:
            continue

        score1 = pd.to_numeric(row.get("score1"), errors="coerce")
        score2 = pd.to_numeric(row.get("score2"), errors="coerce")
        score_complete = pd.notna(score1) and pd.notna(score2)

        status_raw = row.get("status", row.get("game_status", row.get("state", None)))
        is_final = _is_final_status(status_raw, bool(score_complete))

        winner = None
        if is_final:
            if score1 > score2:
                winner = away_team
            elif score2 > score1:
                winner = home_team
            else:
                winner = "TIE"

        matchup = normalize_matchup(f"{away_team} @ {home_team}")
        if not matchup:
            continue

        rows.append({
            "week": int(week),
            "matchup": matchup,
            "winner": winner,
            "is_final": bool(is_final),
        })

    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).drop_duplicates(subset=["week", "matchup"], keep="last")


def grade_picks(saved_picks_df: pd.DataFrame, results_df: pd.DataFrame | None) -> pd.DataFrame:
    graded_cols = ["week", "matchup", "pick", "timestamp", "winner", "is_final", "status", "result"]
    if saved_picks_df is None or saved_picks_df.empty:
        return pd.DataFrame(columns=graded_cols)

    picks = _normalize_saved_picks_frame(saved_picks_df, drop_invalid=True)
    if picks.empty:
        return pd.DataFrame(columns=graded_cols)

    if results_df is None or results_df.empty:
        merged = picks.copy()
        merged["winner"] = None
        merged["is_final"] = False
    else:
        results = results_df.copy()
        results["week"] = pd.to_numeric(results["week"], errors="coerce")
        results = results.dropna(subset=["week", "matchup"])
        results["matchup"] = results["matchup"].apply(normalize_matchup)
        if "winner" in results.columns:
            results["winner"] = results["winner"].apply(
                lambda winner: "TIE" if winner == "TIE" else _normalize_team_name(winner)
            )
        else:
            results["winner"] = None
        if "is_final" not in results.columns:
            results["is_final"] = False
        results = results.dropna(subset=["week", "matchup"])
        results["week"] = results["week"].astype(int)
        merged = picks.merge(results[["week", "matchup", "winner", "is_final"]], on=["week", "matchup"], how="left")
        merged["is_final"] = merged["is_final"].fillna(False)

    merged["status"] = np.where(
        merged["is_final"] != True,
        "pending",
        np.where(
            merged["winner"] == "TIE",
            "tie/push",
            np.where(merged["pick"] == merged["winner"], "correct", "wrong"),
        ),
    )
    merged["result"] = np.where(
        merged["status"] == "correct",
        "✅ Correct",
        np.where(
            merged["status"] == "wrong",
            "❌ Wrong",
            np.where(merged["status"] == "tie/push", "➖ Tie/Push", "⏳ Pending"),
        ),
    )
    return merged[graded_cols]


__all__ = [
    "load_games",
    "load_saved_picks",
    "_read_saved_picks_from_excel",
    "_write_picks_sheet",
    "picks_file_lock",
    "save_week_picks",
    "build_actual_results_by_week",
    "grade_picks",
]
