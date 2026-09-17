import math
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from openpyxl import Workbook

os.environ["DRATINGS_SKIP_APP_MAIN"] = "1"

import app


class AppLogicTests(unittest.TestCase):
    def test_excel_file_uses_app_directory(self) -> None:
        self.assertTrue(Path(app.EXCEL_FILE).is_absolute())
        self.assertEqual(Path(app.EXCEL_FILE), app.APP_DIR / "games.xlsx")

    def test_update_ratings_uses_winner_loser_gap_regardless_of_team_order(self) -> None:
        underdog_first = {"Underdog": 1400.0, "Favorite": 1600.0}
        underdog_second = {"Favorite": 1600.0, "Underdog": 1400.0}

        app.update_ratings(underdog_first, "Underdog", "Favorite", 24, 14, None)
        app.update_ratings(underdog_second, "Favorite", "Underdog", 14, 24, None)

        underdog_gain_first = underdog_first["Underdog"] - 1400.0
        underdog_gain_second = underdog_second["Underdog"] - 1400.0
        favorite_loss_first = 1600.0 - underdog_first["Favorite"]
        favorite_loss_second = 1600.0 - underdog_second["Favorite"]

        self.assertAlmostEqual(underdog_gain_first, underdog_gain_second, places=9)
        self.assertAlmostEqual(favorite_loss_first, favorite_loss_second, places=9)

    def test_update_ratings_guards_extreme_underdog_multiplier(self) -> None:
        ratings = {"Huge Underdog": 0.0, "Huge Favorite": 4000.0}

        app.update_ratings(ratings, "Huge Underdog", "Huge Favorite", 17, 16, None)

        self.assertTrue(math.isfinite(ratings["Huge Underdog"]))
        self.assertTrue(math.isfinite(ratings["Huge Favorite"]))
        self.assertGreater(ratings["Huge Underdog"], 0.0)
        self.assertLess(ratings["Huge Favorite"], 4000.0)

    def test_kelly_fraction_respects_cap(self) -> None:
        self.assertLessEqual(app.kelly_fraction(0.99, 5.0, max_fraction=0.05), 0.05)
        self.assertEqual(app.kelly_fraction(0.40, 1.0), 0.0)

    def test_prepare_final_games_filters_non_final_statuses_and_keeps_scored_unknowns(self) -> None:
        history = pd.DataFrame(
            [
                {"season": 2025, "week": 1, "team1": "BUF", "team2": "NYJ", "score1": 24, "score2": 20, "status": "Final"},
                {"season": 2025, "week": 1, "team1": "MIA", "team2": "NE", "score1": 10, "score2": 7, "status": "Live"},
                {"season": 2025, "week": 1, "team1": "DAL", "team2": "PHI", "score1": 14, "score2": 17, "status": "Scheduled"},
                {"season": 2025, "week": 2, "team1": "KC", "team2": "DEN", "score1": 21, "score2": 17, "status": None},
            ]
        )

        prepared = app._prepare_final_games(history)

        self.assertEqual(prepared[["team1", "team2"]].values.tolist(), [["BUF", "NYJ"], ["KC", "DEN"]])

    def test_build_actual_results_and_grade_picks_only_finalize_completed_games(self) -> None:
        picks = pd.DataFrame(
            [
                {"week": 1, "matchup": "BUF @ NY Jets", "pick": "BUF", "timestamp": "2026-09-17T00:00:00"},
                {"week": 1, "matchup": "MIA @ NE", "pick": "NE", "timestamp": "2026-09-17T00:01:00"},
                {"week": 1, "matchup": "DAL @ PHI", "pick": "DAL", "timestamp": "2026-09-17T00:02:00"},
            ]
        )
        history = pd.DataFrame(
            [
                {"week": 1, "team1": "BUF", "team2": "NY Jets", "score1": 20, "score2": 17, "status": "Completed"},
                {"week": 1, "team1": "MIA", "team2": "NE", "score1": 14, "score2": 14, "status": "Final"},
                {"week": 1, "team1": "DAL", "team2": "PHI", "score1": 7, "score2": 3, "status": "Live"},
            ]
        )

        results = app.build_actual_results_by_week(history)
        graded = app.grade_picks(picks, results).sort_values("matchup").reset_index(drop=True)

        self.assertEqual(
            graded[["matchup", "status", "result", "winner"]].values.tolist(),
            [
                ["Buffalo Bills @ New York Jets", "correct", "✅ Correct", "Buffalo Bills"],
                ["Dallas Cowboys @ Philadelphia Eagles", "pending", "⏳ Pending", None],
                ["Miami Dolphins @ New England Patriots", "tie/push", "➖ Tie/Push", "TIE"],
            ],
        )

    def test_load_games_accepts_file_mtime_and_missing_schedule_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workbook_path = Path(temp_dir) / "games.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = app.HIST_SHEET
            sheet.append(["week", "team1", "team2", "score1", "score2"])
            sheet.append([1, "BUF", "NYJ", 24, 21])
            workbook.save(workbook_path)

            hist_df, sched_df = app.load_games(str(workbook_path), os.path.getmtime(workbook_path))

        self.assertEqual(hist_df.iloc[0]["team1"], "BUF")
        self.assertTrue(sched_df.empty)


if __name__ == "__main__":
    unittest.main()
