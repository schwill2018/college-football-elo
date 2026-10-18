"""Missing bulk metadata must be verified against matching ESPN final results."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cfb_elo as model


class OvertimeTests(unittest.TestCase):
    def setUp(self):
        self.games = pd.DataFrame([
            {"game_id": 123, "completed": True, "overtime": np.nan,
             "home_id": 1, "away_id": 2, "home_team": "Home", "away_team": "Away",
             "home_points": 28, "away_points": 21},
            {"game_id": 124, "completed": False, "overtime": np.nan,
             "home_id": 3, "away_id": 4, "home_team": "Next", "away_team": "Later",
             "home_points": 0, "away_points": 0},
        ])
        self.competition = {"id": "123", "status": {"type": {
            "completed": True, "state": "post", "detail": "Final"}},
            "competitors": [{"id": "1", "homeAway": "home", "score": "28"},
                            {"id": "2", "homeAway": "away", "score": "21"}]}

    def response(self, competition):
        return io.BytesIO(json.dumps({"header": {"competitions": [competition]}}).encode())

    def seed(self, folder):
        path = Path(folder) / "supplement.csv"
        path.write_text("game_id,overtime,start_date,source\n99,0,2004-12-04T21:00Z,original\n", encoding="utf-8")
        return path

    def test_regulation_and_overtime_are_verified_and_cached(self):
        for detail, overtime in [("Final", 0), ("Final/OT", 1), ("Final/3OT", 1)]:
            with self.subTest(detail=detail), tempfile.TemporaryDirectory(dir=ROOT / ".validation", prefix="ot-test-") as folder:
                path = self.seed(folder)
                games = self.games.copy()
                competition = copy.deepcopy(self.competition)
                competition["status"]["type"]["detail"] = detail
                with patch.object(model, "urlopen", return_value=self.response(competition)) as fetch:
                    model.fill_missing_overtime(games, path, True)
                    self.assertEqual(fetch.call_count, 1)
                self.assertEqual(games.loc[0, "overtime"], overtime)
                self.assertTrue(pd.isna(games.loc[1, "overtime"]))
                saved = pd.read_csv(path).set_index("game_id")
                self.assertEqual(saved.loc[123, "overtime"], overtime)
                self.assertEqual(saved.loc[99, "start_date"], "2004-12-04T21:00Z")
                before = path.read_bytes()
                with patch.object(model, "urlopen", side_effect=AssertionError("Cached result fetched again")):
                    model.fill_missing_overtime(games, path, True)
                self.assertEqual(before, path.read_bytes())

    def test_unverified_or_mismatched_results_fail_without_saving(self):
        cases = []
        for field, value in [("completed", False), ("state", "in"), ("detail", "End of 4th"),
                              ("detail", "Final/Suspended")]:
            competition = copy.deepcopy(self.competition)
            competition["status"]["type"][field] = value
            cases.append(competition)
        for field, value in [("score", "27"), ("id", "999")]:
            competition = copy.deepcopy(self.competition)
            competition["competitors"][0][field] = value
            cases.append(competition)
        competition = copy.deepcopy(self.competition)
        competition["id"] = "999"
        cases.append(competition)
        for competition in cases:
            with tempfile.TemporaryDirectory(dir=ROOT / ".validation", prefix="ot-test-") as folder:
                path = self.seed(folder)
                before = path.read_bytes()
                with patch.object(model, "urlopen", return_value=self.response(competition)):
                    with self.assertRaisesRegex(ValueError, "Cannot verify completed game 123"):
                        model.fill_missing_overtime(self.games.copy(), path, True)
                self.assertEqual(before, path.read_bytes())

    def test_offline_failure_lists_game_and_does_not_fetch(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".validation", prefix="ot-test-") as folder:
            path = self.seed(folder)
            with patch.object(model, "urlopen", side_effect=AssertionError("Offline network access")):
                with self.assertRaisesRegex(ValueError, "game IDs 123"):
                    model.fill_missing_overtime(self.games.copy(), path, False)

    def test_delayed_bulk_status_gets_final_scores_and_ot_from_espn(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".validation", prefix="ot-test-") as folder:
            path = self.seed(folder)
            games = self.games.iloc[:1].copy()
            games["completed"], games["status"] = False, "STATUS_IN_PROGRESS"
            games["season"], games["start_date"] = 2026, "2026-10-17T17:00Z"
            games["home_points"], games["away_points"] = 7, 3
            competition = copy.deepcopy(self.competition)
            competition["status"]["type"]["detail"] = "Final/2OT"
            with patch.object(model, "urlopen", return_value=self.response(competition)):
                model.refresh_recent_results(games, path, 2026, "2026-10-18T05:00Z")
            self.assertTrue(games.iloc[0].completed)
            self.assertEqual(games.iloc[0].home_points, 28)
            self.assertEqual(games.iloc[0].away_points, 21)
            self.assertEqual(games.iloc[0].status, "STATUS_FINAL")
            self.assertEqual(pd.read_csv(path).set_index("game_id").loc[123, "overtime"], 1)

    def test_live_result_is_not_marked_final(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".validation", prefix="ot-test-") as folder:
            path = self.seed(folder)
            games = self.games.iloc[:1].copy()
            games["completed"], games["status"] = False, "STATUS_IN_PROGRESS"
            games["season"], games["start_date"] = 2026, "2026-10-17T17:00Z"
            competition = copy.deepcopy(self.competition)
            competition["status"]["type"]["completed"] = False
            before = path.read_bytes()
            with patch.object(model, "urlopen", return_value=self.response(competition)):
                model.refresh_recent_results(games, path, 2026, "2026-10-18T05:00Z")
            self.assertFalse(games.iloc[0].completed)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
