import copy
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cfb_site_data import portfolio_outputs


class SiteDataTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for game_id, year, phase, week, kickoff, done, away in [
            (1, 2025, "REG", 1, "2025-09-01T10:00-05:00", True, "2"),
            (2, 2026, "REG", 1, "2026-09-01T10:00-05:00", True, "2"),
            (3, 2026, "REG", 2, "2026-09-08T10:00-05:00", True, "3"),
            (4, 2026, "REG", 2, "2026-09-09T10:00-05:00", False, "2"),
            (5, 2026, "POST", 1, "2027-01-01T10:00-06:00", False, "2"),
        ]:
            rows.append(dict(game_id=game_id, season=year, game_type=phase, week=week,
                             kickoff=pd.Timestamp(kickoff).tz_convert("America/Chicago"),
                             completed=done, start_time_tbd=False, gameday=kickoff[:10], gametime="10:00:00",
                             home_team="1", away_team=away, home_team_name="First", away_team_name="FCS" if away == "3" else "Second",
                             home_conference="ACC", away_conference="FCS League" if away == "3" else "Big Ten",
                             home_division="fbs", away_division="fcs" if away == "3" else "fbs",
                             home_score=28 if done else float("nan"), away_score=14 if done else float("nan")))
        self.games = pd.DataFrame(rows)
        history = self.games.loc[self.games.completed].copy()
        history["elo_home_pre"], history["elo_away_pre"] = 1100., 900.
        history["elo_home_post"], history["elo_away_post"] = 1110., 890.
        history["p_home"], history["p_away"] = .7, .3
        self.fit = {"current_season": 2026, "games": history}
        self.spreads = history.assign(model_home_spread=-7.)
        self.outputs = {"current_elo_rankings.csv": pd.DataFrame({"rank": [1, 2], "team": ["1", "2"], "team_name": ["First", "Second"], "elo": [1110., 890.]})}
        future = self.games.loc[~self.games.completed].copy()
        future["home_elo"], future["away_elo"] = 1110., 890.
        future["p_home"], future["p_away"] = .75, .25
        future["model_home_spread"] = -8.
        self.outputs["future_projections.csv"] = future

    def test_week_includes_completed_game_and_fcs_opponent(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".validation", prefix="site-test-") as folder:
            result = portfolio_outputs(self.games, self.fit, copy.deepcopy(self.outputs), self.spreads,
                                       Path(folder) / "archive.csv", "2026-09-08T12:00-05:00")
            weekly = result["weekly_projections.csv"]
            self.assertEqual(weekly.game_id.tolist(), ["3", "4"])
            self.assertEqual(weekly.iloc[0].outcome, "First")
            self.assertEqual(weekly.iloc[0].p_home, .7)
            self.assertEqual(weekly.iloc[0].prediction_source, "Historical model")
            self.assertTrue(pd.isna(weekly.iloc[0].away_rank))
            self.assertTrue(pd.notna(weekly.iloc[0].away_elo))
            self.assertEqual(result["performance_games.csv"].evaluation_type.tolist(), ["historical"])
            self.assertEqual(result["current_elo_rankings.csv"].elo_delta.tolist(), [10., -10.])

    def test_recorded_forecast_freezes_and_only_then_counts_as_deployed(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".validation", prefix="site-test-") as folder:
            path = Path(folder) / "archive.csv"
            saved = self.games.loc[self.games.game_id == 3].copy()
            saved["home_elo"], saved["away_elo"] = 1090., 910.
            saved["p_home"], saved["p_away"] = .65, .35
            saved["model_home_spread"] = -5.
            saved["prediction_timestamp"] = "2026-09-08T09:00-05:00"
            saved.to_csv(path, index=False)
            result = portfolio_outputs(self.games, self.fit, copy.deepcopy(self.outputs), self.spreads,
                                       path, "2026-09-08T12:00-05:00")
            weekly = result["weekly_projections.csv"].set_index("game_id")
            self.assertEqual(weekly.loc["3", "p_home"], .65)
            self.assertEqual(weekly.loc["3", "prediction_source"], "Recorded pregame")
            self.assertEqual(result["performance_games.csv"].evaluation_type.tolist(), ["historical", "deployed"])

    def test_postseason_week_does_not_mix_with_regular_week(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".validation", prefix="site-test-") as folder:
            result = portfolio_outputs(self.games, self.fit, copy.deepcopy(self.outputs), self.spreads,
                                       Path(folder) / "archive.csv", "2026-12-31T12:00-06:00")
            self.assertEqual(result["weekly_projections.csv"].game_id.tolist(), ["5"])

    def test_prediction_recorded_after_kickoff_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".validation", prefix="site-test-") as folder:
            path = Path(folder) / "archive.csv"
            pd.DataFrame([{"game_id": "3", "prediction_timestamp": "2026-09-08T11:00-05:00", "kickoff": "2026-09-08T10:00-05:00"}]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "invalid prediction timestamp"):
                portfolio_outputs(self.games, self.fit, copy.deepcopy(self.outputs), self.spreads, path, "2026-09-08T12:00-05:00")
