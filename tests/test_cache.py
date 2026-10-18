"""Cache lifecycle and parity with the saved notebook's model functions."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cfb_elo as model


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads((ROOT / "CFB_elo_v3.ipynb").read_text(encoding="utf-8"))
        cls.reference = {}
        for index in [7, 11, 15, 20, 25]:
            exec("".join(notebook["cells"][index]["source"]), cls.reference)

    def setUp(self):
        self.games = pd.DataFrame([
            [2020, "2020-09-01", "12:00", 1, "1", "2", 28, 14, 0, "Home"],
            [2020, "2020-09-02", "12:00", 2, "3", "1", 24, 21, 1, "Neutral"],
            [2021, "2021-09-01", "12:00", 3, "2", "3", 21, 28, 0, "Home"],
            [2021, "2021-09-02", "12:00", 4, "1", "4", 17, 10, 0, "Home"],
            [2022, "2022-09-01", "12:00", 5, "4", "2", 10, 17, 1, "Home"],
            [2022, "2022-09-02", "12:00", 6, "3", "1", 28, 35, 0, "Neutral"],
        ], columns=["season", "gameday", "gametime", "game_id", "home_team", "away_team",
                    "home_score", "away_score", "overtime", "location"])
        self.kwargs = dict(all_games=self.games, first_train_year=2020, first_eval_year=2021,
                           K_grid=[20, 30], home_adv_grid=[25, 40], carryover_grid=[0.8, 0.95],
                           param_tune_window=1, projection_season=2022)

    def run_model(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return model.run_expanding_k_elo(**kwargs)

    def test_tune_once_then_exact_replay_and_new_season_append(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".validation", prefix="cache-test-") as folder:
            path = Path(folder) / "elo_parameters.csv"
            reference = self.reference["run_expanding_k_elo"](**self.kwargs)
            with patch.object(model, "tune_elo_params", wraps=model.tune_elo_params) as tuner:
                first = self.run_model(**self.kwargs, parameter_file=path)
                self.assertEqual(tuner.call_count, 2)
            pd.testing.assert_frame_equal(reference["games"], first["games"], check_exact=True)
            self.assertEqual(reference["ratings"], first["ratings"])
            original = path.read_bytes()
            with patch.object(model, "tune_elo_params", side_effect=AssertionError("Cache miss")):
                cached = self.run_model(**self.kwargs, parameter_file=path)
            pd.testing.assert_frame_equal(first["games"], cached["games"], check_exact=True)
            self.assertEqual(first["ratings"], cached["ratings"])
            self.assertEqual(original, path.read_bytes())

            upcoming = dict(self.kwargs, projection_season=2023)
            with patch.object(model, "tune_elo_params", wraps=model.tune_elo_params) as tuner:
                result = self.run_model(**upcoming, parameter_file=path)
                self.assertEqual(tuner.call_count, 1)
                self.assertEqual(tuner.call_args.kwargs["seasons_tune"], [2022])
            self.assertEqual(result["current_season"], 2023)
            saved = pd.read_csv(path)
            self.assertEqual(saved.eval_year.tolist(), [2021, 2022, 2023])
            self.assertTrue((saved.train_end < saved.eval_year).all())
            self.assertFalse(path.with_suffix(".csv.tmp").exists())
            with patch.object(model, "tune_elo_params", side_effect=AssertionError("Cache miss")):
                self.run_model(**upcoming, parameter_file=path)

    def test_bad_cache_fails_without_retuning(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".validation", prefix="cache-test-") as folder:
            path = Path(folder) / "elo_parameters.csv"
            self.run_model(**self.kwargs, parameter_file=path)
            original = pd.read_csv(path)
            cases = []
            cases.append(pd.concat([original, original.iloc[:1]], ignore_index=True))
            cases.append(original.drop(columns="mov_scale"))
            for column, value in [("train_end", 2021), ("K_best", float("inf")),
                                  ("carry_best", 1.5), ("scale", 500), ("K_grid", "[99]")]:
                invalid = original.copy()
                invalid.loc[0, column] = value
                cases.append(invalid)
            for invalid in cases:
                invalid.to_csv(path, index=False)
                with patch.object(model, "tune_elo_params", side_effect=AssertionError("Must not retune")):
                    with self.assertRaises(ValueError):
                        self.run_model(**self.kwargs, parameter_file=path)


if __name__ == "__main__":
    unittest.main()
