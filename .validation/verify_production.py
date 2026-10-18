"""Replay the real saved data without tuning or touching notebook exports."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cfb_elo as model

start = time.monotonic()
notebook_hash = hashlib.sha256((ROOT / "CFB_elo_v3.ipynb").read_bytes()).hexdigest()
cache_bytes = (ROOT / "elo_parameters.csv").read_bytes()
with tempfile.TemporaryDirectory(dir=ROOT / ".validation", prefix="production-test-") as folder:
    output = Path(folder)
    shutil.copyfile(ROOT / "elo_parameters.csv", output / "elo_parameters.csv")
    with patch.object(model, "tune_elo_params", side_effect=AssertionError("Unexpected retuning")), \
            patch.object(model, "urlretrieve", side_effect=AssertionError("Unexpected download")):
        fit, outputs = model.main(refresh_data=False, output_dir=output,
                                  schedule_season=2026, as_of_date="2026-10-17")
    for filename in ["current_elo_rankings.csv", "daily_projections.csv",
                     "weekly_projections.csv", "future_projections.csv"]:
        expected = pd.read_csv(ROOT / filename, dtype={"team": str, "away_team": str, "home_team": str},
                               float_precision="round_trip")
        actual = pd.read_csv(output / filename, dtype={"team": str, "away_team": str, "home_team": str},
                             float_precision="round_trip")
        pd.testing.assert_frame_equal(expected, actual[expected.columns], check_exact=False, rtol=1e-12, atol=1e-12)
        print("PASS notebook CSV parity:", filename)
    assert (output / "elo_parameters.csv").read_bytes() == cache_bytes
    evaluation = fit["games"]
    accuracy = ((evaluation.p_home >= .5) == (evaluation.home_score > evaluation.away_score)).mean()
    loss = model.logloss(evaluation.p_home, (evaluation.home_score > evaluation.away_score).astype(int)).mean()
    np.testing.assert_allclose(accuracy, 0.7324750654311382, rtol=0, atol=1e-14)
    np.testing.assert_allclose(loss, 0.523675563335924, rtol=0, atol=1e-14)
    futures = outputs["future_projections.csv"]
    non_fbs = futures.home_rank.isna() | futures.away_rank.isna()
    assert non_fbs.any()
    assert futures.loc[non_fbs, ["home_elo", "away_elo", "p_home", "p_away"]].notna().all().all()
    assert len(outputs["current_elo_rankings.csv"]) == 138
    assert (ROOT / "elo_parameters.csv").read_bytes() == cache_bytes
    assert hashlib.sha256((ROOT / "CFB_elo_v3.ipynb").read_bytes()).hexdigest() == notebook_hash
    report = {"accuracy": accuracy, "logloss": loss, "historical_games": len(evaluation),
              "ranked_fbs_teams": 138, "cached_seasons": len(fit["K_by_year"]),
              "tuner_calls": 0, "notebook_csv_parity": True, "cache_unchanged": True,
              "notebook_unchanged": True, "elapsed_seconds": time.monotonic() - start}
    (ROOT / ".validation" / "production_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
