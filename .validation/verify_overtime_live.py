"""Exercise the live-source fallback using isolated data and output folders."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cfb_elo as model

cache_hash = hashlib.sha256((ROOT / "elo_parameters.csv").read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(dir=ROOT / ".validation", prefix="ot-live-test-") as folder:
    folder = Path(folder)
    data = folder / "data"
    shutil.copytree(ROOT / "data", data)
    shutil.copyfile(ROOT / "elo_parameters.csv", folder / "elo_parameters.csv")
    original_supplement = pd.read_csv(data / "cfb_overtime_supplement.csv")
    with patch.object(model, "tune_elo_params", side_effect=AssertionError("Unexpected retuning")):
        fit, outputs = model.main(refresh_data=True, output_dir=folder, data_dir=data, schedule_season=2026)
    updated_supplement = pd.read_csv(data / "cfb_overtime_supplement.csv")
    assert hashlib.sha256((folder / "elo_parameters.csv").read_bytes()).hexdigest() == cache_hash
    assert fit["games"]["overtime"].notna().all()
    with patch.object(model, "urlopen", side_effect=AssertionError("Unexpected offline ESPN lookup")):
        offline = model.load_games(data_dir=data, refresh_data=False, schedule_season=2026)
    assert offline.loc[offline.completed, "overtime"].notna().all()
    report = {"live_refresh_passed": True, "cached_seasons": len(fit["K_by_year"]),
              "tuner_calls": 0, "cache_unchanged": True,
              "new_verified_overtime_entries": len(updated_supplement) - len(original_supplement),
              "completed_model_games": len(fit["games"]),
              "fbs_rankings": len(outputs["current_elo_rankings.csv"]),
              "offline_reuse_passed": True}
    (ROOT / ".validation/overtime_live_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
