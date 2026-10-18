"""Compare fresh schedule/OT sources without overwriting the user's data."""
import json
from pathlib import Path
import tempfile
from urllib.request import urlretrieve, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(dir=ROOT / ".validation", prefix="ot-diagnostic-") as folder:
    folder = Path(folder)
    schedule_file = folder / "schedule.parquet"
    master_file = folder / "master.parquet"
    urlretrieve("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/cfb_schedules/cfb_schedules_2026.parquet", schedule_file)
    urlretrieve("https://raw.githubusercontent.com/sportsdataverse/cfbfastR-cfb-raw/main/cfb/cfb_schedule_master.parquet", master_file)
    frames = [pd.read_parquet(ROOT / "data" / f"cfb_schedules_{year}.parquet") for year in range(2003, 2026)]
    frames.append(pd.read_parquet(schedule_file))
    games = pd.concat(frames, ignore_index=True)
    games = games[games.fbs_participant.fillna(False) & games.season_type.isin(["regular", "postseason"]) &
                  ~games.status.isin(["STATUS_CANCELED", "STATUS_CANCELLED"])].copy()
    games["fixture_date"] = games.start_date.str[:10]
    games = games.sort_values("game_id").drop_duplicates(["season", "home_team", "away_team", "fixture_date"], keep="last")
    master = pd.read_parquet(master_file).set_index("game_id")
    ot = ((master.status_period > 4) | master.status_type_detail.str.contains("OT", na=False)).astype(float)
    ot = ot.where(master.status_type_completed)
    supplement = pd.read_csv(ROOT / "data/cfb_overtime_supplement.csv").set_index("game_id")
    games["overtime"] = games.game_id.map(ot).fillna(games.game_id.map(supplement.overtime))
    missing = games[games.completed & games.overtime.isna()]
    print("Missing completed games:", len(missing), flush=True)
    print(missing[["game_id", "season", "home_team", "away_team", "home_points", "away_points", "status"]].to_string(index=False), flush=True)
    report = []
    for row in missing.itertuples():
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary?event={int(row.game_id)}"
        try:
            with urlopen(url, timeout=30) as response:
                competition = json.load(response)["header"]["competitions"][0]
            entry = {"game_id": int(row.game_id), "status": competition["status"],
                     "scores": {c["homeAway"]: c["score"] for c in competition["competitors"]}}
        except Exception as exc:
            entry = {"game_id": int(row.game_id), "error": str(exc)}
        report.append(entry)
        print(json.dumps(entry), flush=True)
    (ROOT / ".validation/overtime_diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
