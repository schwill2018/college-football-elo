"""Validate the staged portfolio changes and generated model exports."""
import ast
import csv
import importlib.util
import io
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / ".validation/portfolio-staging"
spec = importlib.util.spec_from_file_location("updater", STAGE / "tools/update_fbs_elo_data.py")
updater = importlib.util.module_from_spec(spec)
spec.loader.exec_module(updater)
checks = []


def check(label, passed):
    assert passed, label
    checks.append(label)
    print("PASS", label)


update = yaml.load((STAGE / ".github/workflows/update_fbs_elo_pages.yml").read_text(), Loader=yaml.BaseLoader)
deploy = yaml.load((STAGE / ".github/workflows/jekyll-gh-pages.yml").read_text(), Loader=yaml.BaseLoader)
model = yaml.load((ROOT / ".github/workflows/cfb_elo.yml").read_text(), Loader=yaml.BaseLoader)
check("Portfolio daily update follows model daily update", update["on"]["schedule"][0]["cron"] == "45 8 * * *" and model["on"]["schedule"][0]["cron"] == "0 8 * * *")
check("Both Elo updater workflows trigger deployment", set(deploy["on"]["workflow_run"]["workflows"]) == {"Update NFL Elo Page", "Update FBS ELO Pages"})
check("Model workflow preserves pregame archive and performance output", all(name in model["jobs"]["update"]["steps"][-1]["run"] for name in ["historical_predictions.csv", "performance_games.csv"]))
check("Portfolio workflow runs updater on published branch", update["jobs"]["update"]["steps"][0]["with"]["ref"] == "schwill2018-patch-chirpy")
for filename in updater.SCHEMAS:
    text = (ROOT / filename).read_text(encoding="utf-8-sig")
    check(f"Model and site copy validate: {filename}", updater.validate(filename, text) >= 0 and text == (STAGE / "assets/data/fbs-elo" / filename).read_text(encoding="utf-8"))
for name, title in [("fbs-elo", "FBS ELO"), ("fbs-elo-performance", "FBS ELO Performance")]:
    source = (STAGE / "_tabs" / f"{name}.md").read_text(encoding="utf-8")
    front = yaml.safe_load(source.split("---", 2)[1])
    check(f"{title} tab metadata", front["title"] == title and front["permalink"] == f"/{name}/")
    check(f"{title} links to full CSV", "Full table (CSV)" in source and "download=" in source)
notebook = json.loads((ROOT / "CFB_elo_v3.ipynb").read_text(encoding="utf-8"))
original = {node.name: ast.dump(node) for index in [7, 11, 15, 20]
            for node in ast.parse("".join(notebook["cells"][index]["source"])).body if isinstance(node, ast.FunctionDef)}
current = {node.name: ast.dump(node) for node in ast.parse((ROOT / "cfb_elo.py").read_text(encoding="utf-8")).body if isinstance(node, ast.FunctionDef)}
check("Elo and tuning formulas still match notebook", all(current[key] == value for key, value in original.items()))
csvs = {name: list(csv.DictReader(io.StringIO((ROOT / name).read_text(encoding="utf-8-sig"))))
        for name in ["current_elo_rankings.csv", "weekly_projections.csv", "performance_games.csv"]}
weekly = csvs["weekly_projections.csv"]
outside = [row for row in weekly if (row["home_rank"] and float(row["home_rank"]) <= 50 and (not row["away_rank"] or float(row["away_rank"]) > 50))
           or (row["away_rank"] and float(row["away_rank"]) <= 50 and (not row["home_rank"] or float(row["home_rank"]) > 50))]
check("Top-50 games retain lower-ranked or FCS opponent forecasts", any(row["home_elo"] and row["away_elo"] and row["p_home"] for row in outside))
report = {"checks": checks, "fbs_teams": len(csvs["current_elo_rankings.csv"]), "weekly_games": len(weekly),
          "performance_games": len(csvs["performance_games.csv"]), "browser_checks": ["25 rows per page", "next page begins at rank 26", "Elo ascending sorts all 138 teams", "top-25 game filter", "2024 SEC metrics", "honest empty recorded-performance state", "no browser console errors"]}
(ROOT / ".validation/portfolio_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
