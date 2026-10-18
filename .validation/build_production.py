"""One-time initial extraction of the saved fit; not a rebuild of maintained cfb_elo.py.

The production loader has since gained live ESPN overtime verification. Do not
rerun this initializer over the maintained script or the active seasonal cache.
"""
import ast
import hashlib
import json
from pathlib import Path
from textwrap import indent

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
notebook_path = ROOT / "CFB_elo_v3.ipynb"
notebook = json.loads(notebook_path.read_text(encoding="utf-8"))


def source(index):
    return "".join(notebook["cells"][index]["source"])


final = ast.parse(source(45))
grids = {node.targets[0].id: ast.literal_eval(node.value)
         for node in final.body if isinstance(node, ast.Assign)
         and isinstance(node.targets[0], ast.Name)
         and node.targets[0].id in {"K_grid", "home_adv_grid", "carry_grid"}}
call = final.body[-1].value
settings = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords
            if isinstance(kw.value, ast.Constant)}
assert settings["first_train_year"] == 2005 and settings["first_eval_year"] == 2010
assert settings["param_tune_window"] == 5

output = notebook["cells"][46]["outputs"][-1]["data"][
    "application/vnd.microsoft.datawrangler.viewer.v0+json"]
cache = pd.DataFrame(output["rows"], columns=[c["name"] for c in output["columns"]])
cache = cache.drop(columns="index").rename(columns={"tuning_logloss": "logloss"})
for column in cache.columns:
    cache[column] = cache[column].map(int if column == "eval_year" else float)
assert cache.eval_year.tolist() == list(range(2010, 2027))
cache["train_start"] = settings["first_train_year"]
cache["train_end"] = cache.eval_year - 1
cache["param_tune_start"] = cache.eval_year - settings["param_tune_window"]
cache["param_tune_end"] = cache.eval_year - 1
cache["param_tune_n"] = settings["param_tune_window"]
cache["param_tune_window"] = settings["param_tune_window"]
for key in ["use_mov", "use_home_adv", "use_carry_grid", "scale", "start_rating"]:
    cache[key] = settings[key]
for key, values in grids.items():
    cache[key] = json.dumps(values)
for column, grid in [("K_best", "K_grid"), ("home_adv_best", "home_adv_grid"), ("carry_best", "carry_grid")]:
    assert cache[column].isin(grids[grid]).all()

header = '''"""Standalone FBS Elo from the saved final fit in CFB_elo_v3.ipynb.

Run: python cfb_elo.py (or --offline to reuse downloaded schedules).
Existing elo_parameters.csv seasons are reused without grid search. Missing
seasons tune once using prior data and the final notebook grid, then are saved.
Rankings remain FBS-only; non-FBS opponent ratings remain available for forecasts.
College spread fitting preserves the notebook's MAE/expanding-history method.
"""
import argparse
import json
from pathlib import Path
from itertools import product
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent
'''
header += "K_GRID = " + repr(grids["K_grid"]) + "\n"
header += "HOME_ADV_GRID = " + repr(grids["home_adv_grid"]) + "\n"
header += "CARRY_GRID = " + repr(grids["carry_grid"]) + "\n\n"
core = "\n\n".join(source(i) for i in [7, 11, 15, 20])
core = core.replace("from IPython.display import display\n", "")

loader = source(3)
loader = loader[loader.index("games = []"):loader.index('print(games.columns)')]
loader = loader.replace("if not schedule_file.exists() or (refresh_data and season == schedule_season):",
    "if not schedule_file.exists() and not refresh_data:\n        raise FileNotFoundError(f'Offline schedule missing: {schedule_file}')\n    if not schedule_file.exists() or (refresh_data and season == schedule_season):")
loader = loader.replace("if not overtime_file.exists() or refresh_data:",
    "if not overtime_file.exists() and not refresh_data:\n    raise FileNotFoundError(f'Offline overtime metadata missing: {overtime_file}')\nif not overtime_file.exists() or refresh_data:")
loader = '''def load_games(data_dir=None, refresh_data=True, schedule_season=None):
    data_dir = ROOT / "data" if data_dir is None else Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if schedule_season is None:
        schedule_season = pd.Timestamp.now(tz="America/Chicago").year
''' + indent(loader, "    ") + "\n    return games\n"

runner = source(25).replace("projection_season=None):", "projection_season=None, parameter_file=None):")
runner = runner.replace("    all_out = []", '''    saved_params = pd.DataFrame()
    if parameter_file is not None:
        parameter_file = Path(parameter_file)
        if parameter_file.exists():
            saved_params = pd.read_csv(parameter_file, float_precision="round_trip")
            required = {"eval_year", "train_start", "train_end", "param_tune_start",
                        "param_tune_end", "param_tune_n", "param_tune_window", "K_best",
                        "home_adv_best", "carry_best", "mov_scale", "logloss",
                        "use_mov", "use_home_adv", "use_carry_grid", "scale", "start_rating",
                        "K_grid", "home_adv_grid", "carry_grid"}
            if not required.issubset(saved_params.columns):
                raise ValueError("Incomplete Elo cache; do not silently retune saved seasons.")
            if saved_params["eval_year"].duplicated().any():
                raise ValueError("Duplicate seasons in Elo cache.")
            numeric = ["eval_year", "K_best", "home_adv_best", "carry_best", "mov_scale", "scale", "start_rating"]
            if not np.isfinite(saved_params[numeric]).all().all():
                raise ValueError("Non-finite Elo cache values.")
            if not (saved_params["carry_best"].between(0, 1) &
                    (saved_params["K_best"] > 0) & (saved_params["mov_scale"] > 0)).all():
                raise ValueError("Invalid Elo cache values.")
            saved_params = saved_params.set_index("eval_year")
    if not eval_years:
        raise ValueError("No evaluation seasons available.")
    new_params = []
    all_out = []''')
runner = runner.replace("        if fixed_param_mode:", '''        if yr in saved_params.index:
            saved = saved_params.loc[yr]
            expected = {"train_start": min(train_years), "train_end": max(train_years),
                        "param_tune_start": min(param_tune_years), "param_tune_end": max(param_tune_years),
                        "param_tune_n": len(param_tune_years), "param_tune_window": param_tune_window,
                        "scale": scale, "start_rating": start_rating, "use_mov": use_mov,
                        "use_home_adv": use_home_adv, "use_carry_grid": use_carry_grid}
            for key, value in expected.items():
                if not (pd.isna(saved[key]) and pd.isna(value)) and saved[key] != value:
                    raise ValueError(f"Cached {yr} {key} differs from requested settings; rebuild explicitly.")
            for key, values in [("K_grid", K_grid), ("home_adv_grid", home_adv_grid),
                                ("carry_grid", carryover_grid)]:
                if json.loads(saved[key]) != list(values):
                    raise ValueError(f"Cached {yr} {key} differs from requested grid; rebuild explicitly.")
            print(f"Using saved parameters for FBS season {yr}.", flush=True)
            K_best, home_adv_best, carry_best = saved["K_best"], saved["home_adv_best"], saved["carry_best"]
            mov_scale, best_logloss = saved["mov_scale"], saved["logloss"]
        elif fixed_param_mode:''')
runner = runner.replace("            param_fit = tune_elo_params(", '            print(f"No saved parameters for {yr}; tuning prior seasons once...", flush=True)\n            param_fit = tune_elo_params(')
runner = runner.replace('            "logloss": best_logloss', '''            "scale": scale, "start_rating": start_rating,
            "K_grid": json.dumps(np.asarray(K_grid).tolist()),
            "home_adv_grid": json.dumps(np.asarray(home_adv_grid).tolist()),
            "carry_grid": json.dumps(np.asarray(carryover_grid).tolist()),
            "logloss": best_logloss''')
runner = runner.replace('    return {\n        "games":', '''        if yr not in saved_params.index:
            new_params.append(param_out[-1])
            if parameter_file is not None:
                # Checkpoint each newly tuned season so an interrupted first run can resume.
                parameters = pd.concat([saved_params.reset_index(), pd.DataFrame(new_params)],
                                       ignore_index=True) if not saved_params.empty else pd.DataFrame(new_params)
                temporary = parameter_file.with_suffix(".csv.tmp")
                parameters.sort_values("eval_year").to_csv(temporary, index=False)
                temporary.replace(parameter_file)

    return {
        "games":''')

spread = source(48).replace("from scipy.optimize import minimize_scalar\n", "")
spread = spread[:spread.index('print("Prospective spread MAE:')]
spread = spread.replace('cfb_elo["games"]', 'historical_elo')
spread += "\n" + source(49).split("display(")[0]
spread += "\n" + source(51).split('print(f"Production')[0]
spread = "def fit_spreads(historical_elo):\n" + indent(spread, "    ")
spread += "\n    return spread_slope, spread_validation, spread_performance_by_year\n"

projections = source(53).replace('\ncurrent_ratings\n', '\n')
projections += "\n" + source(55).replace('as_of_date = pd.Timestamp.now(tz="America/Chicago").date()',
    'as_of_date = pd.Timestamp.now(tz="America/Chicago").date() if as_of_date is None else pd.Timestamp(as_of_date).date()')
projections = projections[:projections.index('display(future_projections)')]
projections = projections.replace('[["as_of_date", "game_type", "week",', '[["as_of_date", "game_id", "game_type", "week",')
projections = "def build_projections(games, cfb_elo, spread_slope, as_of_date=None):\n" + indent(projections, "    ")
projections += '''
    return {"current_elo_rankings.csv": current_ratings,
            "daily_projections.csv": daily_projections,
            "weekly_projections.csv": weekly_projections,
            "future_projections.csv": future_projections}
'''
main = '''
def main(refresh_data=True, output_dir=None, data_dir=None, schedule_season=None, as_of_date=None):
    output_dir = ROOT if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    games = load_games(data_dir, refresh_data, schedule_season)
    elo_df = games[games["game_type"].isin(["REG", "POST"]) &
                   games["home_score"].notna() & games["away_score"].notna()].copy()
    cfb_elo = run_expanding_k_elo(
        all_games=elo_df, first_train_year=2005, first_eval_year=2010,
        K_grid=K_GRID, home_adv_grid=HOME_ADV_GRID, carryover_grid=CARRY_GRID,
        scale=400, start_rating=1000, use_carry_grid=True, param_tune_window=5,
        use_mov=True, use_home_adv=True, projection_season=int(games["season"].max()),
        parameter_file=output_dir / "elo_parameters.csv")
    historical_elo = cfb_elo["games"]
    spread_slope, spread_validation, spread_performance = fit_spreads(historical_elo)
    outputs = build_projections(games, cfb_elo, spread_slope, as_of_date)
    evaluation = historical_elo.query("home_score != away_score").copy()
    evaluation["actual"] = (evaluation["home_score"] > evaluation["away_score"]).astype(int)
    evaluation["correct"] = (evaluation["p_home"] >= 0.5) == evaluation["actual"]
    evaluation["ll"] = logloss(evaluation["p_home"], evaluation["actual"])
    performance = evaluation.groupby("season").agg(
        games=("game_id", "size"), accuracy=("correct", "mean"), logloss=("ll", "mean")).reset_index()
    outputs.update({"cfb_elo_history.csv": historical_elo,
                    "performance_by_year.csv": performance,
                    "spread_performance_by_year.csv": spread_performance})
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)
        print(f"Saved {filename}: {len(frame):,} rows")
    print("Accuracy:", evaluation["correct"].mean())
    print("Log loss:", evaluation["ll"].mean())
    print("Prospective spread MAE:", spread_validation["spread_abs_error"].mean())
    print("Production spread slope:", spread_slope)
    return cfb_elo, outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Reuse downloaded schedule data.")
    parser.add_argument("--output-dir", type=Path, default=ROOT,
                        help="Output and Elo-cache folder (defaults to this script's folder).")
    args = parser.parse_args()
    main(refresh_data=not args.offline, output_dir=args.output_dir)
'''
script = "\n\n".join([header, core, loader, runner, spread, projections, main])
compile(script, "cfb_elo.py", "exec")
(ROOT / "cfb_elo.py").write_text(script, encoding="utf-8")
cache.to_csv(ROOT / "elo_parameters.csv", index=False)
provenance = {"notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
              "fit_cell": 45, "output_cell": 46, "fit_execution_count": notebook["cells"][45]["execution_count"],
              "cache_source": "Saved full-precision Data Wrangler output, not rounded display text",
              "grids": grids, "settings": settings, "cache_seasons": cache.eval_year.tolist()}
(ROOT / ".validation" / "production_cache_source.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
print("Built cfb_elo.py and seeded", len(cache), "full-precision seasonal cache rows.")
