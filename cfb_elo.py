"""Standalone FBS Elo from the saved final fit in CFB_elo_v3.ipynb.

Run: python cfb_elo.py (or --offline to reuse downloaded schedules).
Existing elo_parameters.csv seasons are reused without grid search. Missing
seasons tune once using prior data and the final notebook grid, then are saved.
Rankings remain FBS-only; non-FBS opponent ratings remain available for forecasts.
College spread fitting preserves the notebook's MAE/expanding-history method.
"""
import argparse
import json
import re
from pathlib import Path
from itertools import product
from urllib.request import urlretrieve, urlopen

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from cfb_site_data import portfolio_outputs

ROOT = Path(__file__).resolve().parent
K_GRID = [47, 50, 53, 56]
HOME_ADV_GRID = [67, 70, 73, 76]
CARRY_GRID = [0.85, 0.9, 0.95]



import numpy as np
import pandas as pd


def null_coalesce(x, y):
    return y if x is None else x

def compress_exp_grid(min_val, max_val, n=10, curve=2):
    grid = np.unique(np.round(
        min_val + (max_val - min_val) *
        (np.exp(curve * np.linspace(0, 1, n)) - 1) / (np.exp(curve) - 1),2))
    return grid

def calc_mov_scale(games):
    mean_log_margin = np.log(
        (games["home_score"] - games["away_score"]).abs() + 1
    ).mean()
    return mean_log_margin

def cfb_home_result(home_score, away_score, overtime):
    conditions = [
        (home_score > away_score) & (overtime == 0),
        (home_score < away_score) & (overtime == 0),
        (home_score > away_score) & (overtime == 1),
        (home_score < away_score) & (overtime == 1),
        home_score == away_score
    ]
    choices = [1.00, 0.00, 0.75, 0.25, 0.50]
    return np.select(conditions, choices, default=np.nan)

def elo_season(games, K, ratings=None, scale=400, start_rating=1000,
               use_mov=True, mov_scale=1, use_home_adv=True, home_adv=0):

    ratings = {} if ratings is None else ratings.copy()
    n = len(games)

    if use_mov and (pd.isna(mov_scale) or mov_scale <= 0):
        mov_scale = 1
    if not use_home_adv:
        home_adv = 0

    elo_home_pre = np.zeros(n)
    elo_away_pre = np.zeros(n)
    elo_home_post = np.zeros(n)
    elo_away_post = np.zeros(n)
    p_home = np.zeros(n)

    for i in range(n):
        g = games.iloc[i]

        hid = str(g["home_team"])
        aid = str(g["away_team"])

        Ra = ratings.get(hid, start_rating)
        Rb = ratings.get(aid, start_rating)

        elo_home_pre[i] = Ra
        elo_away_pre[i] = Rb

        game_home_adv = 0 if g["location"] == "Neutral" else home_adv

        p = 1 / (1 + 10 ** ((Rb - Ra - game_home_adv) / scale))
        p_home[i] = p

        y = cfb_home_result(home_score=g["home_score"], away_score=g["away_score"], overtime=g["overtime"])

        if pd.isna(y):
            raise ValueError(f"Invalid NCAA FBS result coding for game_id: {g['game_id']}")

        point_diff = abs(g["home_score"] - g["away_score"])
        if g["home_score"] > g["away_score"]:
            winner_elo_diff = (Ra + game_home_adv) - Rb
        elif g["home_score"] < g["away_score"]:
            winner_elo_diff = Rb - (Ra + game_home_adv)
        else:
            winner_elo_diff = 0

        if not use_mov:
          mov_mult = 1
        elif g["home_score"] == g["away_score"]:
          mov_mult = 1
        else:
          mov_mult = np.log(point_diff + 1) * (2.2 / (0.001 * winner_elo_diff + 2.2)) / mov_scale

        delta = K * (y - p) * mov_mult

        ratings[hid] = Ra + delta
        ratings[aid] = Rb - delta

        elo_home_post[i] = ratings[hid]
        elo_away_post[i] = ratings[aid]

    games_out = games.copy()
    games_out["elo_home_pre"] = elo_home_pre
    games_out["elo_away_pre"] = elo_away_pre
    games_out["elo_home_post"] = elo_home_post
    games_out["elo_away_post"] = elo_away_post
    games_out["p_home"] = p_home
    games_out["p_away"] = 1 - p_home

    return {"games": games_out, "ratings": ratings}

def carry_over(ratings, carry=0.6):
    if len(ratings) == 0:
        return ratings

    mu = np.mean(list(ratings.values()))
    return {team: carry * rating + (1 - carry) * mu
            for team, rating in ratings.items()}

def logloss(p, y, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))

def score_elo_params(all_games, seasons_tune, K, home_adv,
                     use_mov=True, mov_scale=1, use_home_adv=True,
                     scale=400, start_rating=1000, carry=0.6,
                     initial_ratings=None):

  ratings = {} if initial_ratings is None else initial_ratings.copy()
  ll_vec = []

  if not use_home_adv:
      home_adv = 0

  for s in seasons_tune:
      g_s = (all_games[all_games["season"] == s]
              .sort_values(["gameday", "game_id"])
              .reset_index(drop=True))

      ratings = carry_over(ratings, carry=carry)
      res = elo_season(games=g_s, ratings=ratings, K=K, scale=scale,
                       start_rating=start_rating, use_mov=use_mov,
                       mov_scale=mov_scale, use_home_adv=use_home_adv,
                       home_adv=home_adv)

      games_res = res["games"]
      non_ties = games_res["home_score"] != games_res["away_score"]

      p = games_res.loc[non_ties, "p_home"].to_numpy()
      y = (games_res.loc[non_ties, "home_score"] >
            games_res.loc[non_ties, "away_score"]).astype(float).to_numpy()

      ll_vec.extend(logloss(p, y))
      ratings = res["ratings"]

  return np.nanmean(ll_vec)

from itertools import product

def tune_elo_params(all_games, seasons_tune, K_grid, home_adv_grid,
                    use_mov=True, mov_scale=1, use_home_adv=True,
                    scale=400, start_rating=1000, carry=0.6,
                    use_carry_grid=True,
                    carryover_grid=(0.35, 0.50, 0.65, 0.80),
                    initial_ratings=None):

    if not use_home_adv:
        home_adv_grid = [0]

    carry_grid = carryover_grid if use_carry_grid else [carry]
    results = []

    for K, home_adv, carry_val in product(K_grid, home_adv_grid, carry_grid):
        ll = score_elo_params(
            all_games=all_games, seasons_tune=seasons_tune,
            K=K, home_adv=home_adv,
            use_mov=use_mov, mov_scale=mov_scale,
            use_home_adv=use_home_adv, scale=scale,
            start_rating=start_rating, carry=carry_val,
            initial_ratings=initial_ratings
        )

        results.append({"K": K, "home_adv": home_adv, "carry": carry_val,
                        "logloss": ll})

    results = pd.DataFrame(results).sort_values("logloss").reset_index(drop=True)

    best = results.iloc[0]

    return {"K_best": best["K"], "home_adv_best": best["home_adv"],
            "carry_best": best["carry"], "logloss": best["logloss"],
            "results": results}

def fill_missing_overtime(games, supplement_file, refresh_data):
    """Verify source gaps against ESPN finals; never infer regulation from a score."""
    missing = games.loc[games["completed"] & games["overtime"].isna()]
    if missing.empty:
        return
    game_ids = ", ".join(missing["game_id"].astype(str))
    if not refresh_data:
        raise ValueError(f"Missing completed-game overtime metadata for game IDs {game_ids}; run online to verify ESPN finals.")

    supplement_file = Path(supplement_file)
    supplement = pd.read_csv(supplement_file)
    additions = []
    print(f"Verifying overtime metadata for {len(missing)} completed games against ESPN finals.", flush=True)
    for index, game in missing.iterrows():
        game_id = int(game["game_id"])
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary?event={game_id}"
        try:
            with urlopen(url, timeout=30) as response:
                competition = json.load(response)["header"]["competitions"][0]
            if str(competition["id"]) != str(game_id):
                raise ValueError("ESPN returned a different game ID")
            status = competition["status"]["type"]
            detail = status.get("detail", "")
            if (status.get("completed") is not True or status.get("state") != "post" or
                    re.fullmatch(r"Final(?:[ /]+(?:\d+)?OT)?", detail, re.IGNORECASE) is None):
                raise ValueError(f"ESPN has no verified final overtime status: {detail!r}")
            competitors = {team["homeAway"]: team for team in competition["competitors"]}
            for side in ["home", "away"]:
                team = competitors[side]
                if str(team["id"]) != str(int(game[f"{side}_id"])):
                    raise ValueError(f"ESPN {side} team does not match the schedule")
                if float(team["score"]) != float(game[f"{side}_points"]):
                    raise ValueError(f"ESPN {side} final score does not match the schedule")
            overtime = int("OT" in detail.upper())
        except Exception as exc:
            raise ValueError(f"Cannot verify completed game {game_id} ({game['away_team']} at {game['home_team']}): {exc}. Retry when the sources agree.") from exc
        games.loc[index, "overtime"] = overtime
        additions.append({"game_id": game_id, "overtime": overtime, "start_date": "", "source": url})

    # Keep historical kickoff corrections; new entries supply only the verified OT flag.
    supplement = pd.concat([supplement, pd.DataFrame(additions)], ignore_index=True)
    supplement = supplement.drop_duplicates("game_id", keep="last").sort_values("game_id")
    temporary = supplement_file.with_suffix(".csv.tmp")
    supplement.to_csv(temporary, index=False)
    temporary.replace(supplement_file)


def refresh_recent_results(games, supplement_file, current_season, now=None):
    """Check recent started games when the bulk schedule still has no final."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
    kickoff = pd.to_datetime(games["start_date"], utc=True, format="mixed", errors="coerce")
    pending = games.loc[(games.season == current_season) & ~games.completed &
                        (kickoff <= now) & (kickoff >= now - pd.Timedelta(days=7))]
    if pending.empty:
        return
    supplement_file = Path(supplement_file)
    supplement = pd.read_csv(supplement_file)
    additions = []
    refreshed = 0
    for index, game in pending.iterrows():
        game_id = int(game["game_id"])
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary?event={game_id}"
        try:
            with urlopen(url, timeout=30) as response:
                competition = json.load(response)["header"]["competitions"][0]
            status = competition["status"]["type"]
            if status.get("completed") is not True:
                continue
            if (str(competition["id"]) != str(game_id) or status.get("state") != "post" or
                    re.fullmatch(r"Final(?:[ /]+(?:\d+)?OT)?", status.get("detail", ""), re.IGNORECASE) is None):
                raise ValueError("unverified final game ID or status")
            competitors = {team["homeAway"]: team for team in competition["competitors"]}
            scores = {}
            for side in ["home", "away"]:
                team = competitors[side]
                if str(team["id"]) != str(int(game[f"{side}_id"])):
                    raise ValueError("final team IDs do not match the schedule")
                score = float(team["score"])
                if not np.isfinite(score) or score < 0 or score != int(score):
                    raise ValueError("invalid final score")
                scores[side] = score
        except Exception as exc:
            print(f"Could not refresh game {game_id}; retaining its source status: {exc}", flush=True)
            continue
        games.loc[index, ["home_points", "away_points", "completed", "status"]] = [scores["home"], scores["away"], True, "STATUS_FINAL"]
        for side, other in [("home", "away"), ("away", "home")]:
            if f"{side}_winner" in games.columns:
                games.loc[index, f"{side}_winner"] = scores[side] > scores[other]
        overtime = int("OT" in status["detail"].upper())
        if game_id in supplement.game_id.values:
            supplement.loc[supplement.game_id == game_id, "overtime"] = overtime
        else:
            additions.append({"game_id": game_id, "overtime": overtime, "start_date": "", "source": url})
        refreshed += 1
    if refreshed:
        supplement = pd.concat([supplement, pd.DataFrame(additions)], ignore_index=True)
        temporary = supplement_file.with_suffix(".csv.tmp")
        supplement.sort_values("game_id").to_csv(temporary, index=False)
        temporary.replace(supplement_file)
        print(f"Updated {refreshed} delayed final results from ESPN.", flush=True)


def load_games(data_dir=None, refresh_data=True, schedule_season=None):
    data_dir = ROOT / "data" if data_dir is None else Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if schedule_season is None:
        schedule_season = pd.Timestamp.now(tz="America/Chicago").year
    games = []
    for season in range(2003, schedule_season + 1):
        schedule_file = data_dir / f"cfb_schedules_{season}.parquet"
        if not schedule_file.exists() and not refresh_data:
            raise FileNotFoundError(f'Offline schedule missing: {schedule_file}')
        if not schedule_file.exists() or (refresh_data and season == schedule_season):
            urlretrieve(
                f"https://github.com/sportsdataverse/sportsdataverse-data/releases/download/cfb_schedules/cfb_schedules_{season}.parquet",
                schedule_file
            )
        games.append(pd.read_parquet(schedule_file))
    games = pd.concat(games, ignore_index=True)
    games = games[games["fbs_participant"].fillna(False) &
                  games["season_type"].isin(["regular", "postseason"]) &
                  ~games["status"].isin(["STATUS_CANCELED", "STATUS_CANCELLED"])].copy()

    # Some historical source fixtures have both a placeholder and an ESPN game ID.
    games["fixture_date"] = games["start_date"].str[:10]
    games = (games.sort_values("game_id")
             .drop_duplicates(["season", "home_team", "away_team", "fixture_date"], keep="last")
             .drop(columns="fixture_date"))

    if refresh_data:
        refresh_recent_results(games, data_dir / "cfb_overtime_supplement.csv", schedule_season)

    overtime_file = data_dir / "cfb_schedule_master.parquet"
    if not overtime_file.exists() and not refresh_data:
        raise FileNotFoundError(f'Offline overtime metadata missing: {overtime_file}')
    if not overtime_file.exists() or refresh_data:
        urlretrieve(
            "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-cfb-raw/main/cfb/cfb_schedule_master.parquet",
            overtime_file
        )
    overtime_games = pd.read_parquet(overtime_file).set_index("game_id")
    overtime = ((overtime_games["status_period"] > 4) |
                overtime_games["status_type_detail"].str.contains("OT", na=False)).astype(float)
    overtime = overtime.where(overtime_games["status_type_completed"])
    overtime_supplement = pd.read_csv(data_dir / "cfb_overtime_supplement.csv").set_index("game_id")
    games["overtime"] = games["game_id"].map(overtime).fillna(
        games["game_id"].map(overtime_supplement["overtime"]))
    games["start_date"] = games["game_id"].map(overtime_supplement["start_date"]).fillna(games["start_date"])
    fill_missing_overtime(games, data_dir / "cfb_overtime_supplement.csv", refresh_data)

    games = games.rename(columns={"home_team": "home_team_name", "away_team": "away_team_name",
                                  "home_points": "home_score", "away_points": "away_score"})
    games["home_team"] = games["home_id"].astype("int64").astype(str)
    games["away_team"] = games["away_id"].astype("int64").astype(str)
    games["game_type"] = games["season_type"].map({"regular": "REG", "postseason": "POST"})
    games["location"] = np.where(games["neutral_site"], "Neutral", "Home")
    games["kickoff"] = pd.to_datetime(games["start_date"], utc=True, format="mixed").dt.tz_convert("America/Chicago")
    games["gameday"] = games["kickoff"].dt.strftime("%Y-%m-%d")
    games["gametime"] = games["kickoff"].dt.strftime("%H:%M:%S")
    games.loc[games["start_time_tbd"].fillna(False), "gametime"] = ""
    # A live or scheduled score (including a placeholder 0-0) is not a final result.
    games.loc[~games["completed"], ["home_score", "away_score"]] = np.nan
    games = games.sort_values(["season", "gameday", "gametime", "game_id"]).reset_index(drop=True)
    assert games["game_id"].is_unique
    assert games.loc[games["completed"], ["home_score", "away_score"]].notna().all().all()

    return games


def run_expanding_k_elo(all_games, first_train_year=2003, first_eval_year=2010,
                        K_grid=None, home_adv_grid=None, scale=400,
                        start_rating=1000, carry=0.6, use_carry_grid=True,
                        carryover_grid=(0.35, 0.50, 0.65, 0.80),
                        param_tune_window=None, use_mov=True, use_home_adv=True,
                        projection_season=None, parameter_file=None):

    if K_grid is None:
        K_grid = compress_exp_grid(1, 60, n=10, curve=-1)
    if home_adv_grid is None:
        home_adv_grid = compress_exp_grid(1, 60, n=10, curve=1)
    if not use_home_adv:
        home_adv_grid = [0]

    K_grid = np.atleast_1d(K_grid)
    home_adv_grid = np.atleast_1d(home_adv_grid)
    fixed_param_mode = len(K_grid) == 1 and len(home_adv_grid) == 1 and not use_carry_grid

    seasons = sorted(all_games["season"].unique())
    eval_years = [s for s in seasons if s >= first_eval_year]

    # Include the current schedule season even if it has no completed games yet.
    if projection_season is not None and projection_season not in eval_years:
        eval_years = sorted(set(eval_years + [projection_season]))

    saved_params = pd.DataFrame()
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
    all_out = []
    param_out = []
    ratings = {}
    ratings_before_season = {}

    for yr in eval_years:
        train_years = [s for s in seasons if first_train_year <= s < yr]
        if len(train_years) == 0:
            raise ValueError(f"No training seasons available before eval year: {yr}")

        param_tune_years = (
            train_years if param_tune_window is None
            else train_years[-param_tune_window:]
        )

        train_games = all_games[all_games["season"].isin(train_years)]
        mov_scale = calc_mov_scale(train_games) if use_mov else np.nan

        if yr in saved_params.index:
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
        elif fixed_param_mode:
            K_best = K_grid[0]
            home_adv_best = home_adv_grid[0] if use_home_adv else 0
            carry_best = carry
            best_logloss = np.nan
        else:
            print(f"No saved parameters for {yr}; tuning prior seasons once...", flush=True)
            param_fit = tune_elo_params(
                all_games=all_games, seasons_tune=param_tune_years,
                K_grid=K_grid, home_adv_grid=home_adv_grid,
                use_mov=use_mov, mov_scale=mov_scale,
                use_home_adv=use_home_adv, scale=scale,
                start_rating=start_rating, carry=carry,
                use_carry_grid=use_carry_grid,
                carryover_grid=carryover_grid,
                initial_ratings=ratings_before_season.get(param_tune_years[0], {})
            )

            K_best = param_fit["K_best"]
            home_adv_best = param_fit["home_adv_best"] if use_home_adv else 0
            carry_best = param_fit["carry_best"] if use_carry_grid else carry
            best_logloss = param_fit["logloss"]

        # Bootstrap training history once; later seasons keep the prior final ratings.
        if yr == eval_years[0]:
            for s in train_years:
                season_games = (all_games[all_games["season"] == s]
                                .sort_values(["gameday", "gametime", "game_id"])
                                .reset_index(drop=True))

                ratings_before_season[s] = ratings.copy()
                ratings = carry_over(ratings, carry=carry_best)
                res = elo_season(
                    games=season_games, ratings=ratings, K=K_best,
                    scale=scale, start_rating=start_rating,
                    use_mov=use_mov, mov_scale=mov_scale,
                    use_home_adv=use_home_adv, home_adv=home_adv_best
                )

                ratings = res["ratings"]

        ratings_before_season[yr] = ratings.copy()
        ratings = carry_over(ratings, carry=carry_best)

        eval_games = (all_games[all_games["season"] == yr]
                      .sort_values(["gameday", "gametime", "game_id"])
                      .reset_index(drop=True))

        res_eval = elo_season(
            games=eval_games, ratings=ratings, K=K_best,
            scale=scale, start_rating=start_rating,
            use_mov=use_mov, mov_scale=mov_scale,
            use_home_adv=use_home_adv, home_adv=home_adv_best
        )

        ratings = res_eval["ratings"]
        games_out = res_eval["games"].copy()
        games_out["K_used"] = K_best
        games_out["home_adv_used"] = np.where(games_out["location"] == "Neutral", 0, home_adv_best)
        games_out["carry_used"] = carry_best
        games_out["mov_scale_used"] = mov_scale
        games_out["use_mov"] = use_mov
        games_out["use_home_adv"] = use_home_adv
        games_out["train_start"] = min(train_years)
        games_out["train_end"] = max(train_years)
        games_out["param_tune_start"] = min(param_tune_years)
        games_out["param_tune_end"] = max(param_tune_years)
        games_out["param_tune_n"] = len(param_tune_years)
        games_out["param_tune_window"] = param_tune_window

        all_out.append(games_out)

        param_out.append({
            "eval_year": yr,
            "train_start": min(train_years),
            "train_end": max(train_years),
            "param_tune_start": min(param_tune_years),
            "param_tune_end": max(param_tune_years),
            "param_tune_n": len(param_tune_years),
            "param_tune_window": param_tune_window,
            "K_best": K_best,
            "home_adv_best": home_adv_best,
            "carry_best": carry_best,
            "mov_scale": mov_scale,
            "use_mov": use_mov,
            "use_home_adv": use_home_adv,
            "use_carry_grid": use_carry_grid,
            "scale": scale, "start_rating": start_rating,
            "K_grid": json.dumps(np.asarray(K_grid).tolist()),
            "home_adv_grid": json.dumps(np.asarray(home_adv_grid).tolist()),
            "carry_grid": json.dumps(np.asarray(carryover_grid).tolist()),
            "logloss": best_logloss
        })

        if yr not in saved_params.index:
            new_params.append(param_out[-1])
            if parameter_file is not None:
                # Checkpoint each newly tuned season so an interrupted first run can resume.
                parameters = pd.concat([saved_params.reset_index(), pd.DataFrame(new_params)],
                                       ignore_index=True) if not saved_params.empty else pd.DataFrame(new_params)
                temporary = parameter_file.with_suffix(".csv.tmp")
                parameters.sort_values("eval_year").to_csv(temporary, index=False)
                temporary.replace(parameter_file)

    return {
        "games": pd.concat(all_out, ignore_index=True),
        "K_by_year": pd.DataFrame(param_out),
        "ratings": res_eval["ratings"],
        "current_season": eval_years[-1]
    }

def fit_spreads(historical_elo):

    spread_games = historical_elo.dropna(subset=["home_score", "away_score", "p_home"]).copy()

    spread_games = spread_games.sort_values(["season", "gameday", "gametime", "game_id"]).reset_index(drop=True)

    spread_games["home_margin"] = (spread_games["home_score"] - spread_games["away_score"])

    p = spread_games["p_home"].clip(1e-6, 1 - 1e-6)
    spread_games["logit_p_home"] = np.log(p / (1 - p))

    spread_seasons = sorted(spread_games["season"].unique())
    spread_validation = []

    for eval_season in spread_seasons[1:]:
        train = spread_games[spread_games["season"] < eval_season].copy()
        test = spread_games[spread_games["season"] == eval_season].copy()

        spread_opt = minimize_scalar(
            lambda slope: np.mean(
                np.abs(
                    train["home_margin"]
                    - slope * train["logit_p_home"]
                )
            ),
            bounds=(0, 20),
            method="bounded"
        )

        slope = spread_opt.x

        test["spread_slope_used"] = slope
        test["elo_home_margin"] = slope * test["logit_p_home"]
        test["model_home_spread"] = -(np.round(test["elo_home_margin"] * 2) / 2)
        test["spread_abs_error"] = (test["home_margin"] - test["elo_home_margin"]).abs()

        spread_validation.append(test)

    spread_validation = pd.concat(spread_validation, ignore_index=True)


    spread_performance_by_year = (spread_validation.groupby("season").agg(games=("game_id", "size"), spread_slope_used=("spread_slope_used", "first"),
                                                                          spread_mae=("spread_abs_error", "mean")).reset_index())


    production_spread_opt = minimize_scalar(
        lambda slope: np.mean(
            np.abs(
                spread_games["home_margin"]
                - slope * spread_games["logit_p_home"]
            )
        ),
        bounds=(0, 20),
        method="bounded"
    )

    spread_slope = production_spread_opt.x


    return spread_slope, spread_validation, spread_performance_by_year


def build_projections(games, cfb_elo, spread_slope, as_of_date=None):
    current_season = cfb_elo["current_season"]
    params = cfb_elo["K_by_year"].set_index("eval_year").loc[current_season]

    # Use the final expanding fit's current Elo state directly.
    ratings = cfb_elo["ratings"].copy()

    current_teams = pd.unique(
        games.loc[
            games["season"] == current_season,
            ["home_team", "away_team"]
        ].values.ravel()
    )

    # As in the original model, a newly appearing team starts at 1000.
    ratings = {team: ratings.get(team, 1000) for team in current_teams}

    current_fbs_teams = pd.unique(np.concatenate([
        games.loc[(games["season"] == current_season) & (games["home_division"] == "fbs"), "home_team"],
        games.loc[(games["season"] == current_season) & (games["away_division"] == "fbs"), "away_team"]
    ]))
    team_names = pd.concat([
        games[["home_team", "home_team_name"]].rename(columns={"home_team": "team", "home_team_name": "team_name"}),
        games[["away_team", "away_team_name"]].rename(columns={"away_team": "team", "away_team_name": "team_name"})
    ]).drop_duplicates("team", keep="last").set_index("team")["team_name"]

    current_ratings = (
        pd.Series(ratings, name="elo")
        .rename_axis("team")
        .reset_index()
        .query("team in @current_fbs_teams")
        .sort_values("elo", ascending=False)
        .reset_index(drop=True)
    )
    current_ratings["team_name"] = current_ratings["team"].map(team_names)
    current_ratings.insert(0, "rank", range(1, len(current_ratings) + 1))


    as_of_date = pd.Timestamp.now(tz="America/Chicago").date() if as_of_date is None else pd.Timestamp(as_of_date).date()
    rank_map = current_ratings.set_index("team")["rank"]

    future_projections = games[
        (games["season"] == current_season) &
        games["game_type"].isin(["REG", "POST"]) &
        games["home_score"].isna() &
        games["away_score"].isna() &
        (games["status"].isna() | games["status"].eq("STATUS_SCHEDULED")) &
        (pd.to_datetime(games["gameday"]).dt.date >= as_of_date)
    ].copy()

    future_projections["home_elo"] = future_projections["home_team"].map(ratings).fillna(1000)
    future_projections["away_elo"] = future_projections["away_team"].map(ratings).fillna(1000)
    future_projections["home_rank"] = future_projections["home_team"].map(rank_map)
    future_projections["away_rank"] = future_projections["away_team"].map(rank_map)

    future_projections["game_home_adv"] = np.where(future_projections["location"] == "Neutral", 0, params["home_adv_best"])
    future_projections["p_home"] = 1 / (1 + 10 ** ((future_projections["away_elo"] - future_projections["home_elo"] - future_projections["game_home_adv"]) / 400))
    future_projections["p_away"] = 1 - future_projections["p_home"]

    projection_p = future_projections["p_home"].clip(1e-6, 1 - 1e-6)
    future_projections["elo_home_margin"] = spread_slope * np.log(projection_p / (1 - projection_p))
    future_projections["model_home_spread"] = -(np.round(future_projections["elo_home_margin"] * 2) / 2)
    future_projections["as_of_date"] = as_of_date

    future_projections = future_projections[["as_of_date", "game_id", "game_type", "week", "gameday", "gametime", "away_team", "home_team", "away_team_name", "home_team_name", "away_rank", "home_rank", "away_elo", "home_elo", "p_away", "p_home", "elo_home_margin", "model_home_spread"]
                                            ].sort_values(["gameday", "gametime"]).reset_index(drop=True)

    daily_projections = future_projections[pd.to_datetime(future_projections["gameday"]).dt.date == as_of_date].reset_index(drop=True)

    # College postseason week numbers restart, so match both phase and week.
    current_week = future_projections.iloc[0]["week"] if not future_projections.empty else None
    current_game_type = future_projections.iloc[0]["game_type"] if not future_projections.empty else None
    weekly_projections = future_projections[
        (future_projections["week"] == current_week) &
        (future_projections["game_type"] == current_game_type)
    ].reset_index(drop=True)

    daily_rankings = current_ratings.assign(as_of_date=as_of_date)


    return {"current_elo_rankings.csv": current_ratings,
            "daily_projections.csv": daily_projections,
            "weekly_projections.csv": weekly_projections,
            "future_projections.csv": future_projections}



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
    outputs = portfolio_outputs(games, cfb_elo, outputs, spread_validation,
                                output_dir / "historical_predictions.csv", spread_slope=spread_slope)
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
