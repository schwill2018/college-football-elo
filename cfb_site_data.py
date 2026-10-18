"""Portfolio exports and recorded forecasts, separate from the Elo calculations."""
from pathlib import Path

import numpy as np
import pandas as pd


def portfolio_outputs(games, fit, outputs, spread_validation, archive_file, run_timestamp=None, spread_slope=None):
    now = pd.Timestamp.now(tz="America/Chicago") if run_timestamp is None else pd.Timestamp(run_timestamp)
    now = now.tz_localize("America/Chicago") if now.tzinfo is None else now.tz_convert("America/Chicago")
    games = games.copy()
    games["game_id"] = games.game_id.astype(str)
    history = fit["games"].copy()
    history["game_id"] = history.game_id.astype(str)
    history = history.sort_values(["kickoff", "game_id"])
    season = fit["current_season"]
    schedule = games.loc[games.season == season].sort_values(["kickoff", "game_id"])
    unfinished = schedule.loc[~schedule.completed & (schedule.kickoff >= now.normalize() - pd.Timedelta(days=7))]
    active = unfinished.iloc[0] if not unfinished.empty else schedule.iloc[-1]
    phase, week = active["game_type"], active["week"]
    weekly = schedule.loc[(schedule.game_type == phase) & (schedule.week == week)].copy()

    rankings = outputs["current_elo_rankings.csv"].copy()
    current_history = history.loc[history.season == season]
    start_ratings, changes = {}, {}
    for game in current_history.itertuples():
        for side in ["home", "away"]:
            team = getattr(game, f"{side}_team")
            start_ratings.setdefault(team, getattr(game, f"elo_{side}_pre"))
            changes[team] = getattr(game, f"elo_{side}_post") - getattr(game, f"elo_{side}_pre")
    for game in current_history.loc[current_history.kickoff < weekly.kickoff.min()].itertuples():
        for side in ["home", "away"]:
            start_ratings[getattr(game, f"{side}_team")] = getattr(game, f"elo_{side}_post")
    rankings["baseline_elo"] = rankings.team.map(start_ratings).fillna(rankings.elo)
    baseline = rankings.sort_values("baseline_elo", ascending=False, kind="stable")
    baseline_rank = {team: index + 1 for index, team in enumerate(baseline.team)}
    rankings["elo_delta"] = rankings.team.map(changes).fillna(0)
    rankings["rank_delta"] = rankings.team.map(baseline_rank) - rankings["rank"]
    conferences = pd.concat([
        schedule[[f"{side}_team", f"{side}_conference"]].rename(columns={f"{side}_team": "team", f"{side}_conference": "conference"})
        for side in ["home", "away"]
    ]).drop_duplicates("team", keep="last").set_index("team").conference
    rankings["conference"] = rankings.team.map(conferences).fillna("Independent")
    rankings["as_of_date"] = now.date().isoformat()
    rankings["comparison_season"], rankings["comparison_game_type"], rankings["comparison_week"] = season, phase, week
    outputs["current_elo_rankings.csv"] = rankings.drop(columns="baseline_elo")
    ranks = rankings.set_index("team")["rank"]

    future = outputs["future_projections.csv"].copy()
    future["game_id"] = future.game_id.astype(str)
    keyed_schedule = games.set_index("game_id")
    future["season"] = future.game_id.map(keyed_schedule.season)
    future["kickoff"] = future.game_id.map(keyed_schedule.kickoff)
    known_time = ~future.game_id.map(keyed_schedule.start_time_tbd).fillna(True).astype(bool)
    recorded = future.loc[known_time & (future.kickoff > now)].copy()
    recorded["prediction_timestamp"] = now.isoformat()
    archive_file = Path(archive_file)
    if archive_file.exists():
        archive = pd.read_csv(archive_file, dtype={"game_id": str, "home_team": str, "away_team": str}, float_precision="round_trip")
        if archive.game_id.duplicated().any():
            raise ValueError("Duplicate game IDs in the pregame forecast archive.")
        made_at = pd.to_datetime(archive.prediction_timestamp, utc=True, errors="coerce")
        kickoff = pd.to_datetime(archive.kickoff, utc=True, errors="coerce")
        if (made_at.isna() | kickoff.isna() | (made_at >= kickoff)).any():
            raise ValueError("The pregame archive contains an invalid prediction timestamp.")
        archive = pd.concat([archive.loc[~archive.game_id.isin(recorded.game_id)], recorded], ignore_index=True)
    else:
        archive = recorded
    outputs["historical_predictions.csv"] = archive.sort_values(["gameday", "game_id"]).reset_index(drop=True)

    # Historical reconstructions never count as recorded pregame forecasts.
    past = history.set_index("game_id")
    prior_spreads = spread_validation.copy()
    prior_spreads["game_id"] = prior_spreads.game_id.astype(str)
    prior_spreads = prior_spreads.set_index("game_id")
    saved = archive.set_index("game_id")
    for column in ["home_elo", "away_elo", "p_home", "p_away", "model_home_spread"]:
        weekly[column] = weekly.game_id.map(saved[column])
    weekly["prediction_source"] = np.where(weekly.game_id.isin(saved.index), "Recorded pregame", "Unavailable")
    reconstruct = weekly.completed & ~weekly.game_id.isin(saved.index)
    for column, source in [("home_elo", "elo_home_pre"), ("away_elo", "elo_away_pre"), ("p_home", "p_home"), ("p_away", "p_away")]:
        weekly.loc[reconstruct, column] = weekly.loc[reconstruct, "game_id"].map(past[source])
    weekly.loc[reconstruct, "model_home_spread"] = weekly.loc[reconstruct, "game_id"].map(prior_spreads.model_home_spread)
    weekly.loc[reconstruct, "prediction_source"] = "Historical model"

    # Display estimates even when recording began after kickoff. Reconstruct
    # ratings before that kickoff rather than substituting today's postgame Elo.
    estimate = ~weekly.completed & ~weekly.game_id.isin(saved.index)
    if estimate.any():
        opening, updates = {}, []
        for game in current_history.itertuples():
            for side in ["home", "away"]:
                team = getattr(game, f"{side}_team")
                opening.setdefault(team, getattr(game, f"elo_{side}_pre"))
                updates.append((game.kickoff, team, getattr(game, f"elo_{side}_post")))
        params = fit["K_by_year"].set_index("eval_year").loc[season]
        seasonal_spreads = spread_validation.loc[spread_validation.season == season]
        slope = (seasonal_spreads.spread_slope_used.iloc[0]
                 if not seasonal_spreads.empty and "spread_slope_used" in seasonal_spreads else spread_slope)
        if slope is None:
            raise ValueError("A spread coefficient is required for unrecorded game estimates.")
        for index, game in weekly.loc[estimate].iterrows():
            pregame = {}
            for side in ["home", "away"]:
                team = game[f"{side}_team"]
                earlier = [rating for kickoff, changed_team, rating in updates if changed_team == team and kickoff < game.kickoff]
                pregame[side] = earlier[-1] if earlier else opening.get(team, fit["ratings"].get(team, 1000))
                weekly.loc[index, f"{side}_elo"] = pregame[side]
            advantage = 0 if game.get("location") == "Neutral" else params.home_adv_best
            p = 1 / (1 + 10 ** ((pregame["away"] - pregame["home"] - advantage) / 400))
            weekly.loc[index, ["p_home", "p_away"]] = [p, 1 - p]
            protected = np.clip(p, 1e-6, 1 - 1e-6)
            weekly.loc[index, "model_home_spread"] = -(np.round(slope * np.log(protected / (1 - protected)) * 2) / 2)
            weekly.loc[index, "prediction_source"] = "Model estimate"
    weekly["away_rank"], weekly["home_rank"] = weekly.away_team.map(ranks), weekly.home_team.map(ranks)
    weekly["outcome"] = "TBD"
    weekly.loc[weekly.completed & (weekly.home_score > weekly.away_score), "outcome"] = weekly.home_team_name
    weekly.loc[weekly.completed & (weekly.home_score < weekly.away_score), "outcome"] = weekly.away_team_name
    weekly.loc[weekly.completed & (weekly.home_score == weekly.away_score), "outcome"] = "TIE"
    weekly["as_of_date"] = now.date().isoformat()
    weekly_columns = ["as_of_date", "season", "game_id", "game_type", "week", "gameday", "gametime",
                      "away_team", "home_team", "away_team_name", "home_team_name", "away_rank", "home_rank",
                      "away_elo", "home_elo", "p_away", "p_home", "model_home_spread", "outcome", "prediction_source"]
    outputs["weekly_projections.csv"] = weekly[weekly_columns].reset_index(drop=True)

    metadata = ["season", "game_type", "week", "game_id", "gameday", "home_team", "away_team",
                "home_team_name", "away_team_name", "home_conference", "away_conference", "home_division", "away_division"]
    historical = history.loc[(history.season < season) & (history.home_score != history.away_score), metadata + ["p_home", "home_score", "away_score"]].copy()
    historical["evaluation_type"] = "historical"
    deployed = schedule.loc[schedule.completed & (schedule.home_score != schedule.away_score), metadata + ["home_score", "away_score"]].merge(
        archive[["game_id", "p_home"]], on="game_id", how="inner", validate="one_to_one")
    deployed["evaluation_type"] = "deployed"
    performance = pd.concat([historical, deployed], ignore_index=True)
    performance["actual_home_win"] = (performance.home_score > performance.away_score).astype(int)
    performance = performance[metadata + ["p_home", "actual_home_win", "evaluation_type"]]
    if performance.duplicated(["evaluation_type", "game_id"]).any() or not performance.p_home.between(0, 1).all():
        raise ValueError("Invalid game-level performance export.")
    outputs["performance_games.csv"] = performance.sort_values(["season", "gameday", "game_id"]).reset_index(drop=True)
    return outputs
