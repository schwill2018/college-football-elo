import ast
import copy
import hashlib
import json
from pathlib import Path

original_path = Path(r'C:\Users\schne\nfl_elo_project\NFL_elo_v3.ipynb')
original = json.loads(original_path.read_text(encoding='utf-8'))
notebook = copy.deepcopy(original)

def source(index, value=None):
    if value is None:
        return ''.join(notebook['cells'][index]['source'])
    notebook['cells'][index]['source'] = value.splitlines(keepends=True)

for i, cell in enumerate(notebook['cells']):
    source(i, source(i).replace('NFL', 'NCAA FBS').replace('nfl_home_result', 'cfb_home_result')
           .replace('nfl_elo', 'cfb_elo'))
    if cell['cell_type'] == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []
    # Remove stale Colab execution/render metadata; retain cell identities.
    cell['metadata'] = {k: v for k, v in cell.get('metadata', {}).items() if k == 'id'}

source(1, '# pip install numpy pandas pyarrow scikit-learn scipy ipython tzdata\n')
source(2, '''## 1. Setup and NCAA FBS Data

Load public [SportsDataverse college schedules](https://py.sportsdataverse.org/docs/0.1.5/cfb/reference/loaders/schedule)
and the [ESPN schedule master](https://github.com/sportsdataverse/cfbfastR-cfb-raw/blob/main/cfb/cfb_schedule_master.parquet).
The loading cell maps these into the original NFL column names. No API key is needed.
Run from this notebook's directory and keep the accompanying `data/cfb_overtime_supplement.csv`.
Files are cached in `data/`; the current season and overtime metadata refresh by default.
Set `refresh_data = False` to replay the included snapshot offline.

Retain regular-season and postseason games involving at least one season-specific FBS team,
including conference championships, bowls, and CFP games. January games retain their source
football season. FCS opponents have individual ratings, start at the same 1000 as every new
team, and carry over normally; FCS-only games are excluded. This is the closest transfer of the
original equal-start methodology, not a separately calibrated FCS-strength model. Rankings are
FBS-only; projections retain FCS opponents. Team IDs are rating keys, with school names alongside.

The original 0.75/0.25 overtime result weights are retained, including multi-overtime games.
Overtime is read from ESPN final-period/status metadata. A 17-row, source-linked supplement
covers historical gaps; missing completed-game overtime metadata raises an error rather than
silently treating a game as regulation. Duplicate source fixtures are counted only once.

All original parameter grids and training/evaluation years are retained. They are inherited
search choices, not established college-football parameter values. Full tuning is computationally
expensive because the original row-by-row calculations and repeated fitting are preserved.
''')
source(3, '''import numpy as np 
import pandas as pd
from pathlib import Path
from urllib.request import urlretrieve
from sklearn.metrics import accuracy_score, log_loss

#Load college games and map them to the original schedule columns.
refresh_data = True
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
schedule_season = pd.Timestamp.now(tz="America/Chicago").year
games = []
for season in range(2003, schedule_season + 1):
    schedule_file = data_dir / f"cfb_schedules_{season}.parquet"
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

overtime_file = data_dir / "cfb_schedule_master.parquet"
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
if games.loc[games["completed"], "overtime"].isna().any():
    raise ValueError("Missing completed-game overtime metadata; refresh the schedule master before fitting.")

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
print(games.columns)
print(games.shape)
print(games.head())
''')
for index in [28, 45, 56]:
    source(index, source(index).replace('["REG", "WC", "DIV", "CON", "SB"]', '["REG", "POST"]'))
source(18, '''Compare `test_score` above with the unchanged naive 50/50 baseline:
$-\\log(0.5)=0.6931$. The NFL notebook's recorded 0.6503 is not a college result.
''')
source(31, source(31).replace('#Calculate out-of-sample performance of 2023-2026',
                              '#Calculate out-of-sample performance for all evaluation seasons'))
source(36, source(36).replace('home_adv_grid = display(np.round(compress_exp_grid(25, 60, n=8, curve=-1))) #',
                              'home_adv_grid = np.round(compress_exp_grid(25, 60, n=8, curve=-1)), #'))
source(53, '''## 9. Current Elo Ratings

Use the current ratings returned directly by the final expanding Elo fit, following the same
2003-forward training/replay path as the fitted model. Keep individual non-FBS opponent ratings
for projections; publish rankings only for teams classified as FBS in the current season.
''')
source(54, source(54).replace('# Match the production script: a newly appearing team starts at 1000.',
                            '# As in the original model, a newly appearing team starts at 1000.'))
source(54, source(54).replace('current_ratings = (', '''current_fbs_teams = pd.unique(np.concatenate([
    games.loc[(games["season"] == current_season) & (games["home_division"] == "fbs"), "home_team"],
    games.loc[(games["season"] == current_season) & (games["away_division"] == "fbs"), "away_team"]
]))
team_names = pd.concat([
    games[["home_team", "home_team_name"]].rename(columns={"home_team": "team", "home_team_name": "team_name"}),
    games[["away_team", "away_team_name"]].rename(columns={"away_team": "team", "away_team_name": "team_name"})
]).drop_duplicates("team", keep="last").set_index("team")["team_name"]

current_ratings = (''').replace('.sort_values("elo", ascending=False)',
                              '.query("team in @current_fbs_teams")\n    .sort_values("elo", ascending=False)'))
source(54, source(54).replace('current_ratings.insert(0,',
                            'current_ratings["team_name"] = current_ratings["team"].map(team_names)\ncurrent_ratings.insert(0,'))
source(56, source(56).replace('games["away_score"].isna()\n', '''games["away_score"].isna() &
    (games["status"].isna() | games["status"].eq("STATUS_SCHEDULED")) &
    (pd.to_datetime(games["gameday"]).dt.date >= as_of_date)
''').replace('.map(ratings)', '.map(ratings).fillna(1000)')
       .replace('"as_of_date", "week", "gameday"', '"as_of_date", "game_type", "week", "gameday"')
       .replace('"away_team", "home_team", "away_rank"',
                '"away_team", "home_team", "away_team_name", "home_team_name", "away_rank"')
       .replace('current_week = future_projections.iloc[0]["week"]\nweekly_projections = future_projections[future_projections["week"] == current_week].reset_index(drop=True)',
'''# College postseason week numbers restart, so match both phase and week.
current_week = future_projections.iloc[0]["week"] if not future_projections.empty else None
current_game_type = future_projections.iloc[0]["game_type"] if not future_projections.empty else None
weekly_projections = future_projections[
    (future_projections["week"] == current_week) &
    (future_projections["game_type"] == current_game_type)
].reset_index(drop=True)'''))
source(58, '')
source(59, '''### Adaptation and verification notes

Cell count, section order, modeling functions, initial 1000, Elo scale 400, MOV adjustment,
offseason regression, tuning grids, expanding windows, binary outcome evaluation, and spread
fit are preserved. The only function edit is renaming `nfl_home_result` to `cfb_home_result`
(and its error message). Data loading, phase filters, FBS ranking membership, and name labels
are college adaptations. OT weighting is inherited by request; it is not an NCAA requirement.

Two original execution defects are repaired here: `home_adv_grid=display(...)` lacked a comma
and returned `None`, and selecting an upcoming week failed when no games remained. A stale
NFL score and stale replay-year description are replaced with college-appropriate explanations.

See `VALIDATION.md` for the exact tests and their limits. Saved outputs from the NFL notebook
were cleared; they must not be interpreted as NCAA results. The delivered full tuning grids
are unchanged; reduced-grid verification runs do not establish their optimality on college data.
''')
notebook['metadata']['kernelspec']['display_name'] = 'Python 3 (sports)'
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        ast.parse(''.join(cell['source']), filename=f'cell_{i}')
Path('CFB_elo_v3.ipynb').write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
Path('.validation/original_sha256.txt').write_text(hashlib.sha256(original_path.read_bytes()).hexdigest() + '\n')
print(f'Created notebook: {len(notebook["cells"])} cells; all code cells parse.')
