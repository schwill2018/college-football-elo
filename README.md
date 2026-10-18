# College FBS Elo

`cfb_elo.py` is the standalone version of the final saved fit in
`CFB_elo_v3.ipynb` (cell 45, zero-based). The notebook remains unchanged.

```powershell
python -m pip install -r requirements.txt
python cfb_elo.py
```

Use `python cfb_elo.py --offline` to reuse the downloaded data in `data/`.
Paths are relative to the script, so it can be launched from another directory.

## Final notebook settings

- Training begins in 2005; evaluation begins in 2010.
- Five prior seasons for parameter tuning; expanding prior history for MOV scale.
- K grid: `[47, 50, 53, 56]`.
- Home-advantage grid: `[67, 70, 73, 76]`.
- Carryover grid: `[0.85, 0.90, 0.95]`.
- Starting rating 1000; Elo scale 400; notebook MOV, home advantage, overtime,
  and offseason regression formulas preserved.
- FBS rankings only; games involving an FBS participant and individual non-FBS
  opponent ratings retained, exactly as in the notebook.

The saved 2026 selection is K **56**, home advantage **67**, carryover **0.95**,
and MOV scale **2.636753264943746**. These are selections within the chosen
grid, not claims of a global optimum. The script preserves that grid.

## Seasonal Elo cache

`elo_parameters.csv` contains all 17 saved selections for 2010–2026. They were
extracted from the notebook's full-precision embedded table, not its rounded
display. `.validation/production_cache_source.json` records provenance.

On each run, the script checks the cache, reuses saved parameters, and replays
completed games to update ratings. Existing seasons do not run the tuning grid
and a fully cached run does not rewrite the cache. A missing season is tuned
once on prior seasons and checkpointed to the CSV. Incompatible settings or a
malformed cache raise an error rather than silently refitting it.

GitHub Actions checks out this CSV with the script and commits newly added rows
after a successful model run. Back up the cache before deliberately removing
rows to retune; changing settings requires an explicit compatible cache rebuild.

## Automatic updates

`.github/workflows/cfb_elo.yml` runs daily at 08:00 UTC (3 a.m. Chicago daylight
time / 2 a.m. standard time) and supports manual runs. It installs the Python
dependencies, runs the cache tests, updates the model, and commits the output
CSVs and seasonal cache to the default branch. Overlapping runs are serialized;
failed tests or model runs do not commit outputs.

The bulk schedule and overtime metadata can update at different times. If a
completed game's overtime flag is missing, online runs check its individual ESPN
summary and require a final result, matching game/team IDs, matching scores, and
an explicit Final/OT status. Verified flags are saved in
`data/cfb_overtime_supplement.csv` and committed by the workflow for future runs.
Unverified results still stop the run; offline runs never call ESPN.

Online runs also check recent started games that the bulk schedule still marks
unfinished. An individual ESPN final with matching team IDs supplies the final
scores and overtime flag, so a delayed bulk update does not leave a finished
game displayed as TBD. In-progress or unverified responses remain unfinished.

After pushing the workflow to the default branch, open the repository's
**Actions** tab, select **Update College Elo**, and choose **Run workflow** to
verify the first update. The workflow requests `contents: write` for its built-in
GitHub token; no custom API key or secret is required for this data source.

## Spreads and outputs

Spread fitting retains the college notebook's **MAE** objective with expanding
prior seasons for historical validation. The future-projection coefficient uses
all completed model history. These inexpensive scalar fits run each time; the
expensive Elo grid search is cached. NFL's five-season RMSE method is not substituted.

The script writes the existing four ranking/projection CSVs, plus
`cfb_elo_history.csv`, `performance_by_year.csv`,
`spread_performance_by_year.csv`, `historical_predictions.csv`, and
`performance_games.csv`. Projections include `game_id` for stable identification.

`cfb_site_data.py` prepares the portfolio exports without changing the model's
calculations or seasonal parameter cache:

- Rankings include conference, last-game Elo change, and rank movement since the
  start of the current regular/postseason week.
- The weekly export contains the full week, including results and forecasts.
  Recorded pregame forecasts freeze at kickoff. An earlier completed game with
  no recorded forecast uses its historical pregame Elo/probability and the
  prospectively validated seasonal spread fit, labelled `Historical model`.
  Started games with no saved forecast or final result show a `Model estimate`
  reconstructed from ratings before kickoff. This estimate is for display only
  and never enters the recorded-prediction archive or deployed performance.
- The pregame archive records only future kickoffs with known start times;
  existing records cannot be replaced after kickoff.
- Performance includes game-level results and each team's conference/division
  in that season. Prior seasons are `historical`; the current season includes
  only completed games with forecasts actually recorded before kickoff as
  `deployed`. Historical reconstructions never count as recorded predictions.

The workflow saves both the archive and performance export after each run. The
portfolio's updater consumes rankings, weekly games, and performance CSVs; page
sorting, pagination, and the top-25/top-50 display filter do not limit ratings.

## Verification

```powershell
python -m unittest discover -s tests -v
python .validation/verify_production.py
```

The notebook snapshot comparison is specific to the original saved data and
metrics; refreshed source files and the expanded weekly export differ by design.
The unit tests cover cache reuse, overtime verification, pregame forecast
freezing, recorded-performance eligibility, FCS opponents, and postseason weeks.
