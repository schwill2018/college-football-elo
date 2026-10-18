import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
r = json.loads((root / '.validation/report.json').read_text())
d = r['data']
out = r['results']
text = f'''# College notebook adaptation and verification

Verified October 17, 2026, using Python 3.12.14 and the versions in `requirements.txt`.

## Run

Open `CFB_elo_v3.ipynb` from this directory and run its cells in order with a Python
kernel containing the listed dependencies (`pip install -r requirements.txt`). Keep
`data/cfb_overtime_supplement.csv` beside the notebook in the included `data/` folder.
The remaining data files are cached public downloads. The loading cell refreshes
the current season and ESPN master by default; set `refresh_data = False` to replay
the included snapshot offline. No API key is required.

The full original tuning grids remain in the delivered notebook. Running all of
them can take substantially longer than the verification run described below.

## Fidelity to the reference

The original `C:/Users/schne/nfl_elo_project/NFL_elo_v3.ipynb` was read only. Its SHA-256
is recorded in `.validation/original_sha256.txt` and was checked after adaptation.

All 60 cells retain their types and order, including the original blank code cell.
There are 37 code cells. Modeling cells 7, 11, 15, 20, and 25 (zero-based indices)
have identical Python syntax trees to the original after the sport-specific
`nfl_home_result` rename and error-message replacement. These preserve the Elo
formula, normalized MOV multiplier, overtime weights, zero-sum updates, offseason
regression, parameter search, rolling five-season tuning, expanding replay,
evaluation, and the treatment of an upcoming season with no completed games.
The 2003 training start, 2008 evaluation start, grids, scale 400, starting rating
1000, spread bounds, rounding, and output workflow are retained.

Changes are limited to data loading and column mapping, REG/POST filters, college
team membership and display labels, phase-aware upcoming weeks, and explanatory
markdown. Two execution defects in the original were repaired in the copy:
`home_adv_grid=display(...)` lacked a comma and would return `None`; selecting the
first upcoming game also failed when no games remained. The former now passes
the numeric grid shown in that original call. Stale NFL-specific recorded results
and replay-year prose were removed. Original NFL outputs were cleared.

## College transfer choices

These are fidelity choices for this adaptation, not NCAA modeling requirements:

- Keep regular and postseason games with at least one team classified as FBS in
  that source season. FCS opponents retain separate IDs and ratings; FCS-only
  games are excluded. Newly appearing teams start at 1000, including FCS teams,
  and returning teams keep the original carryover rule. No division offset or
  new roster/transfer model was introduced. FCS ratings therefore reflect only
  games in this FBS-involving history.
- Rate and project using string team IDs, and show school names alongside them.
  Current published rankings include only FBS teams; non-FBS opponents remain in
  the rating dictionary for projections and have no FBS rank.
- Retain 0.75/0.25 OT result weights, even though college OT rules differ. Multiple
  overtime periods map to the original binary overtime flag. Binary winner
  evaluation and tie exclusion are unchanged.
- Use source football-season labels for January bowls/playoffs, neutral-site
  flags for home advantage, and America/Chicago dates for daily projections.
  Match both phase and week because postseason week numbers restart.
- Exclude canceled fixtures, discard scores until games are completed, and remove
  duplicate school matchups on the same source date. This snapshot has 12 such
  duplicate rows and one canceled row. Seventeen source-linked supplementary
  records cover missing OT metadata and incorrect placeholder kickoff times;
  four Hawaii games retain the source's unique placeholder game IDs. Those four
  regulation outcomes were checked against the school's official box scores.
  Unknown completed-game OT metadata raises an error before fitting.

The parameter grids are inherited search choices, not previously validated
college-football values. No new grid recommendation is implied.

## Verification performed

- Notebook schema validated with `nbformat`; every code cell compiled.
- Exact original/adapted per-game equality on regulation wins/losses, ties, OT,
  neutral sites, new teams, and MOV/home-field switches.
- Exact original/adapted expanding-window equality for fixed and tuned modes,
  carryover, a rolling tuning window, and an empty projection season.
- Every code cell executed in order against the complete real 2003–2026 snapshot:
  {d['scheduled_games']:,} retained fixtures, {d['completed_games']:,} completed,
  {d['ot_games']:,} OT games, and {d['fbs_ranked_teams']} current FBS rankings.
  The source's latest completed kickoff was {d['source_latest_completed_kickoff']}.
- Live refresh separately executed successfully. The sequential verification
  replayed those downloaded files offline.
- Cell 22 ran its entire original 64-combination grid. For the three expensive
  expanding calls only, the verification runner used two K candidates, home
  advantage 25, and carry 0.5. Every season, game, training/evaluation year, and
  five-season window was retained. K was [45, 60] in cell 28 and [40, 60] in cells
  36 and 45. The delivered notebook's grids were never reduced.
- Finite Elo/probabilities, zero-sum updates, full rating mean 1000, strictly prior
  tuning years, zero neutral-site home advantage, FBS-only rankings, FCS opponent
  projections, regular/postseason week separation, missing-OT failure, empty
  upcoming schedules, and all four CSV exports passed.

The reduced-grid final pass had accuracy {out['accuracy']:.6f}, log loss
{out['logloss']:.6f}, prospective spread MAE {out['spread_mae']:.6f}, and production
spread slope {out['production_spread_slope']:.6f}. These are verification results,
not the results of the delivered full-grid search. The full-grid search was not
completed; its selected parameters, rankings, probabilities, and spreads can differ.
The notebook is intentionally delivered without saved reduced-grid outputs.

The sequential verification took {r['seconds']:.1f} seconds. Reproduce it with
`python .validation/verify_notebook.py`; loader edge cases use
`python .validation/check_data_edges.py`. The raw report, execution log, and
verification CSVs are in `.validation/`.

## Data sources

- [SportsDataverse schedule schema](https://py.sportsdataverse.org/docs/0.1.5/cfb/reference/loaders/schedule)
- [Public schedule release](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/cfb_schedules)
- [ESPN schedule master](https://github.com/sportsdataverse/cfbfastR-cfb-raw/blob/main/cfb/cfb_schedule_master.parquet)
- Supplement source URLs are recorded per game in `data/cfb_overtime_supplement.csv`.
- [Official Hawaii 2006 schedule](https://hawaiiathletics.com/sports/football/schedule/2006)

Source classifications, schedule completeness, and update timing remain properties
of the supplied data. Team names are labels; modeling keys are IDs.
'''
(root / 'VALIDATION.md').write_text(text, encoding='utf-8')
manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root / 'data').iterdir()) if p.is_file()}
(root / '.validation/data_sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Wrote VALIDATION.md and data snapshot hashes.')
