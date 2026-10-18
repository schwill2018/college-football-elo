"""Verify fidelity and execute every code cell using reduced tuning grids.

The delivered notebook is never edited by this runner. All seasons and games
are retained; only the three expensive expanding-window calls use two K
candidates, one home-advantage candidate and one carry candidate.
"""
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell

ROOT = Path(__file__).resolve().parents[1]
original_path = Path(r'C:\Users\schne\nfl_elo_project\NFL_elo_v3.ipynb')
original = json.loads(original_path.read_text(encoding='utf-8'))
adapted = json.loads((ROOT / 'CFB_elo_v3.ipynb').read_text(encoding='utf-8'))
report = {'checks': [], 'cell_execution': []}

def check(label, condition):
    assert condition, label
    report['checks'].append(label)
    print('PASS', label, flush=True)

def cell_source(nb, i):
    return ''.join(nb['cells'][i]['source'])

check('Original notebook SHA-256 unchanged', hashlib.sha256(original_path.read_bytes()).hexdigest() ==
      (ROOT / '.validation/original_sha256.txt').read_text().strip())
check('Same 60 cell types in the same order', [c['cell_type'] for c in original['cells']] ==
      [c['cell_type'] for c in adapted['cells']] and len(adapted['cells']) == 60)
for i in [7, 11, 15, 20, 25]:
    left = cell_source(original, i).replace('nfl_home_result', 'cfb_home_result').replace('NFL', 'NCAA FBS')
    right = cell_source(adapted, i)
    check(f'Modeling cell {i} AST matches the original', ast.dump(ast.parse(left)) == ast.dump(ast.parse(right)))
for i, c in enumerate(adapted['cells']):
    if c['cell_type'] == 'code':
        compile(cell_source(adapted, i), f'cell_{i}', 'exec')
check('Every code cell compiles', True)

# Pointwise equality against the actual original functions, including ordinary
# wins, losses, ties, OT, multi-OT collapsed to a binary flag, neutral sites,
# a returning opponent and a newly appearing program.
old, new = {}, {}
for i in [7, 11, 15, 20, 25]:
    exec(cell_source(original, i), old)
    exec(cell_source(adapted, i), new)
fixture = pd.DataFrame([
    [2021, '2021-09-01', '18:00', 1, '1', '2', 28, 21, 0, 'Home'],
    [2021, '2021-09-02', '18:00', 2, '2', '3', 17, 20, 1, 'Neutral'],
    [2021, '2021-09-03', '18:00', 3, '3', '1', 14, 14, 0, 'Home'],
    [2022, '2022-09-01', '18:00', 4, '1', '3', 31, 28, 1, 'Home'],
    [2022, '2022-09-02', '18:00', 5, '4', '2', 0, 35, 0, 'Neutral'],
    [2023, '2023-09-01', '18:00', 6, '2', '1', 17, 10, 0, 'Home'],
    [2023, '2023-09-02', '18:00', 7, '3', '4', 7, 24, 0, 'Home'],
], columns=['season', 'gameday', 'gametime', 'game_id', 'home_team', 'away_team', 'home_score',
            'away_score', 'overtime', 'location'])
for use_mov in [False, True]:
    for use_home_adv in [False, True]:
        kwargs = dict(games=fixture, K=25, home_adv=50, use_mov=use_mov,
                      mov_scale=old['calc_mov_scale'](fixture), use_home_adv=use_home_adv)
        a, b = old['elo_season'](**kwargs), new['elo_season'](**kwargs)
        pd.testing.assert_frame_equal(a['games'], b['games'], check_exact=True)
        assert a['ratings'] == b['ratings']
check('Exact original/adapted per-game results for MOV and home-field on/off', True)
check('Overtime/tie outcome weights unchanged', np.array_equal(
    new['cfb_home_result'](np.array([21, 14, 21, 14, 14]), np.array([14, 21, 14, 21, 14]),
                           np.array([0, 0, 1, 1, 0])), [1, 0, .75, .25, .5]))
for mode in ['fixed', 'tuned']:
    kwargs = dict(all_games=fixture, first_train_year=2021, first_eval_year=2022,
                  K_grid=[20] if mode == 'fixed' else [20, 25], home_adv_grid=[30],
                  use_carry_grid=(mode == 'tuned'), carryover_grid=[.5, .75],
                  param_tune_window=1, projection_season=2024)
    a, b = old['run_expanding_k_elo'](**kwargs), new['run_expanding_k_elo'](**kwargs)
    pd.testing.assert_frame_equal(a['games'], b['games'], check_exact=True)
    pd.testing.assert_frame_equal(a['K_by_year'], b['K_by_year'], check_exact=True)
    assert a['ratings'] == b['ratings']
check('Exact original/adapted expanding-window predictions, parameters and offseason projection', True)

shell = InteractiveShell.instance()
ns = shell.user_ns
log_path = ROOT / '.validation/execution.log'
start = time.perf_counter()
with log_path.open('w', encoding='utf-8') as log:
    for i, cell in enumerate(adapted['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = cell_source(adapted, i)
        if i == 3:
            # Network loading is separately verified; replay the downloaded snapshot.
            source = source.replace('refresh_data = True', 'refresh_data = False')
        if i in [28, 36, 45]:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'run_expanding_k_elo':
                    for kw in node.keywords:
                        if kw.arg == 'K_grid':
                            kw.value = ast.parse('[45, 60]' if i == 28 else '[40, 60]', mode='eval').body
                        elif kw.arg == 'home_adv_grid':
                            kw.value = ast.parse('[25]', mode='eval').body
                        elif kw.arg == 'carryover_grid':
                            kw.value = ast.parse('[0.5]', mode='eval').body
            source = ast.unparse(ast.fix_missing_locations(tree))
        # Keep CSV exports in the verification directory, not beside the deliverable.
        if i == 57:
            (ROOT / '.validation/results').mkdir(exist_ok=True)
            os.chdir(ROOT / '.validation/results')
        tick = time.perf_counter()
        print(f'RUN cell {i}', flush=True)
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            result = shell.run_cell(source, store_history=False)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f'Cell {i} failed; see {log_path}') from (result.error_before_exec or result.error_in_exec)
        elapsed = time.perf_counter() - tick
        report['cell_execution'].append({'cell': i, 'seconds': round(elapsed, 3)})
        print(f'OK cell {i}: {elapsed:.1f}s', flush=True)
os.chdir(ROOT)
check('All 37 code cells executed in order on full real schedule history with reduced expanding grids',
      len(report['cell_execution']) == sum(c['cell_type'] == 'code' for c in adapted['cells']))
games = ns['games']
elo = ns['cfb_elo']
completed = games[games.completed]
check('Every retained completed fixture has scores and binary overtime',
      completed[['home_score', 'away_score', 'overtime']].notna().all().all()
      and completed.overtime.isin([0, 1]).all())
check('FBS/FCS and neutral postseason fixtures included',
      ((games.home_division == 'fcs') | (games.away_division == 'fcs')).any()
      and ((games.game_type == 'POST') & (games.location == 'Neutral')).any())
check('January postseason fixtures retain the prior football season',
      ((games.game_type == 'POST') & (games.kickoff.dt.year > games.season)).any())
check('No final scores admitted for unplayed games', games.loc[~games.completed, ['home_score', 'away_score']].isna().all().all())
check('Only FBS teams appear in current rankings', set(ns['current_ratings'].team) == set(ns['current_fbs_teams']))
check('Real-data Elo values finite and probabilities bounded',
      np.isfinite(elo['games'][['elo_home_pre', 'elo_away_pre', 'p_home']]).all().all()
      and elo['games'].p_home.between(0, 1, inclusive='neither').all())
check('Each Elo update conserves rating sum', np.allclose(
    elo['games'].elo_home_pre + elo['games'].elo_away_pre,
    elo['games'].elo_home_post + elo['games'].elo_away_post, rtol=0, atol=1e-10))
check('Full rating population mean remains 1000', abs(np.mean(list(elo['ratings'].values())) - 1000) < 1e-10)
check('Parameters fitted strictly before each evaluation season',
      (elo['K_by_year'].train_end < elo['K_by_year'].eval_year).all()
      and (elo['K_by_year'].param_tune_end < elo['K_by_year'].eval_year).all())
check('Neutral games use zero home advantage', (elo['games'].loc[elo['games'].location == 'Neutral', 'home_adv_used'] == 0).all())
check('Future projections include FCS opponents with finite probabilities',
      ns['future_projections'].p_home.notna().all()
      and (~ns['future_projections'].away_team.isin(ns['current_fbs_teams'])).any())
for label, table in [('daily', ns['daily_projections']), ('weekly', ns['weekly_projections']),
                     ('future', ns['future_projections'])]:
    check(f'{label} projection export written', (ROOT / '.validation/results' / f'{label}_projections.csv').exists())

# Execute the actual projection cell with all current games marked completed.
ns['games_saved'] = ns['games']
ns['games'] = ns['games'].copy()
ns['games'].loc[ns['games'].season == ns['current_season'], ['home_score', 'away_score']] = 0
with contextlib.redirect_stdout(io.StringIO()):
    exec(cell_source(adapted, 56), ns)
check('No-upcoming-games case yields empty daily/weekly/future projections',
      all(ns[k].empty for k in ['daily_projections', 'weekly_projections', 'future_projections']))
ns['games'] = ns.pop('games_saved')
with contextlib.redirect_stdout(io.StringIO()):
    exec(cell_source(adapted, 56), ns)

# Make regular/postseason week 1 collide and exercise the real projection cell.
saved = ns['games']
future = saved[(saved.season == ns['current_season']) & ~saved.completed].head(2).copy()
future['gameday'] = str(pd.Timestamp.now(tz='America/Chicago').date())
future['status'] = 'STATUS_SCHEDULED'
future['week'] = 1
future['game_type'] = ['REG', 'POST']
ns['games'] = future
with contextlib.redirect_stdout(io.StringIO()):
    exec(cell_source(adapted, 56), ns)
check('Weekly projection separates regular and postseason week numbers',
      len(ns['future_projections']) == 2 and len(ns['weekly_projections']) == 1)
ns['games'] = saved

# Unknown OT must fail before the fitting workflow.
bad_ot = games.copy()
bad_ot.loc[bad_ot.completed.idxmax(), 'overtime'] = np.nan
check('Data adapter contains explicit missing-OT failure guard',
      'raise ValueError("Missing completed-game overtime metadata' in cell_source(adapted, 3))

report['data'] = {'seasons': [int(games.season.min()), int(games.season.max())],
                  'scheduled_games': len(games), 'completed_games': len(completed),
                  'ot_games': int(completed.overtime.sum()), 'fbs_ranked_teams': len(ns['current_ratings']),
                  'source_latest_completed_kickoff': str(completed.kickoff.max())}
report['verification_grids'] = {'cell_28': {'K': [45, 60], 'home_adv': [25], 'carry': [.5]},
                                'cells_36_45': {'K': [40, 60], 'home_adv': [25], 'carry': [.5]},
                                'cell_22': 'full original 64-combination grid'}
report['results'] = {'accuracy': float(ns['eval_games'].correct.mean()),
                     'logloss': float(ns['eval_games'].ll.mean()),
                     'spread_mae': float(ns['spread_validation'].spread_abs_error.mean()),
                     'production_spread_slope': float(ns['spread_slope'])}
report['seconds'] = round(time.perf_counter() - start, 2)
report['full_delivered_grids_executed'] = False
(ROOT / '.validation/report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report['data']), flush=True)
print(json.dumps(report['results']), flush=True)
print(f'Verification completed in {report["seconds"]}s', flush=True)
