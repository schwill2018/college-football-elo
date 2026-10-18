"""Exercise loader failure behavior without changing the real data files."""
import contextlib
import io
import json
from pathlib import Path
from unittest.mock import patch
import pandas as pd

n = json.loads(Path('CFB_elo_v3.ipynb').read_text(encoding='utf-8'))
source = ''.join(n['cells'][3]['source']).replace('refresh_data = True', 'refresh_data = False')
read_parquet = pd.read_parquet

def omit_one_result(path, *args, **kwargs):
    frame = read_parquet(path, *args, **kwargs)
    if Path(path).name == 'cfb_schedule_master.parquet':
        frame = frame[frame.game_id != 232350023]
    return frame

with patch.object(pd, 'read_parquet', side_effect=omit_one_result):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(source, {})
    except ValueError as exc:
        assert 'Missing completed-game overtime metadata' in str(exc)
        print('PASS: Missing completed-game overtime metadata fails before model fitting.')
    else:
        raise AssertionError('Missing overtime metadata was silently accepted.')

with contextlib.redirect_stdout(io.StringIO()):
    ns = {}
    exec(source, ns)
games = ns['games']
assert not games.status.isin(['STATUS_CANCELED', 'STATUS_CANCELLED']).any()
assert not ((games.season == 2004) & (games.game_id.isin([63827, 63840]))).any()
assert games.loc[games.game_id == 243252065, 'overtime'].item() == 1
assert games.loc[games.game_id == 63843, 'gameday'].item() == '2006-09-16'
assert (games.loc[games.game_id == 63843, 'gametime'].item() == '23:05:00')
assert games.loc[games.game_id == 63842, 'gametime'].item() == '15:00:00'
assert all(isinstance(t, str) for t in games.home_team)
print('PASS: Cancellations excluded, historical duplicates removed, supplemental OT and local dates correct, IDs are strings.')
Path('.validation/data_edges.txt').write_text('PASS: missing OT error, cancellations, duplicates, supplemental overtime, Central dates and stable ID types.\n')
