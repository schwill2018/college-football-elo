from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import pandas as pd
import requests

data = Path('data')
games = pd.concat([pd.read_parquet(data / f'cfb_schedules_{yr}.parquet')
                   for yr in range(2003, 2027)], ignore_index=True)
games = games[games.fbs_participant & games.season_type.isin(['regular', 'postseason'])].copy()
games['fixture_date'] = games.start_date.str[:10]
games = games.sort_values('game_id').drop_duplicates(
    ['season', 'home_team', 'away_team', 'fixture_date'], keep='last')
master = pd.read_parquet(data / 'cfb_schedule_master.parquet')
missing = games[games.completed & ~games.game_id.isin(master.game_id)]

def fetch(row):
    game_id = int(row.game_id)
    if game_id in [63843, 63844, 63845, 63846]:
        # Official school box scores have exactly four periods; the schedule
        # lists these four home kickoffs at 6:05 PM Hawaii time.
        day = pd.Timestamp(row.start_date[:10], tz='Pacific/Honolulu') + pd.Timedelta(hours=18, minutes=5)
        return dict(game_id=game_id, overtime=0, start_date=day.tz_convert('UTC').isoformat(),
                    source=f'https://static.hawaiiathletics.com/custompages/Stats/Fball/2006/{day:%m%d%y}.HTM')
    espn_id = 243392226 if game_id == 63842 else game_id
    url = f'https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary?event={espn_id}'
    path = data / f'espn_summary_{espn_id}.json'
    if path.exists():
        payload = json.loads(path.read_text(encoding='utf-8'))
    else:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        payload = response.json()
    competition = payload['header']['competitions'][0]
    status = competition['status']['type']
    assert status['completed']
    detail = status['detail']
    assert detail == 'Final' or 'OT' in detail, detail
    scores = {c['homeAway']: float(c['score']) for c in competition['competitors']}
    assert scores == {'home': float(row.home_points), 'away': float(row.away_points)}
    return dict(game_id=game_id, overtime=int('OT' in detail),
                start_date=competition['date'], source=url)

with ThreadPoolExecutor(max_workers=6) as pool:
    supplement = pd.DataFrame(pool.map(fetch, missing.itertuples()))
supplement.sort_values('game_id').to_csv(data / 'cfb_overtime_supplement.csv', index=False)
print(supplement.to_string(index=False))
