from nba_api.stats.endpoints import leaguegamelog
import pandas as pd
import json

# Query the API
logs = leaguegamelog.LeagueGameLog(season="2024-25")
df = logs.get_data_frames()[0]

# Confirm it worked
print(f"✅ API works — pulled {len(df)} rows")
print(df.columns.tolist())
print(df.head(3))
