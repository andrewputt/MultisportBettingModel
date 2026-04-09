from nba_api.stats.endpoints import leaguegamelog
import pandas as pd
import json
import os

os.makedirs("src/NBA/data", exist_ok=True)

logs = leaguegamelog.LeagueGameLog(season="2024-25")
df = logs.get_data_frames()[0]

# Keep only the fields we need
fields = [
    "GAME_ID",
    "GAME_DATE",
    "TEAM_ABBREVIATION",
    "MATCHUP",
    "WL",
    "PTS",
    "FG_PCT",
    "REB",
    "AST",
    "PLUS_MINUS",
]
df = df[fields]

df.to_csv("src/NBA/data/game_logs.csv", index=False)

# Save top 5 as JSON sample
sample = df.head(5).to_dict(orient="records")
with open("src/NBA/data/sample_game_logs.json", "w") as f:
    json.dump(sample, f, indent=2)

print(f"✅ Saved {len(df)} rows to CSV + JSON sample")
