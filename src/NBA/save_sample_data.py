from nba_api.stats.endpoints import leaguegamelog
import pandas as pd
import json
import time
import os

os.makedirs("src/NBA/data", exist_ok=True)

SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
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

frames = []
for season in SEASONS:
    print(f"Fetching {season}...")
    logs = leaguegamelog.LeagueGameLog(season=season, season_type_all_star="Regular Season")
    df = logs.get_data_frames()[0][fields]
    df["SEASON"] = season
    frames.append(df)
    time.sleep(1)  # avoid rate limiting

combined = pd.concat(frames, ignore_index=True)
combined.to_csv("src/NBA/data/game_logs.csv", index=False)

# Save top 5 as JSON sample
sample = combined.head(5).to_dict(orient="records")
with open("src/NBA/data/sample_game_logs.json", "w") as f:
    json.dump(sample, f, indent=2)

print(f"Saved {len(combined)} rows across {len(SEASONS)} seasons to CSV + JSON sample")
