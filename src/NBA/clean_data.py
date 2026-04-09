import pandas as pd
import json
import os

os.makedirs("src/NBA/data", exist_ok=True)

# Load raw data
df = pd.read_csv("src/NBA/data/game_logs.csv")

print(f"Raw rows: {len(df)}")
print(f"Nulls:\n{df.isnull().sum()}")

# 1. Convert date to datetime
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

# 2. Convert W/L to binary
df["WIN"] = df["WL"].map({"W": 1, "L": 0})

# 3. Drop rows missing critical fields
df = df.dropna(subset=["PTS", "FG_PCT", "REB", "AST", "PLUS_MINUS"])

# 4. Sort by team and date
df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)

# 5. Add rest days column (days since last game per team)
df["REST_DAYS"] = df.groupby("TEAM_ABBREVIATION")["GAME_DATE"].diff().dt.days.fillna(0)

# 6. Add home/away flag
df["IS_HOME"] = df["MATCHUP"].apply(lambda x: 1 if "vs." in x else 0)

print(f"\nClean rows: {len(df)}")
print(df.head(3))

# Save
df.to_csv("src/NBA/data/game_logs_clean.csv", index=False)

# Save sample JSON
sample = df.head(5).to_dict(orient="records")
# Convert dates to string for JSON
for row in sample:
    row["GAME_DATE"] = str(row["GAME_DATE"])
with open("src/NBA/data/sample_clean.json", "w") as f:
    json.dump(sample, f, indent=2)

print("✅ Saved game_logs_clean.csv and sample_clean.json")
