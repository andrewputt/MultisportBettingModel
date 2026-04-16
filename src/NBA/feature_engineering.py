import pandas as pd
import json
import os

os.makedirs("src/NBA/data", exist_ok=True)

df = pd.read_csv("src/NBA/data/game_logs_clean.csv", parse_dates=["GAME_DATE"])


# ── Rolling window helper ──────────────────────────────────────────
def rolling_avg(df, col, window=10):
    return df.groupby("TEAM_ABBREVIATION")[col].transform(
        lambda x: x.shift(1).rolling(window, min_periods=3).mean()
    )


# ── 1. Rolling win % (last 10 games) ──────────────────────────────
df["WIN_PCT_L10"] = rolling_avg(df, "WIN")

# ── 2. Home vs Away win % ─────────────────────────────────────────
df["WIN_PCT_HOME"] = (
    df[df["IS_HOME"] == 1]
    .groupby("TEAM_ABBREVIATION")["WIN"]
    .transform(lambda x: x.shift(1).expanding().mean())
)
df["WIN_PCT_AWAY"] = (
    df[df["IS_HOME"] == 0]
    .groupby("TEAM_ABBREVIATION")["WIN"]
    .transform(lambda x: x.shift(1).expanding().mean())
)

# ── 3. Offensive rating proxy (rolling avg PTS) ───────────────────
df["OFF_RATING_L10"] = rolling_avg(df, "PTS")

# ── 4. Defensive rating proxy (rolling avg opponent PTS) ──────────
# Build opponent PTS by merging game on GAME_ID
opp = df[["GAME_ID", "TEAM_ABBREVIATION", "PTS"]].copy()
opp.columns = ["GAME_ID", "OPP_TEAM", "OPP_PTS"]
df = df.merge(opp, on="GAME_ID", how="left")
df = df[df["TEAM_ABBREVIATION"] != df["OPP_TEAM"]]  # remove self-join rows
df["DEF_RATING_L10"] = rolling_avg(df, "OPP_PTS")

# ── 5. Pace proxy (rolling avg REB + AST) ─────────────────────────
df["PACE_PROXY_L10"] = rolling_avg(df, "REB") + rolling_avg(df, "AST")

# ── 6. Plus/minus trend ───────────────────────────────────────────
df["PM_TREND_L10"] = rolling_avg(df, "PLUS_MINUS")

# ── 7. Head-to-head opponent features ────────────────────────────
# Build a lookup of each team's L10 stats per game, then merge onto
# the opponent's row so the model sees both sides of the matchup.
team_stats = df[["GAME_ID", "TEAM_ABBREVIATION",
                  "WIN_PCT_L10", "OFF_RATING_L10", "DEF_RATING_L10",
                  "PACE_PROXY_L10", "PM_TREND_L10"]].copy()
team_stats.columns = ["GAME_ID", "OPP_TEAM",
                       "OPP_WIN_PCT_L10", "OPP_OFF_RATING_L10", "OPP_DEF_RATING_L10",
                       "OPP_PACE_PROXY_L10", "OPP_PM_TREND_L10"]
df = df.merge(team_stats, left_on=["GAME_ID", "OPP_TEAM"],
              right_on=["GAME_ID", "OPP_TEAM"], how="left")

# ── Save ──────────────────────────────────────────────────────────
feature_cols = [
    "GAME_ID",
    "GAME_DATE",
    "TEAM_ABBREVIATION",
    "MATCHUP",
    "IS_HOME",
    "REST_DAYS",
    "WIN",
    "WIN_PCT_L10",
    "WIN_PCT_HOME",
    "WIN_PCT_AWAY",
    "OFF_RATING_L10",
    "DEF_RATING_L10",
    "PACE_PROXY_L10",
    "PM_TREND_L10",
    "OPP_WIN_PCT_L10",
    "OPP_OFF_RATING_L10",
    "OPP_DEF_RATING_L10",
    "OPP_PACE_PROXY_L10",
    "OPP_PM_TREND_L10",
]
df_out = df[feature_cols].dropna(subset=["WIN_PCT_L10"])
df_out.to_csv("src/NBA/data/features.csv", index=False)

# Sample JSON
sample = df_out.head(5).copy()
sample["GAME_DATE"] = sample["GAME_DATE"].astype(str)
with open("src/NBA/data/sample_features.json", "w") as f:
    json.dump(sample.to_dict(orient="records"), f, indent=2)

print(f"Engineered features for {len(df_out)} games")
print(df_out.head(3).to_string())
