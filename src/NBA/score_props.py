"""
score_props.py
────────────────────────────────────────────────────────────────────────────
Fetches actual player stats from completed games and scores past predictions
in props_log.csv. Writes results to props_scored.csv and prints a summary.

Usage:
    python src/NBA/score_props.py              # score all pending predictions
    python src/NBA/score_props.py --date 2026-05-03   # score a specific date
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players as nba_players

PROPS_LOG    = Path("src/NBA/data/props_log.csv")
PROPS_SCORED = Path("src/NBA/data/props_scored.csv")

CURRENT_SEASON = "2025-26"

STAT_COL_MAP = {
    "PTS":  "PTS",
    "REB":  "REB",
    "AST":  "AST",
    "FG3M": "FG3M",
    "STL":  "STL",
    "BLK":  "BLK",
}


def get_player_id(name: str) -> int | None:
    results = nba_players.find_players_by_full_name(name)
    if results:
        return results[0]["id"]
    last = name.split()[-1]
    results = nba_players.find_players_by_last_name(last)
    active = [p for p in results if p["is_active"]]
    if len(active) == 1:
        return active[0]["id"]
    return None


def fetch_player_game_stats(player_id: int) -> pd.DataFrame:
    """Fetch all games this season for a player."""
    frames = []
    for stype in ["Playoffs", "Regular Season"]:
        try:
            df = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=CURRENT_SEASON,
                season_type_all_star=stype,
                timeout=30,
            ).get_data_frames()[0]
            if not df.empty:
                frames.append(df)
            time.sleep(0.5)
        except Exception as e:
            print(f"  Warning: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="mixed").dt.strftime("%Y-%m-%d")
    return df


def score_predictions(log: pd.DataFrame, score_date: str | None) -> pd.DataFrame:
    today     = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    scoreable = log[log["game_date"] < today].copy()
    if score_date:
        scoreable = scoreable[scoreable["game_date"] == score_date].copy()

    if scoreable.empty:
        print("No completed predictions to score.")
        return log

    # Fetch actual stats per player
    pid_cache:   dict[str, int | None]   = {}
    stats_cache: dict[str, pd.DataFrame] = {}

    for player_name in scoreable["player"].unique():
        if player_name not in pid_cache:
            pid_cache[player_name] = get_player_id(player_name)
        pid = pid_cache[player_name]
        if pid is None:
            print(f"  Could not resolve player: {player_name}")
            continue
        if player_name not in stats_cache:
            print(f"  Fetching actuals: {player_name}")
            stats_cache[player_name] = fetch_player_game_stats(pid)

    # Match each prediction row to actual stat
    actual_vals = []
    for _, row in scoreable.iterrows():
        player = row["player"]
        gdate  = row["game_date"]
        stat   = row["stat"]
        thresh = row["threshold"]
        direction = row["direction"]

        act_stat_val = np.nan
        hit          = pd.NA
        model_correct = pd.NA

        pstats = stats_cache.get(player, pd.DataFrame())
        if not pstats.empty and stat in STAT_COL_MAP:
            api_col  = STAT_COL_MAP[stat]
            game_row = pstats[pstats["GAME_DATE"] == gdate]
            if not game_row.empty and api_col in game_row.columns:
                act_stat_val = pd.to_numeric(game_row[api_col].iloc[0], errors="coerce")
                if not pd.isna(act_stat_val):
                    hit = int(act_stat_val >= thresh)
                    if direction == "YES":
                        model_correct = int(hit == 1)
                    else:
                        model_correct = int(hit == 0)

        actual_vals.append({
            "actual_stat":    act_stat_val,
            "hit":            hit,
            "model_correct":  model_correct,
        })

    scored = scoreable.copy()
    scored["actual_stat"]   = [v["actual_stat"]   for v in actual_vals]
    scored["hit"]           = [v["hit"]           for v in actual_vals]
    scored["model_correct"] = [v["model_correct"] for v in actual_vals]

    # Merge back into full log
    future = log[log["game_date"] >= today].copy() if score_date is None else \
             log[log["game_date"] != score_date].copy()

    combined = pd.concat([scored, future], ignore_index=True)
    combined = combined.sort_values(["game_date", "player", "stat", "threshold"]).reset_index(drop=True)
    return combined


def print_summary(df: pd.DataFrame):
    scored = df[df["model_correct"].notna()].copy()
    if scored.empty:
        print("No scored predictions yet.")
        return

    scored["model_correct"] = pd.to_numeric(scored["model_correct"], errors="coerce")
    total   = len(scored)
    correct = int(scored["model_correct"].sum())
    acc     = correct / total if total else 0

    print(f"\n{'='*60}")
    print(f"PROP MODEL PERFORMANCE  ({total} predictions scored)")
    print(f"{'='*60}")
    print(f"Overall correct: {correct}/{total} = {acc:.1%}")

    # By stat
    print("\nBy stat:")
    for stat, grp in scored.groupby("stat"):
        n  = len(grp)
        c  = int(grp["model_correct"].sum())
        print(f"  {stat:<5}  {c}/{n} = {c/n:.1%}")

    # By direction
    print("\nBy direction:")
    for d, grp in scored.groupby("direction"):
        n  = len(grp)
        c  = int(grp["model_correct"].sum())
        print(f"  {d:<4}  {c}/{n} = {c/n:.1%}")

    # Edges >= 10%
    big = scored[scored["edge"] >= 0.10]
    if not big.empty:
        n  = len(big)
        c  = int(big["model_correct"].sum())
        print(f"\nHigh-edge (≥10%) bets: {c}/{n} = {c/n:.1%}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None,
                        help="Score predictions for a specific date, e.g. 2026-05-03")
    args = parser.parse_args()

    if not PROPS_LOG.exists():
        print(f"No props log found at {PROPS_LOG}. Run predict_props.py first.")
        raise SystemExit(1)

    log = pd.read_csv(PROPS_LOG)
    print(f"Loaded {len(log)} predictions from {PROPS_LOG}")

    scored = score_predictions(log, args.date)
    scored.to_csv(PROPS_SCORED, index=False)
    print(f"Saved → {PROPS_SCORED}")
    print_summary(scored)
