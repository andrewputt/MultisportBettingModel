"""
score_predictions.py
Fetches actual NBA game results and scores past predictions in predictions_log.csv.
Adds ACTUAL_WIN and MODEL_CORRECT columns, saves to predictions_scored.csv.

Run this daily after games finish to track live model performance.
"""

import time
from datetime import datetime, timezone

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

LOG_PATH = "src/NBA/data/predictions_log.csv"
SCORED_PATH = "src/NBA/data/predictions_scored.csv"


def fetch_actual_results():
    """Fetch all 2025-26 game results (regular season + playoffs)."""
    print("Fetching actual game results from NBA API...")
    fields = ["GAME_DATE", "TEAM_ABBREVIATION", "WL"]
    frames = []
    for season_type in ["Regular Season", "Playoffs"]:
        logs = leaguegamelog.LeagueGameLog(season="2025-26", season_type_all_star=season_type)
        part = logs.get_data_frames()[0][fields].copy()
        frames.append(part)
        time.sleep(1)
    results = pd.concat(frames, ignore_index=True)
    results["GAME_DATE"] = pd.to_datetime(results["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    results["ACTUAL_WIN"] = results["WL"].map({"W": 1, "L": 0})
    print(f"  Fetched results for {len(results)} team-game records")
    return results[["GAME_DATE", "TEAM_ABBREVIATION", "ACTUAL_WIN"]]


def score_predictions(log: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log["GAME_DATE"] = pd.to_datetime(log["GAME_DATE"]).dt.strftime("%Y-%m-%d")

    # Only score games that have already been played
    scoreable = log[log["GAME_DATE"] < today].copy()
    future = log[log["GAME_DATE"] >= today].copy()

    scored = scoreable.merge(
        results,
        left_on=["GAME_DATE", "ABB"],
        right_on=["GAME_DATE", "TEAM_ABBREVIATION"],
        how="left",
    ).drop(columns=["TEAM_ABBREVIATION"])

    # Model is correct if it predicted win (prob > 0.5) and team actually won, or vice versa
    scored["MODEL_CORRECT"] = (
        ((scored["INJ_ADJUSTED_PROB"] > 0.5) & (scored["ACTUAL_WIN"] == 1)) |
        ((scored["INJ_ADJUSTED_PROB"] <= 0.5) & (scored["ACTUAL_WIN"] == 0))
    ).astype("Int64")

    combined = pd.concat([scored, future], ignore_index=True)
    combined = combined.sort_values(["GAME_DATE", "MATCHUP"]).reset_index(drop=True)
    return combined


def print_summary(df: pd.DataFrame):
    scored = df[df["ACTUAL_WIN"].notna()]
    if scored.empty:
        print("No completed games to score yet.")
        return

    total = len(scored)
    correct = int(scored["MODEL_CORRECT"].sum())
    accuracy = correct / total

    print(f"\n{'='*60}")
    print(f"LIVE MODEL PERFORMANCE ({total} predictions scored)")
    print(f"{'='*60}")
    print(f"Overall accuracy: {correct}/{total} = {accuracy:.1%}")

    # Break down by edge bucket
    edges = scored[scored["EDGE"].notna() & (scored["EDGE"] >= 0.05)]
    if not edges.empty:
        e_total = len(edges)
        e_correct = int(edges["MODEL_CORRECT"].sum())
        print(f"Edges >= 5% accuracy: {e_correct}/{e_total} = {e_correct/e_total:.1%}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    try:
        log = pd.read_csv(LOG_PATH)
    except FileNotFoundError:
        print(f"No predictions log found at {LOG_PATH}. Run predict_today.py first.")
        raise SystemExit(1)

    print(f"Loaded {len(log)} predictions from log")

    results = fetch_actual_results()
    scored = score_predictions(log, results)
    scored.to_csv(SCORED_PATH, index=False)
    print(f"Saved scored predictions to {SCORED_PATH}")

    print_summary(scored)
