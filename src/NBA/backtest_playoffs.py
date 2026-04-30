"""
backtest_playoffs.py
Evaluate model performance specifically on playoff games from recent seasons.
Shows accuracy and edge realization for playoff predictions.
"""

import pickle
import pandas as pd
import numpy as np

# Load trained model
with open("src/NBA/models/nba_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load features
df = pd.read_csv("src/NBA/data/features.csv")
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

# Filter to playoff games only
playoff_df = df[df["IS_PLAYOFF"] == 1].copy()

print("=" * 80)
print("PLAYOFF BACKTEST: Model Performance on Playoff Games")
print("=" * 80)

# Feature columns (must match training - these exist in features.csv)
FEATURE_COLS = [
    "IS_HOME",
    "IS_PLAYOFF",
    "REST_DAYS",
    "WIN_PCT_L10",
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

# Remove rows with missing features
playoff_df = playoff_df.dropna(subset=FEATURE_COLS + ["WIN"])

if playoff_df.empty:
    print("No playoff data available.")
else:
    # Get predictions
    X = playoff_df[FEATURE_COLS]
    y_true = playoff_df["WIN"]
    probs = model.predict_proba(X)[:, 1]
    predictions = (probs >= 0.5).astype(int)

    # Calculate metrics
    accuracy = (predictions == y_true).mean()
    correct = (predictions == y_true).sum()
    total = len(y_true)

    # By season
    playoff_df["PRED_PROB"] = probs
    playoff_df["PRED"] = predictions
    playoff_df["CORRECT"] = (predictions == y_true).astype(int)
    playoff_df["YEAR"] = playoff_df["GAME_DATE"].dt.year

    print(f"\nOVERALL PLAYOFF PERFORMANCE:")
    print(f"  Games: {total}")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Correct: {correct}/{total}")

    print(f"\nBY YEAR:")
    for year in sorted(playoff_df["YEAR"].unique()):
        year_data = playoff_df[playoff_df["YEAR"] == year]
        year_acc = year_data["CORRECT"].mean()
        year_n = len(year_data)
        print(f"  {year}: {year_acc:.2%} ({year_n} games)")

    # Edge analysis: simulate betting on predicted favorites
    playoff_df["EDGE_IF_BET"] = playoff_df["PRED_PROB"] - 0.5  # Simple edge metric
    positive_edge_bets = playoff_df[playoff_df["PRED"] == 1]
    if not positive_edge_bets.empty:
        edge_acc = positive_edge_bets["CORRECT"].mean()
        edge_count = len(positive_edge_bets)
        print(f"\nBETS ON PREDICTED FAVORITES:")
        print(f"  Games: {edge_count}")
        print(f"  Accuracy: {edge_acc:.2%}")

    # Year-by-year detailed breakdown
    print(f"\nDETAILED BY YEAR:")
    for year in sorted(playoff_df["YEAR"].unique()):
        year_data = playoff_df[playoff_df["YEAR"] == year]
        year_acc = year_data["CORRECT"].mean()
        year_n = len(year_data)
        year_avg_prob = year_data["PRED_PROB"].mean()
        print(
            f"  {year}: {year_acc:.2%} accuracy, {year_n} games, avg pred prob {year_avg_prob:.3f}"
        )

print("\n" + "=" * 80)
