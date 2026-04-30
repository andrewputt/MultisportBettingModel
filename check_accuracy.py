#!/usr/bin/env python3
"""
check_accuracy.py
Quick script to display your current prediction accuracy and stats.
Run this after games finish to see how you're doing.
"""

import pandas as pd
from datetime import datetime

try:
    df = pd.read_csv("src/NBA/data/predictions_scored.csv")

    # Only count games that have been scored
    scored = df[df["MODEL_CORRECT"].notna()].copy()

    if scored.empty:
        print("No scored games yet. Check back after games finish!")
    else:
        accuracy = scored["MODEL_CORRECT"].mean()
        total = len(scored)
        correct = scored["MODEL_CORRECT"].sum()

        print("=" * 60)
        print("NBA PREDICTION MODEL — ACCURACY REPORT")
        print("=" * 60)
        print(f"\nAccuracy: {accuracy:.1%} ({correct}/{total} games correct)")

        # By run date
        if "RUN_DATE" in scored.columns:
            by_date = scored.groupby("RUN_DATE")["MODEL_CORRECT"].agg(
                ["sum", "count", "mean"]
            )
            by_date.columns = ["Correct", "Total", "Accuracy"]
            by_date["Accuracy"] = by_date["Accuracy"].apply(lambda x: f"{x:.1%}")
            print(f"\nBy Date:")
            print(by_date.to_string())

        # High edge games
        if "EDGE" in scored.columns:
            high_edge = scored[(scored["EDGE"].notna()) & (scored["EDGE"] >= 0.05)]
            if not high_edge.empty:
                high_edge_acc = high_edge["MODEL_CORRECT"].mean()
                print(f"\nHigh Edge Games (>=5%):")
                print(
                    f"  Accuracy: {high_edge_acc:.1%} ({high_edge['MODEL_CORRECT'].sum()}/{len(high_edge)})"
                )

        print("\n" + "=" * 60)
        print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

except FileNotFoundError:
    print("predictions_scored.csv not found. Run predictions first!")
except Exception as e:
    print(f"Error: {e}")
