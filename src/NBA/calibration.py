#!/usr/bin/env python3
"""
calibration.py
Reliability diagram: compares model predicted probability against actual win rate.
Run after accumulating scored predictions to see if the model is over/underconfident.
50+ predictions recommended for reliable results.
"""

import matplotlib.pyplot as plt
import pandas as pd


def run_calibration():
    try:
        df = pd.read_csv("src/NBA/data/predictions_scored.csv")
    except FileNotFoundError:
        print("predictions_scored.csv not found. Run score_predictions.py first.")
        return

    scored = df[df["MODEL_CORRECT"].notna() & df["ACTUAL_WIN"].notna()].copy()
    scored["ACTUAL_WIN"] = scored["ACTUAL_WIN"].astype(int)

    n = len(scored)
    if n < 10:
        print(f"Only {n} scored predictions — need more data for calibration.")
        return

    print(f"\nCalibration report — {n} scored predictions")
    if n < 50:
        print("(note: fewer than 50 samples, treat as directional only)\n")
    else:
        print()

    # ── 1. Reliability by model probability ───────────────────────────────────
    bins = [0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 1.0]
    labels = ["<40%", "40-50%", "50-60%", "60-70%", "70-80%", ">80%"]
    scored["PROB_BIN"] = pd.cut(scored["INJ_ADJUSTED_PROB"], bins=bins, labels=labels)
    reliability = (
        scored.groupby("PROB_BIN", observed=True)
        .agg(n=("ACTUAL_WIN", "count"), actual=("ACTUAL_WIN", "mean"), avg_prob=("INJ_ADJUSTED_PROB", "mean"))
        .reset_index()
    )

    print("── Model Probability vs Actual Win Rate ──────────────────────")
    print(f"{'Bucket':<10} {'N':>5} {'Model Avg':>10} {'Actual Win%':>12} {'Delta':>8}")
    print("-" * 50)
    for _, row in reliability.iterrows():
        if row["n"] == 0:
            continue
        delta = row["actual"] - row["avg_prob"]
        flag = "  overconfident" if delta < -0.07 else ("  underconfident" if delta > 0.07 else "")
        print(f"{str(row['PROB_BIN']):<10} {int(row['n']):>5} {row['avg_prob']:>9.1%} {row['actual']:>11.1%} {delta:>+7.1%}{flag}")

    # ── 2. Edge bucket performance (value bets only) ───────────────────────────
    value_bets = scored[scored["KELLY_FRACTION"].notna()].copy()
    edge_cal = pd.DataFrame()
    if not value_bets.empty:
        edge_bins = [0.05, 0.10, 0.15, 0.20, 0.25, 1.0]
        edge_labels = ["5-10%", "10-15%", "15-20%", "20-25%", ">25%"]
        value_bets["EDGE_BIN"] = pd.cut(value_bets["EDGE"], bins=edge_bins, labels=edge_labels)
        edge_cal = (
            value_bets.groupby("EDGE_BIN", observed=True)
            .agg(n=("ACTUAL_WIN", "count"), win_rate=("ACTUAL_WIN", "mean"), avg_edge=("EDGE", "mean"))
            .reset_index()
        )

        print(f"\n── Edge Bucket Win Rate (value bets: {len(value_bets)}) ────────────────────")
        print(f"{'Edge Bucket':<12} {'N':>5} {'Avg Edge':>10} {'Win Rate':>10}")
        print("-" * 42)
        for _, row in edge_cal.iterrows():
            if row["n"] == 0:
                continue
            print(f"{str(row['EDGE_BIN']):<12} {int(row['n']):>5} {row['avg_edge']:>9.1%} {row['win_rate']:>9.1%}")

    # ── 3. Brier score ─────────────────────────────────────────────────────────
    brier = ((scored["INJ_ADJUSTED_PROB"] - scored["ACTUAL_WIN"]) ** 2).mean()
    print(f"\nBrier Score: {brier:.4f}  (0.0 = perfect, 0.25 = random coin flip)")

    # ── 4. Chart ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Model Calibration  |  n={n} predictions", fontsize=13)

    # Reliability diagram
    ax = axes[0]
    cal_plot = reliability[reliability["n"] > 0]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax.scatter(
        cal_plot["avg_prob"], cal_plot["actual"],
        s=cal_plot["n"] * 25, color="#1d4e89", zorder=5
    )
    for _, row in cal_plot.iterrows():
        ax.annotate(
            f"n={int(row['n'])}",
            (row["avg_prob"], row["actual"]),
            textcoords="offset points", xytext=(6, 4), fontsize=8,
        )
    ax.set_xlabel("Model Predicted Probability")
    ax.set_ylabel("Actual Win Rate")
    ax.set_title("Reliability Diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    # Edge vs win rate
    ax2 = axes[1]
    if not edge_cal.empty:
        plot_edge = edge_cal[edge_cal["n"] > 0]
        bars = ax2.bar(
            plot_edge["EDGE_BIN"].astype(str), plot_edge["win_rate"],
            color="#1d4e89", alpha=0.85
        )
        ax2.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="50% baseline")
        for bar, (_, row) in zip(bars, plot_edge.iterrows()):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"n={int(row['n'])}", ha="center", fontsize=8,
            )
        ax2.set_xlabel("Edge Bucket")
        ax2.set_ylabel("Actual Win Rate")
        ax2.set_title("Win Rate by Edge Size (value bets only)")
        ax2.set_ylim(0, 1.1)
        ax2.legend()

    plt.tight_layout()
    out_path = "src/NBA/models/calibration.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    run_calibration()
