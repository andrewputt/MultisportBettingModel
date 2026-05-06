"""
backtest.py
─────────────────────────────────────────────────────────────────────────────
Simulates historical bets from the Golden CSV and produces a performance
report with A-F grade assignments.

Weighting scheme
  Before May 15 2026:  75 % 2025 data + 25 % 2026 data (sample-size caution)
  From May 15 2026 on: 100 % 2026 data  (sufficient 2026 sample)

EV filter
  Only bets where  edge  ≥  EV_THRESHOLD (8 %)  are simulated.

Grading rubric  (based on edge % of winning bets in a market)
  A  ≥ 15 %
  B  ≥ 10 %
  C  ≥  6 %
  D  ≥  2 %
  F  <  2 %

Output
  data/processed/backtest_report.json
  data/processed/backtest_report.csv

Usage
  python backtest.py
  python backtest.py --ev-threshold 0.10 --report-date 2026-04-01
"""

import argparse
import json
import os
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning)

# ── env ──────────────────────────────────────────────────────────────────────
ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
EV_THRESHOLD = 0.08                          # 8 % probability edge
PIVOT_DATE   = date(2026, 5, 15)             # shift to 100 % 2026 data

WEIGHT_BEFORE_PIVOT = {2025: 0.75, 2026: 0.25}
WEIGHT_AFTER_PIVOT  = {2025: 0.00, 2026: 1.00}

GRADE_THRESHOLDS = [
    (0.15, "A"),
    (0.10, "B"),
    (0.06, "C"),
    (0.02, "D"),
    (0.00, "F"),
]

UNIT_SIZE = 1.0      # 1 unit per bet (all graded bets same stake for simplicity)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def assign_grade(edge_pct: float) -> str:
    for threshold, grade in GRADE_THRESHOLDS:
        if edge_pct >= threshold:
            return grade
    return "F"


def season_weight(row_date: Optional[date], report_date: date) -> dict:
    """Return the season weight dict appropriate for the row's date."""
    if row_date is None:
        return WEIGHT_BEFORE_PIVOT
    return WEIGHT_AFTER_PIVOT if report_date >= PIVOT_DATE else WEIGHT_BEFORE_PIVOT


def american_from_prob(prob: float) -> float:
    """Convert a true probability back to approximate American odds."""
    prob = max(0.01, min(0.99, prob))
    if prob >= 0.5:
        return round(-prob / (1 - prob) * 100)
    return round((1 - prob) / prob * 100)


def simulate_pnl(row: pd.Series) -> float:
    """
    Naive PnL simulation for a single prop over/under bet.
    Checks 'w_l' column (Baseball-Reference format: 'W', 'L', 'W-wo', etc.)
    to determine game outcome.  Falls back to Monte Carlo if unavailable.
    """
    line_american = row.get("american", None)
    model_prob    = float(row.get("model_prob", 0.5))
    result        = row.get("result", None)

    if result is None:
        wl = str(row.get("w_l", ""))
        if "W" in wl:
            result = 1
        elif "L" in wl:
            result = 0
        else:
            # no outcome recorded — simulate probabilistically
            result = int(np.random.random() < model_prob)

    if result == 1:
        # Compute profit multiplier — handle both decimal and American odds formats.
        # Decimal odds (1.0–100 range): profit per unit = odds − 1
        # American +150: profit per unit = 150/100 = 1.50
        # American −150: profit per unit = 100/150 ≈ 0.667
        if pd.notna(line_american):
            price = float(line_american)
            if 1.0 < price < 100.0:
                # Decimal format (Odds API default)
                profit_mult = price - 1.0
            elif price >= 100.0:
                # Positive American
                profit_mult = price / 100.0
            else:
                # Negative American
                profit_mult = 100.0 / abs(price)
        else:
            profit_mult = 1.0   # even money fallback
        return UNIT_SIZE * profit_mult
    else:
        return -UNIT_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# CORE BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    report_date: date,
    ev_threshold: float,
) -> pd.DataFrame:
    """
    Filter to EV-positive bets, apply season weighting, simulate PnL,
    and return the annotated DataFrame.
    """
    if df.empty:
        return df

    # ── ensure required columns ───────────────────────────────────────────────
    for col, default in [
        ("edge", 0.0),
        ("model_prob", 0.5),
        ("no_vig_over_prob", 0.5),
        ("market", "unknown"),
        ("season", 2025),
    ]:
        if col not in df.columns:
            df[col] = default

    df["edge"]       = pd.to_numeric(df["edge"],       errors="coerce").fillna(0.0)
    df["model_prob"] = pd.to_numeric(df["model_prob"], errors="coerce").fillna(0.5)
    df["season"]     = pd.to_numeric(df["season"],     errors="coerce").fillna(2025).astype(int)

    # ── EV filter ─────────────────────────────────────────────────────────────
    bet_df = df[df["edge"] >= ev_threshold].copy()
    print(f"  Props meeting EV threshold ({ev_threshold:.0%}): {len(bet_df):,} / {len(df):,}")

    if bet_df.empty:
        print("  No qualifying bets — lowering threshold check or awaiting more data.")
        return bet_df

    # ── season weighting ──────────────────────────────────────────────────────
    weights = WEIGHT_AFTER_PIVOT if report_date >= PIVOT_DATE else WEIGHT_BEFORE_PIVOT
    bet_df["season_weight"] = bet_df["season"].apply(
        lambda s: weights.get(int(s), 0.0)
    )
    # drop rows from excluded seasons (weight = 0)
    bet_df = bet_df[bet_df["season_weight"] > 0].copy()
    print(f"  After season weighting: {len(bet_df):,} bets  "
          f"(regime: {'POST-PIVOT 100% 2026' if report_date >= PIVOT_DATE else 'PRE-PIVOT 75/25'})")

    # ── simulate PnL ──────────────────────────────────────────────────────────
    np.random.seed(42)
    bet_df["simulated_pnl"] = bet_df.apply(simulate_pnl, axis=1)
    bet_df["weighted_pnl"]  = bet_df["simulated_pnl"] * bet_df["season_weight"]

    return bet_df


# ─────────────────────────────────────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def build_report(bet_df: pd.DataFrame, report_date: date, ev_threshold: float) -> dict:
    # 1. MOVE THESE UP: Define these before the "if bet_df.empty" check
    weights = WEIGHT_AFTER_PIVOT if report_date >= PIVOT_DATE else WEIGHT_BEFORE_PIVOT
    regime = "POST_PIVOT" if report_date >= PIVOT_DATE else "PRE_PIVOT"

    if bet_df.empty:
        return {
            "report_date": report_date.isoformat(),
            "ev_threshold": ev_threshold,
            "status": "NO_QUALIFYING_BETS",
            "regime": regime,           # Now the script won't crash here
            "season_weights": weights,  # Now the summary will print correctly
            "markets": [],
        }

    market_reports: list[dict] = []

    for market, grp in bet_df.groupby("market"):
        n_bets     = len(grp)
        total_pnl  = float(grp["weighted_pnl"].sum())
        avg_edge   = float(grp["edge"].mean())
        win_rate   = float((grp["simulated_pnl"] > 0).mean())
        roi        = total_pnl / (n_bets * UNIT_SIZE) if n_bets else 0.0
        grade      = assign_grade(avg_edge)

        # top plays within this market
        top_plays = (
            grp.nlargest(5, "edge")[
                [c for c in ["player", "line", "edge", "model_prob", "no_vig_over_prob", "season"]
                 if c in grp.columns]
            ]
            .to_dict(orient="records")
        )

        market_reports.append({
            "market":     market,
            "n_bets":     n_bets,
            "avg_edge":   round(avg_edge, 4),
            "win_rate":   round(win_rate, 4),
            "total_pnl":  round(total_pnl, 2),
            "roi":        round(roi, 4),
            "grade":      grade,
            "top_plays":  top_plays,
        })

    # sort by grade then ROI
    grade_order = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}
    market_reports.sort(key=lambda x: (grade_order.get(x["grade"], 5), -x["roi"]))

    overall_pnl   = float(bet_df["weighted_pnl"].sum())
    overall_bets  = len(bet_df)
    overall_roi   = overall_pnl / (overall_bets * UNIT_SIZE) if overall_bets else 0.0

    weights = WEIGHT_AFTER_PIVOT if report_date >= PIVOT_DATE else WEIGHT_BEFORE_PIVOT

    return {
        "report_date":    report_date.isoformat(),
        "ev_threshold":   ev_threshold,
        "pivot_date":     PIVOT_DATE.isoformat(),
        "season_weights": weights,
        "regime":         "POST_PIVOT" if report_date >= PIVOT_DATE else "PRE_PIVOT",
        "overall": {
            "total_bets":  overall_bets,
            "total_pnl":   round(overall_pnl, 2),
            "overall_roi": round(overall_roi, 4),
            "overall_grade": assign_grade(float(bet_df["edge"].mean())),
        },
        "markets": market_reports,
    }


def print_report_summary(report: dict) -> None:
    """Pretty-print the report to stdout."""
    ov = report.get("overall", {})
    print("\n" + "═" * 60)
    print(f"  MLB PROP BACKTEST REPORT  —  {report['report_date']}")
    print("═" * 60)
    print(f"  Regime        : {report['regime']}")
    print(f"  Season weights: {report['season_weights']}")
    print(f"  EV threshold  : {report['ev_threshold']:.0%}")
    print(f"  Total bets    : {ov.get('total_bets', 0):,}")
    print(f"  Total PnL     : {ov.get('total_pnl', 0):+.2f} units")
    print(f"  Overall ROI   : {ov.get('overall_roi', 0):.1%}")
    print(f"  Overall Grade : {ov.get('overall_grade', 'N/A')}")
    print("─" * 60)
    print(f"  {'Market':<35}  {'Bets':>5}  {'Avg Edge':>8}  {'ROI':>7}  {'Grade':>5}")
    print("─" * 60)
    for m in report.get("markets", []):
        print(
            f"  {m['market']:<35}  {m['n_bets']:>5}  "
            f"{m['avg_edge']:>8.1%}  {m['roi']:>7.1%}  {m['grade']:>5}"
        )
    print("═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run(report_date: date, ev_threshold: float) -> dict:
    golden_path = PROCESSED_DIR / "golden.csv"
    if not golden_path.exists():
        print(
            f"[backtest] ERROR: {golden_path} not found.\n"
            "  Run process_model.py first to generate the Golden CSV."
        )
        return {}

    print(f"[backtest] Loading {golden_path}…")
    df = pd.read_csv(golden_path, low_memory=False)
    print(f"  Rows loaded: {len(df):,}")

    bet_df = run_backtest(df, report_date, ev_threshold)
    report = build_report(bet_df, report_date, ev_threshold)

    # save JSON
    json_path = PROCESSED_DIR / "backtest_report.json"
    json_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"[backtest] Report saved → {json_path}")

    # save CSV (bet-level detail)
    if not bet_df.empty:
        csv_path = PROCESSED_DIR / "backtest_report.csv"
        bet_df.to_csv(csv_path, index=False)
        print(f"[backtest] Bet-level CSV saved → {csv_path}")

    print_report_summary(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest MLB prop model bets.")
    parser.add_argument(
        "--report-date",
        default=date.today().isoformat(),
        help="Date to treat as 'today' for weighting decisions (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--ev-threshold",
        type=float,
        default=EV_THRESHOLD,
        help=f"Minimum edge to qualify a bet (default: {EV_THRESHOLD}).",
    )
    args = parser.parse_args()
    run(
        report_date=date.fromisoformat(args.report_date),
        ev_threshold=args.ev_threshold,
    )