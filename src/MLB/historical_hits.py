"""
historical_hits.py
─────────────────────────────────────────────────────────────────────────────
Retroactive "what would the model have said?" analysis for the last N days.

Two data sources:

  A. picks_history.json  (primary — prop-level)
     Top-5 picks saved daily by generate_dashboard.py, with results recorded
     via `python generate_dashboard.py --record YYYY-MM-DD WIN LOSS ...`.
     When results are recorded this gives an exact prop-level hit rate.

  B. h2h_2026.csv  (retroactive game-level proxy)
     For each date D in the window, team features are computed from ALL games
     BEFORE date D (no lookahead bias).  A simple home-team model is scored
     against a 0.50 baseline, top-5 by predicted edge are selected, then
     w_l is checked.  This is a game-level proxy — team W/L ≠ player prop
     result — but it shows model directional accuracy without cheating.

     Model features (rolling L20 prior to target date):
       • home_win_rate  — wins / games played
       • home_run_rate  — avg runs scored per game
       • away_run_rate  — avg runs allowed per game (used as pitching proxy)
       • edge = home_win_rate + (home_run_rate − away_run_rate) / 20 − 0.50

w_l mapping  →  'W*' = 1  |  'L*' = 0  |  anything else = skip

Usage
  python src/MLB/historical_hits.py            # last 14 days, both sources
  python src/MLB/historical_hits.py --days 7   # shorter window
  python src/MLB/historical_hits.py --csv      # also write CSV to data/processed/
  python src/MLB/historical_hits.py --source picks   # picks_history only
  python src/MLB/historical_hits.py --source h2h     # h2h proxy only
"""

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent.parent
RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PICKS_JSON    = PROCESSED_DIR / "picks_history.json"
H2H_CSV       = RAW_DIR / "h2h_2026.csv"

ROLLING_WINDOW = 20   # games of history to use when computing features


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def wl_to_result(wl) -> "int | None":
    """'W', 'W-wo', 'W-pk' → 1  |  'L', 'L-wo' → 0  |  else → None"""
    s = str(wl).strip().upper()
    if not s or s in ("NAN", "NONE", ""):
        return None
    if s.startswith("W"):
        return 1
    if s.startswith("L"):
        return 0
    return None


def _parse_bref_dates(raw_series: pd.Series, year: int = 2026) -> pd.Series:
    """Parse 'Tuesday, May 5' → datetime.date, year forced to `year`."""
    cleaned = (
        raw_series.astype(str)
        .str.strip()
        .str.replace(r"^[A-Za-z]+,\s*", "", regex=True)
    )
    return pd.to_datetime(
        cleaned + f", {year}",
        format="%b %d, %Y",
        errors="coerce",
    ).dt.date


# ─────────────────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_h2h() -> pd.DataFrame:
    if not H2H_CSV.exists():
        print(f"  [historical_hits] WARNING: {H2H_CSV} not found.")
        return pd.DataFrame()

    df = pd.read_csv(H2H_CSV, low_memory=False)
    df.columns = [c.lower().strip() for c in df.columns]
    print(f"  [h2h] Loaded {len(df):,} rows from {H2H_CSV.name}")

    # date
    _date_col = next(
        (c for c in ("date", "game_date", "date_game") if c in df.columns), None
    )
    if _date_col:
        df["date_parsed"] = _parse_bref_dates(df[_date_col])
        bad = df["date_parsed"].isna().sum()
        if bad:
            print(f"  [h2h] {bad} rows had unparseable dates — skipped.")
    else:
        df["date_parsed"] = None

    # numeric columns
    for col in ("r", "ra"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # binary outcome
    if "w_l" in df.columns:
        df["result"] = df["w_l"].apply(wl_to_result)
    else:
        df["result"] = None

    # sort chronologically (required for rolling feature build)
    df = df.dropna(subset=["date_parsed"]).sort_values("date_parsed").reset_index(drop=True)
    return df


def load_picks_history() -> dict:
    if not PICKS_JSON.exists():
        print(f"  [historical_hits] WARNING: {PICKS_JSON} not found.")
        return {}
    try:
        with open(PICKS_JSON, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        print(f"  [historical_hits] ERROR reading picks_history.json: {exc}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING FEATURE ENGINE  (no lookahead — uses only prior game data)
# ─────────────────────────────────────────────────────────────────────────────

def build_rolling_features(h2h_df: pd.DataFrame) -> dict[str, dict]:
    """
    For every (team, date) pair, compute rolling L20 stats using only
    games that occurred BEFORE that date.  Returns a nested dict:

        features[team][date] = {win_rate, avg_r, avg_ra, n_games}

    Used to score each game without any lookahead.
    """
    if h2h_df.empty or "team" not in h2h_df.columns:
        return {}

    features: dict[str, dict] = {}

    for team, grp in h2h_df.groupby("team"):
        grp = grp.sort_values("date_parsed").reset_index(drop=True)
        features[team] = {}

        for i, row in grp.iterrows():
            target_date = row["date_parsed"]
            # strictly prior games only
            prior = grp[grp["date_parsed"] < target_date].tail(ROLLING_WINDOW)

            if prior.empty:
                features[team][target_date] = {
                    "win_rate": 0.50,
                    "avg_r":    4.5,
                    "avg_ra":   4.5,
                    "n_games":  0,
                }
                continue

            wins   = (prior["result"] == 1).sum()
            n      = len(prior)
            avg_r  = float(prior["r"].mean())  if "r"  in prior.columns else 4.5
            avg_ra = float(prior["ra"].mean()) if "ra" in prior.columns else 4.5

            features[team][target_date] = {
                "win_rate": wins / n if n > 0 else 0.50,
                "avg_r":    avg_r  if not np.isnan(avg_r)  else 4.5,
                "avg_ra":   avg_ra if not np.isnan(avg_ra) else 4.5,
                "n_games":  n,
            }

    return features


def score_game(home_feat: dict, away_feat: dict) -> float:
    """
    Simple edge score for the home team:
      edge = home_win_rate + (home_avg_r − away_avg_ra) / 20 − 0.50

    Positive = model prefers home team, negative = away team favored.
    """
    edge = (
        home_feat["win_rate"]
        + (home_feat["avg_r"] - away_feat["avg_ra"]) / 20.0
        - 0.50
    )
    return round(edge, 4)


# ─────────────────────────────────────────────────────────────────────────────
# REPORT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def report_from_picks_history(history: dict, lookback: int) -> pd.DataFrame:
    today = date.today()
    rows: list[dict] = []

    for i in range(1, lookback + 1):
        day   = (today - timedelta(days=i)).isoformat()
        entry = history.get(day)
        if not entry:
            continue

        picks      = entry.get("picks", [])
        recorded   = [p for p in picks if p.get("result") in ("WIN", "LOSS", "PUSH")]
        unrecorded = [p for p in picks if p.get("result") not in ("WIN", "LOSS", "PUSH")]

        if not recorded and unrecorded:
            rows.append({
                "date":     day,
                "picks":    len(unrecorded),
                "wins":     None,
                "losses":   None,
                "pushes":   None,
                "hit_rate": None,
                "avg_edge": round(
                    sum(p.get("edge", 0) for p in unrecorded) / len(unrecorded), 1
                ),
                "note":     "no results yet",
                "source":   "picks_history",
            })
            continue

        if not recorded:
            continue

        wins     = sum(1 for p in recorded if p["result"] == "WIN")
        losses   = sum(1 for p in recorded if p["result"] == "LOSS")
        pushes   = sum(1 for p in recorded if p["result"] == "PUSH")
        n        = wins + losses
        hit_rate = wins / n if n > 0 else None
        avg_edge = sum(p.get("edge", 0) for p in recorded) / len(recorded)

        rows.append({
            "date":     day,
            "picks":    len(recorded),
            "wins":     wins,
            "losses":   losses,
            "pushes":   pushes,
            "hit_rate": round(hit_rate * 100, 1) if hit_rate is not None else None,
            "avg_edge": round(avg_edge, 1),
            "note":     "",
            "source":   "picks_history",
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def report_from_h2h_rolling(h2h_df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """
    Retroactive top-5 prediction using rolling pre-game features.
    NO lookahead: features for date D use only games before D.
    """
    if h2h_df.empty:
        return pd.DataFrame()

    today  = date.today()
    cutoff = today - timedelta(days=lookback)

    # build rolling features across full history (pre-computes for every team×date)
    print(f"  [h2h] Building rolling L{ROLLING_WINDOW} features (this takes a moment)…")
    roll_feats = build_rolling_features(h2h_df)

    # games in window with known outcomes
    window = h2h_df[
        (h2h_df["date_parsed"] > cutoff) &
        (h2h_df["date_parsed"] < today) &
        (h2h_df["result"].notna())
    ].copy()

    if window.empty:
        print("  [h2h] No completed games found in the lookback window.")
        return pd.DataFrame()

    # opp column → away_team for feature lookup
    opp_col = next((c for c in ("opp", "away_team") if c in window.columns), None)
    team_col = "team" if "team" in window.columns else None
    if not team_col or not opp_col:
        print("  [h2h] Missing team/opp columns — cannot score games.")
        return pd.DataFrame()

    rows: list[dict] = []

    for day, grp in window.groupby("date_parsed"):
        scored: list[dict] = []

        for _, row in grp.iterrows():
            home = str(row[team_col])
            away = str(row[opp_col])

            home_feat = (roll_feats.get(home) or {}).get(day)
            away_feat = (roll_feats.get(away) or {}).get(day)

            # skip if no prior history for either team
            if not home_feat or not away_feat:
                continue
            if home_feat["n_games"] < 3 or away_feat["n_games"] < 3:
                continue

            edge = score_game(home_feat, away_feat)
            scored.append({
                "home":   home,
                "away":   away,
                "edge":   edge,
                "result": row["result"],
            })

        if not scored:
            continue

        # top 5 by predicted edge (no lookahead — edge computed from prior data)
        top5 = sorted(scored, key=lambda x: -x["edge"])[:5]
        wins    = sum(1 for g in top5 if g["result"] == 1)
        losses  = sum(1 for g in top5 if g["result"] == 0)
        n       = wins + losses
        hit_rate = wins / n if n > 0 else None
        avg_edge = sum(g["edge"] for g in top5) / len(top5) if top5 else 0

        top5_str = ", ".join(f"{g['home']}({g['result']})" for g in top5)

        rows.append({
            "date":     str(day),
            "picks":    len(top5),
            "wins":     wins,
            "losses":   losses,
            "pushes":   0,
            "hit_rate": round(hit_rate * 100, 1) if hit_rate is not None else None,
            "avg_edge": round(avg_edge * 100, 2),   # display as %
            "note":     top5_str[:40],
            "source":   "h2h_rolling",
        })

    return (
        pd.DataFrame(rows)
        .sort_values("date", ascending=False)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def print_report(df: pd.DataFrame, title: str) -> None:
    W = 80
    print()
    print("╔" + "═" * W + "╗")
    print(f"║  {title:<{W-2}}║")
    print("╠" + "═" * W + "╣")
    print(f"║  {'Date':<12}  {'Picks':>5}  {'W':>3}  {'L':>3}  {'Hit%':>7}  {'AvgEdge':>8}  {'Top Picks (home, result)':<18}║")
    print("╠" + "─" * W + "╣")

    if df.empty:
        print(f"║  {'No data available for this window.':<{W-2}}║")
        print("╚" + "═" * W + "╝")
        return

    total_w = total_l = total_picks = 0

    for _, row in df.iterrows():
        hit_str  = f"{row['hit_rate']:.1f}%"  if row.get("hit_rate")  is not None else "—"
        w_str    = str(int(row["wins"]))       if row.get("wins")      is not None else "—"
        l_str    = str(int(row["losses"]))     if row.get("losses")    is not None else "—"
        note     = str(row.get("note", ""))[:30]
        avg_edge = row.get("avg_edge") or 0

        print(
            f"║  {str(row['date']):<12}  {int(row['picks'] or 0):>5}  "
            f"{w_str:>3}  {l_str:>3}  {hit_str:>7}  {float(avg_edge):>7.2f}%  {note:<18}║"
        )

        if row.get("wins") is not None:
            total_w     += int(row["wins"]   or 0)
            total_l     += int(row["losses"] or 0)
            total_picks += int(row["picks"]  or 0)

    print("╠" + "─" * W + "╣")
    n = total_w + total_l
    overall_hr = f"{total_w / n * 100:.1f}%" if n > 0 else "—"
    print(
        f"║  {'OVERALL':<12}  {total_picks:>5}  "
        f"{total_w:>3}  {total_l:>3}  {overall_hr:>7}  {'':>8}  {'':18}║"
    )
    print("╚" + "═" * W + "╝")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retroactive MLB model hit-rate report (last N days)."
    )
    parser.add_argument("--days",   type=int, default=14,
                        help="Lookback window in days (default: 14).")
    parser.add_argument("--source", choices=["picks", "h2h", "both"], default="both",
                        help="Data source (default: both).")
    parser.add_argument("--csv",    action="store_true",
                        help="Write CSV to data/processed/historical_hits_report.csv.")
    args = parser.parse_args()

    print(f"\n[historical_hits] Lookback: {args.days} days  |  Source: {args.source}")
    print(f"  Today: {date.today().isoformat()}")

    all_frames: list[pd.DataFrame] = []

    # ── Source A: picks_history.json ─────────────────────────────────────────
    if args.source in ("picks", "both"):
        history  = load_picks_history()
        picks_df = report_from_picks_history(history, args.days)
        print_report(picks_df, f"Recorded Picks (picks_history.json) — last {args.days} days")
        if not picks_df.empty and picks_df["wins"].isna().any():
            print(
                "\n  ⚠  Some days have no results yet.  Record them with:\n"
                "    python src/MLB/generate_dashboard.py --record YYYY-MM-DD WIN LOSS WIN WIN PUSH"
            )
        if not picks_df.empty:
            all_frames.append(picks_df)

    # ── Source B: h2h rolling proxy ───────────────────────────────────────────
    if args.source in ("h2h", "both"):
        h2h_df   = load_h2h()
        rpt_df   = report_from_h2h_rolling(h2h_df, args.days)
        print_report(
            rpt_df,
            f"Retroactive Model (rolling L{ROLLING_WINDOW} prior-game features) — last {args.days} days"
        )
        print(
            "\n  Legend: AvgEdge = avg predicted home-team edge vs 0.50 baseline.\n"
            "  'Top Picks' shows home team + actual result (1=W, 0=L).\n"
            "  This is game-level — player prop accuracy requires picks_history results."
        )
        if not rpt_df.empty:
            all_frames.append(rpt_df)

    # ── CSV export ────────────────────────────────────────────────────────────
    if args.csv and all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        out      = PROCESSED_DIR / "historical_hits_report.csv"
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out, index=False)
        print(f"\n  → Report saved: {out}")


if __name__ == "__main__":
    main()