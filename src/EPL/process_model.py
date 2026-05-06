#!/usr/bin/env python3
"""
process_model.py
─────────────────────────────────────────────────────────────────────────────
Premier League prediction / edge model.

Input:
  data/raw/epl/kalshi_epl_markets.json
  data/raw/epl/epl_fixtures.csv
  data/raw/epl/epl_h2h.json

Output:
  data/processed/epl/golden.csv
  data/processed/epl/summary.json
  data/processed/epl/dashboard.html

Basic idea:
  market probability = Kalshi YES price
  model probability  = blend of Kalshi + recent form + H2H + goal differential
  edge               = model probability - market probability

This is intentionally simple and readable so it matches the MLB project format
without pretending to be a fully trained betting model.
"""

import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "data" / "raw" / "epl"
PROCESSED_DIR = ROOT / "data" / "processed" / "epl"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TEAM_ALIASES = {
    "Arsenal": ["arsenal"],
    "Aston Villa": ["aston villa", "villa"],
    "Bournemouth": ["bournemouth", "afc bournemouth"],
    "Brentford": ["brentford"],
    "Brighton": ["brighton", "brighton & hove albion", "brighton and hove albion"],
    "Burnley": ["burnley"],
    "Chelsea": ["chelsea"],
    "Crystal Palace": ["crystal palace", "palace"],
    "Everton": ["everton"],
    "Fulham": ["fulham"],
    "Leeds": ["leeds", "leeds united"],
    "Liverpool": ["liverpool"],
    "Manchester City": ["manchester city", "man city", "mancity"],
    "Manchester United": ["manchester united", "man united", "man utd", "manutd"],
    "Newcastle": ["newcastle", "newcastle united"],
    "Nottingham Forest": ["nottingham forest", "forest"],
    "Sunderland": ["sunderland"],
    "Tottenham": ["tottenham", "tottenham hotspur", "spurs"],
    "West Ham": ["west ham", "west ham united"],
    "Wolves": ["wolves", "wolverhampton", "wolverhampton wanderers"],
}


def normalize_text(text: str) -> str:
    text = text.lower().replace("&", "and")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def teams_in_text(text: str) -> list[str]:
    norm = normalize_text(text)
    found = []
    for team, aliases in TEAM_ALIASES.items():
        if any(alias in norm for alias in aliases):
            found.append(team)
    return list(dict.fromkeys(found))


def load_inputs() -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    kalshi_path = RAW_DIR / "kalshi_epl_markets.json"
    fixtures_path = RAW_DIR / "epl_fixtures.csv"
    h2h_path = RAW_DIR / "epl_h2h.json"

    kalshi = json.loads(kalshi_path.read_text()) if kalshi_path.exists() else []
    fixtures = pd.read_csv(fixtures_path) if fixtures_path.exists() else pd.DataFrame()
    h2h = json.loads(h2h_path.read_text()) if h2h_path.exists() else {}
    return kalshi, fixtures, h2h


def clean_finished_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    if fixtures.empty:
        return fixtures
    df = fixtures.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    df = df[df["home_goals"].notna() & df["away_goals"].notna()].copy()
    df = df.sort_values("date")
    return df


def team_match_rows(fixtures: pd.DataFrame, team: str) -> pd.DataFrame:
    if fixtures.empty:
        return pd.DataFrame()
    norm_team = normalize_text(team)
    mask = fixtures["home_team"].astype(str).apply(lambda x: norm_team in normalize_text(x) or normalize_text(x) in norm_team) | \
           fixtures["away_team"].astype(str).apply(lambda x: norm_team in normalize_text(x) or normalize_text(x) in norm_team)
    return fixtures[mask].copy()


def recent_form_score(fixtures: pd.DataFrame, team: str, n: int = 10) -> dict[str, float]:
    rows = team_match_rows(fixtures, team).tail(n)
    if rows.empty:
        return {"points_per_match": 1.35, "goal_diff_per_match": 0.0, "win_rate": 0.33}

    points = 0
    wins = 0
    gd = 0
    for _, r in rows.iterrows():
        is_home = normalize_text(team) in normalize_text(str(r["home_team"]))
        gf = float(r["home_goals"] if is_home else r["away_goals"])
        ga = float(r["away_goals"] if is_home else r["home_goals"])
        gd += gf - ga
        if gf > ga:
            points += 3
            wins += 1
        elif gf == ga:
            points += 1
    games = max(1, len(rows))
    return {
        "points_per_match": points / games,
        "goal_diff_per_match": gd / games,
        "win_rate": wins / games,
    }


def parse_h2h_games(h2h_payload: list[dict[str, Any]], team_a: str, team_b: str) -> dict[str, float]:
    if not h2h_payload:
        return {"h2h_a_win_rate": 0.33, "h2h_b_win_rate": 0.33, "h2h_draw_rate": 0.34, "h2h_games": 0}

    a_wins = b_wins = draws = games = 0
    for item in h2h_payload:
        teams = item.get("teams", {})
        goals = item.get("goals", {})
        home = teams.get("home", {}).get("name", "")
        away = teams.get("away", {}).get("name", "")
        hg = goals.get("home")
        ag = goals.get("away")
        if hg is None or ag is None:
            continue
        games += 1
        home_is_a = normalize_text(team_a) in normalize_text(home) or normalize_text(home) in normalize_text(team_a)
        away_is_a = normalize_text(team_a) in normalize_text(away) or normalize_text(away) in normalize_text(team_a)
        if hg == ag:
            draws += 1
        elif (hg > ag and home_is_a) or (ag > hg and away_is_a):
            a_wins += 1
        else:
            b_wins += 1

    if games == 0:
        return {"h2h_a_win_rate": 0.33, "h2h_b_win_rate": 0.33, "h2h_draw_rate": 0.34, "h2h_games": 0}
    return {
        "h2h_a_win_rate": a_wins / games,
        "h2h_b_win_rate": b_wins / games,
        "h2h_draw_rate": draws / games,
        "h2h_games": games,
    }


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def model_probability(market_prob: float, team_a_form: dict[str, float], team_b_form: dict[str, float], h2h_stats: dict[str, float]) -> float:
    """
    Estimate Team A win probability.

    The model keeps Kalshi as the anchor but adjusts it with soccer history:
      - recent points per match
      - recent goal difference
      - H2H win rate

    Weighting is conservative so one small H2H sample does not dominate.
    """
    form_edge = (team_a_form["points_per_match"] - team_b_form["points_per_match"]) / 3.0
    gd_edge = np.tanh((team_a_form["goal_diff_per_match"] - team_b_form["goal_diff_per_match"]) / 2.0)
    h2h_edge = h2h_stats["h2h_a_win_rate"] - h2h_stats["h2h_b_win_rate"]

    history_prob = sigmoid((1.35 * form_edge) + (0.85 * gd_edge) + (0.55 * h2h_edge))

    # Blend: Kalshi market is useful, but the point of the project is to add history.
    blended = (0.55 * market_prob) + (0.45 * history_prob)
    return float(np.clip(blended, 0.02, 0.98))


def build_golden() -> pd.DataFrame:
    kalshi, fixtures_raw, h2h = load_inputs()
    fixtures = clean_finished_fixtures(fixtures_raw)

    rows = []
    for market in kalshi:
        title = " ".join(str(market.get(k, "")) for k in ["title", "subtitle", "ticker", "event_ticker"])
        teams = market.get("teams_found") or teams_in_text(title)
        if len(teams) < 2:
            continue

        team_a, team_b = teams[0], teams[1]
        market_prob = float(market.get("yes_prob", 0.5))
        team_a_form = recent_form_score(fixtures, team_a)
        team_b_form = recent_form_score(fixtures, team_b)

        h2h_key = next((k for k in h2h.keys() if team_a in k and team_b in k), f"{team_a} vs {team_b}")
        h2h_stats = parse_h2h_games(h2h.get(h2h_key, []), team_a, team_b)

        model_prob = model_probability(market_prob, team_a_form, team_b_form, h2h_stats)
        edge = model_prob - market_prob

        rows.append({
            "market_ticker": market.get("ticker", ""),
            "market_title": market.get("title", ""),
            "team_a": team_a,
            "team_b": team_b,
            "prediction": f"{team_a} YES / wins market",
            "market_prob": round(market_prob, 4),
            "model_prob": round(model_prob, 4),
            "edge": round(edge, 4),
            "edge_pct": round(edge * 100, 2),
            "team_a_ppm_l10": round(team_a_form["points_per_match"], 3),
            "team_b_ppm_l10": round(team_b_form["points_per_match"], 3),
            "team_a_gd_l10": round(team_a_form["goal_diff_per_match"], 3),
            "team_b_gd_l10": round(team_b_form["goal_diff_per_match"], 3),
            "h2h_games": h2h_stats["h2h_games"],
            "h2h_team_a_win_rate": round(h2h_stats["h2h_a_win_rate"], 3),
            "h2h_team_b_win_rate": round(h2h_stats["h2h_b_win_rate"], 3),
            "h2h_draw_rate": round(h2h_stats["h2h_draw_rate"], 3),
            "kalshi_volume": market.get("volume", 0),
            "close_time": market.get("close_time", ""),
            "source": "Kalshi + API-Football",
        })

    columns = [
        "rank", "market_ticker", "market_title", "team_a", "team_b", "prediction",
        "market_prob", "model_prob", "edge", "edge_pct", "team_a_ppm_l10",
        "team_b_ppm_l10", "team_a_gd_l10", "team_b_gd_l10", "h2h_games",
        "h2h_team_a_win_rate", "h2h_team_b_win_rate", "h2h_draw_rate",
        "kalshi_volume", "close_time", "source",
    ]
    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(columns=columns)

# 🔥 REMOVE DUPLICATE GAMES (this is the fix)
    df = df.sort_values(
        by=["team_a", "team_b", "kalshi_volume"],
        ascending=[True, True, False]
    )

    df = df.drop_duplicates(
        subset=["team_a", "team_b"],
        keep="first"
    )

# Now rank after deduplication
    df = df.sort_values("edge", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)

    return df


def main() -> None:
    golden = build_golden()
    golden_path = PROCESSED_DIR / "golden.csv"
    summary_path = PROCESSED_DIR / "summary.json"

    golden.to_csv(golden_path, index=False)

    summary = {
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "total_markets": int(len(golden)),
        "positive_edges": int((golden["edge"] > 0).sum()) if not golden.empty else 0,
        "top_edges": golden.head(10).to_dict(orient="records") if not golden.empty else [],
        "notes": "Model probability blends Kalshi YES price with API-Football recent EPL form and H2H history.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 70)
    print("EPL EDGE MODEL")
    print("=" * 70)
    if golden.empty:
        print("No EPL Kalshi markets were matched. Try again when Kalshi has open EPL markets, or set KALSHI_EPL_SERIES in .env if you know the exact series ticker.")
    else:
        print(golden[["rank", "team_a", "team_b", "market_prob", "model_prob", "edge_pct", "h2h_games"]].head(10).to_string(index=False))
    print(f"\nSaved → {golden_path}")
    print(f"Saved → {summary_path}")

    # Match MLB workflow: generate dashboard after processing.
    dash_script = Path(__file__).resolve().parent / "generate_dashboard.py"
    if dash_script.exists():
        subprocess.run([sys.executable, str(dash_script)], check=False)


if __name__ == "__main__":
    main()
