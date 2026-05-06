#!/usr/bin/env python3
"""
fetch_all_data.py
─────────────────────────────────────────────────────────────────────────────
Premier League data ingestion.

This is the EPL version of the MLB fetch script. It replaces The Odds API with:

  Stream 1 – Kalshi       : Open prediction markets related to Premier League teams
  Stream 2 – API-Football : EPL fixtures/results for the last two seasons
  Stream 3 – API-Football : Head-to-head history for matched Kalshi teams

Outputs land in data/raw/epl/:
  kalshi_epl_markets.json
  epl_fixtures.csv
  epl_h2h.json

Usage:
  python3 src/EPL/fetch_all_data.py
  python3 src/EPL/fetch_all_data.py --streams kalshi fixtures h2h

.env needed:
  API_FOOTBALL_KEY=your_api_football_key

Optional:
  KALSHI_BASE_URL=https://api.elections.kalshi.com/trade-api/v2
  KALSHI_EPL_SERIES=optional_series_ticker_if_you_find_one
"""

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

RAW_DIR = ROOT / "data" / "raw" / "epl"
RAW_DIR.mkdir(parents=True, exist_ok=True)

API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")
API_FOOTBALL_BASE_URL = os.getenv("API_FOOTBALL_BASE_URL", "https://v3.football.api-sports.io")

KALSHI_BASE_URL = os.getenv("KALSHI_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2").rstrip("/")
KALSHI_EPL_SERIES = os.getenv("KALSHI_EPL_SERIES", "").strip()

EPL_LEAGUE_ID = 39

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


def current_epl_season() -> int:
    """API-Football uses the starting year for a season, e.g. 2025 for 2025-26."""
    today = datetime.now()
    return today.year if today.month >= 8 else today.year - 1


def season_window() -> list[int]:
    season = current_epl_season()
    return [season - 1, season]


def api_football_headers() -> dict[str, str]:
    return {"x-apisports-key": API_FOOTBALL_KEY}


def kalshi_headers() -> dict[str, str]:
    return {"Accept": "application/json"}


def get_json(url: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> dict[str, Any]:
    resp = requests.get(url, params=params or {}, headers=headers or {}, timeout=25)
    resp.raise_for_status()
    return resp.json()


def normalize_title(text: str) -> str:
    text = text.lower()
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def teams_in_text(text: str) -> list[str]:
    norm = normalize_title(text)
    found = []
    for team, aliases in TEAM_ALIASES.items():
        if any(alias in norm for alias in aliases):
            found.append(team)
    return list(dict.fromkeys(found))


def extract_kalshi_price(market: dict[str, Any]) -> float | None:
    """Return a YES price on a 0-1 scale from the fields Kalshi usually returns."""
    for key in ["yes_ask_dollars", "yes_bid_dollars", "last_price_dollars"]:
        val = market.get(key)
        if val is not None:
            try:
                return max(0.01, min(0.99, float(val)))
            except (TypeError, ValueError):
                pass

    for key in ["yes_ask", "yes_bid", "last_price"]:
        val = market.get(key)
        if val is not None:
            try:
                val = float(val)
                return max(0.01, min(0.99, val / 100 if val > 1 else val))
            except (TypeError, ValueError):
                pass
    return None


def fetch_kalshi(output_path: Path) -> list[dict[str, Any]]:
    print("[EPL fetch] Stream 1 — Kalshi EPL upcoming game markets…")

    url = f"{KALSHI_BASE_URL}/markets"
    markets: list[dict[str, Any]] = []
    cursor = None

    while True:
        params: dict[str, Any] = {
            "status": "open",
            "limit": 1000,
            "series_ticker": KALSHI_EPL_SERIES or "KXEPLGAME",
        }

        if cursor:
            params["cursor"] = cursor

        try:
            payload = get_json(url, params=params, headers=kalshi_headers())
        except Exception as exc:
            print(f"  [Kalshi Error] {exc}")
            output_path.write_text(json.dumps([], indent=2), encoding="utf-8")
            return []

        markets.extend(payload.get("markets", []))
        cursor = payload.get("cursor")

        if not cursor:
            break

        time.sleep(0.25)

    filtered = []

    for m in markets:
        title = " ".join(
            str(m.get(k, ""))
            for k in ["title", "subtitle", "yes_sub_title", "no_sub_title", "event_ticker", "ticker"]
        )

        found_teams = teams_in_text(title)

        if len(found_teams) < 2:
            continue

        yes_prob = extract_kalshi_price(m)

        if yes_prob is None:
            continue

        filtered.append({
            "ticker": m.get("ticker"),
            "event_ticker": m.get("event_ticker"),
            "title": m.get("title", ""),
            "subtitle": m.get("subtitle", ""),
            "yes_sub_title": m.get("yes_sub_title", ""),
            "no_sub_title": m.get("no_sub_title", ""),
            "status": m.get("status", ""),
            "close_time": m.get("close_time", ""),
            "open_time": m.get("open_time", ""),
            "volume": m.get("volume", 0),
            "yes_prob": yes_prob,
            "yes_ask": m.get("yes_ask"),
            "yes_bid": m.get("yes_bid"),
            "teams_found": found_teams[:2],
        })

    output_path.write_text(json.dumps(filtered, indent=2), encoding="utf-8")
    print(f"  saved {len(filtered)} Kalshi EPL upcoming game markets → {output_path}")
    return filtered


def fetch_fixtures(output_path: Path) -> pd.DataFrame:
    print("[EPL fetch] Stream 2 — API-Football EPL fixtures/results for last two seasons…")
    if not API_FOOTBALL_KEY:
        print("  [API-Football Error] Missing API_FOOTBALL_KEY in .env")
        pd.DataFrame().to_csv(output_path, index=False)
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for season in season_window():
        params = {"league": EPL_LEAGUE_ID, "season": season}
        try:
            payload = get_json(f"{API_FOOTBALL_BASE_URL}/fixtures", params=params, headers=api_football_headers())
        except Exception as exc:
            print(f"  [API-Football Error] fixtures season={season}: {exc}")
            continue

        for item in payload.get("response", []):
            fixture = item.get("fixture", {})
            teams = item.get("teams", {})
            goals = item.get("goals", {})
            league = item.get("league", {})
            status = fixture.get("status", {})
            home = teams.get("home", {})
            away = teams.get("away", {})

            rows.append({
                "fixture_id": fixture.get("id"),
                "season": league.get("season", season),
                "date": fixture.get("date"),
                "timestamp": fixture.get("timestamp"),
                "round": league.get("round", ""),
                "status_short": status.get("short", ""),
                "status_long": status.get("long", ""),
                "home_id": home.get("id"),
                "home_team": home.get("name", ""),
                "away_id": away.get("id"),
                "away_team": away.get("name", ""),
                "home_goals": goals.get("home"),
                "away_goals": goals.get("away"),
                "venue": fixture.get("venue", {}).get("name", ""),
            })
        print(f"  season {season}: {len(payload.get('response', []))} fixtures")
        time.sleep(0.3)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  saved {len(df)} fixtures → {output_path}")
    return df


def team_id_lookup(fixtures: pd.DataFrame) -> dict[str, int]:
    lookup: dict[str, int] = {}
    if fixtures.empty:
        return lookup
    for _, row in fixtures.iterrows():
        if pd.notna(row.get("home_id")):
            lookup[str(row.get("home_team"))] = int(row.get("home_id"))
        if pd.notna(row.get("away_id")):
            lookup[str(row.get("away_team"))] = int(row.get("away_id"))
    return lookup


def resolve_team_id(team_name: str, lookup: dict[str, int]) -> int | None:
    norm = normalize_title(team_name)
    for api_name, team_id in lookup.items():
        api_norm = normalize_title(api_name)
        aliases = TEAM_ALIASES.get(team_name, [team_name.lower()])
        if norm == api_norm or any(normalize_title(alias) in api_norm or api_norm in normalize_title(alias) for alias in aliases):
            return team_id
    return None


def fetch_h2h(output_path: Path, kalshi_markets: list[dict[str, Any]], fixtures: pd.DataFrame) -> dict[str, Any]:
    print("[EPL fetch] Stream 3 — API-Football head-to-head for Kalshi matchups…")

    if not API_FOOTBALL_KEY:
        output_path.write_text(json.dumps({}, indent=2), encoding="utf-8")
        return {}

    lookup = team_id_lookup(fixtures)
    result: dict[str, Any] = {}

    # Build unique matchups FIRST
    unique_matchups = {}

    for market in kalshi_markets:
        teams = market.get("teams_found", [])

        if len(teams) < 2:
            continue

        team_a = teams[0]
        team_b = teams[1]

        # This makes Liverpool-Man United and Man United-Liverpool count as same game
        matchup_id = " vs ".join(sorted([team_a, team_b]))

        if matchup_id not in unique_matchups:
            unique_matchups[matchup_id] = (team_a, team_b)

    print(f"  found {len(unique_matchups)} unique H2H matchups")

    # Now call API only once per unique matchup
    for matchup_id, (team_a, team_b) in unique_matchups.items():
        id_a = resolve_team_id(team_a, lookup)
        id_b = resolve_team_id(team_b, lookup)

        if not id_a or not id_b:
            print(f"  [H2H Skip] Could not resolve IDs for {team_a} vs {team_b}")
            continue

        params = {"h2h": f"{id_a}-{id_b}"}

        try:
            payload = get_json(
                f"{API_FOOTBALL_BASE_URL}/fixtures/headtohead",
                params=params,
                headers=api_football_headers()
            )

            result[matchup_id] = payload.get("response", [])
            print(f"  {matchup_id}: {len(result[matchup_id])} H2H fixtures")

            time.sleep(6)

        except Exception as exc:
            print(f"  [H2H Error] {matchup_id}: {exc}")
            result[matchup_id] = []

    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  saved H2H data → {output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", nargs="*", default=["kalshi", "fixtures", "h2h"], choices=["kalshi", "fixtures", "h2h"])
    args = parser.parse_args()

    kalshi_path = RAW_DIR / "kalshi_epl_markets.json"
    fixtures_path = RAW_DIR / "epl_fixtures.csv"
    h2h_path = RAW_DIR / "epl_h2h.json"

    if "kalshi" in args.streams:
        kalshi = fetch_kalshi(kalshi_path)
    else:
        kalshi = json.loads(kalshi_path.read_text()) if kalshi_path.exists() else []

    if "fixtures" in args.streams:
        fixtures = fetch_fixtures(fixtures_path)
    else:
        fixtures = pd.read_csv(fixtures_path) if fixtures_path.exists() else pd.DataFrame()

    if "h2h" in args.streams:
        fetch_h2h(h2h_path, kalshi, fixtures)


if __name__ == "__main__":
    main()
