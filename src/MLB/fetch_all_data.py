"""
fetch_all_data.py
─────────────────────────────────────────────────────────────────────────────
Triple-stream MLB data ingestion.

  Stream 1 – Kalshi          : Historical KXMLBGAME candlesticks (2025 + 2026)
  Stream 2 – The Odds API    : Live player props (Outs, Ks, Total Bases, Hits)
  Stream 3 – pybaseball      : Team H2H schedule/results logs (2025 + 2026)

All outputs land in data/raw/:
  kalshi_candles.json
  odds_props.json
  h2h_2025.csv  /  h2h_2026.csv

Auth note: Kalshi authentication (JWT / API-key header) is assumed to be
           already handled by the caller or a shared auth module.  This
           script only focuses on the data-handling logic as requested.

Usage
  python fetch_all_data.py
  python fetch_all_data.py --streams kalshi odds   # run specific streams
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv
import requests
from curl_cffi import requests as cffi_requests

# ── MONKEY PATCH 3.0 ─────────────────────────────────────────────────────────
# Globally intercept requests to route B-Ref traffic through curl_cffi bypass
_original_get = requests.get
_original_session_request = requests.Session.request

def smart_get(url, **kwargs):
    if isinstance(url, str) and "baseball-reference.com" in url:
        kwargs.pop("timeout", None) # curl_cffi handles timeouts differently
        return cffi_requests.get(url, impersonate="chrome120", **kwargs)
    return _original_get(url, **kwargs)

def smart_session_request(self, method, url, **kwargs):
    if isinstance(url, str) and "baseball-reference.com" in url and method.upper() == "GET":
        kwargs.pop("timeout", None)
        return cffi_requests.get(url, impersonate="chrome120", **kwargs)
    return _original_session_request(self, method, url, **kwargs)

# Overwrite the standard library functions before pybaseball initializes
requests.get = smart_get
requests.Session.request = smart_session_request
# ─────────────────────────────────────────────────────────────────────────────
# ── env ──────────────────────────────────────────────────────────────────────
ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

KALSHI_API_KEY   = os.getenv("KALSHI_API_KEY", "")
KALSHI_BASE_URL  = os.getenv("KALSHI_BASE_URL", "https://trading-api.kalshi.com/trade-api/v2")
ODDS_API_KEY     = os.getenv("ODDS_API_KEY", "")
ODDS_BASE_URL    = os.getenv("ODDS_BASE_URL", "https://api.the-odds-api.com/v4")

# ── project paths ─────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
KALSHI_SERIES      = "KXMLBGAME"
KALSHI_START_2025  = "2025-03-20T00:00:00Z"
KALSHI_END_2025    = "2025-11-01T00:00:00Z"
KALSHI_START_2026  = "2026-01-01T00:00:00Z"
KALSHI_END_NOW     = None  

ODDS_SPORT         = "baseball_mlb"
ODDS_REGIONS       = "us"
ODDS_MARKETS       = "pitcher_strikeouts,batter_hits,batter_total_bases,pitcher_outs"
ODDS_BOOKMAKERS    = "draftkings,fanduel,betmgm,caesars"

PYBASEBALL_TEAMS   = [
    # Standard abbreviations ──────────────────────────────────────────────────
    "NYY", "BOS", "LAD", "CHC", "SFG", "HOU", "ATL", "PHI", "COL", "MIN",
    "NYM", "CLE", "SEA", "TEX", "DET", "MIL", "STL", "CIN", "PIT", "ARI",
    "MIA", "WSN", "BAL", "TOR",
    # Baseball-Reference / pybaseball URL slugs (differ from ESPN/statsapi) ──
    "ANA",   # Los Angeles Angels  (LAA on ESPN; ANA on B-Ref)
    "TBR",   # Tampa Bay Rays      (TB  on ESPN; TBR on B-Ref)
    "SDP",   # San Diego Padres    (SD  on ESPN; SDP on B-Ref)
    "KCR",   # Kansas City Royals  (KC  on ESPN; KCR on B-Ref)
    "CHW",   # Chicago White Sox   (CWS on ESPN; CHW on B-Ref)
    "OAK",   # Oakland Athletics
]

# ─────────────────────────────────────────────────────────────────────────────
# STREAM 1 — KALSHI
# ─────────────────────────────────────────────────────────────────────────────

from datetime import datetime, timezone

KALSHI_PUBLIC_URL = "https://api.elections.kalshi.com/trade-api/v2"

def _iso_to_unix(iso_str: Optional[str]) -> Optional[int]:
    """Helper to convert ISO-8601 strings to Unix epoch integers for Kalshi."""
    if not iso_str:
        return None
    dt = datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())

def _fetch_kalshi_markets(series_ticker: str) -> list[dict]:
    """Fetch markets using the public Elections Kalshi endpoint."""
    url = f"{KALSHI_PUBLIC_URL}/markets"
    params = {
        "series_ticker": series_ticker,
        "limit": 1000
    }
    
    try:
        resp = requests.get(url, headers=_kalshi_headers(), params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return data.get("markets", [])
    except Exception as e:
        print(f"    [Kalshi Error] Series fetch failed: {e}")
        return []

def _kalshi_headers() -> dict:
    return {"Accept": "application/json"}

def _fetch_kalshi_candles(
    market_ticker: str,
    period_interval: int = 60,  # minutes
    start_ts: Optional[int] = None, # Now expects an integer
    end_ts: Optional[int] = None,   # Now expects an integer
) -> list[dict]:
    """Fetch OHLCV candlestick data for a single market."""
    url = f"{KALSHI_PUBLIC_URL}/markets/{market_ticker}/candlesticks"
    
    params = {"period_interval": period_interval}
    if start_ts: params["start_ts"] = start_ts
    if end_ts: params["end_ts"] = end_ts

    resp = requests.get(url, headers=_kalshi_headers(), params=params, timeout=20)
    if resp.status_code == 404:
        return []
    
    resp.raise_for_status()
    return resp.json().get("candlesticks", [])

def fetch_kalshi(output_path: Path) -> None:
    """
    Pull candlesticks for every KXMLBGAME market across 2025 and 2026 season.
    Saves a JSON list to output_path.
    """
    print("[fetch_all_data] Stream 1 — Kalshi candlesticks…")
    markets = _fetch_kalshi_markets(KALSHI_SERIES)
    all_candles: list[dict] = []

    print(f"    Scanning {len(markets)} markets (Estimated time: {len(markets)*0.2/60:.1f} min)...")
    
    empty_count = 0
    for mkt in markets:
        ticker = mkt.get("ticker", "")
        close_time = mkt.get("close_time", "")

        # Convert the string constants to Unix integers
        if close_time.startswith("2025"):
            start = _iso_to_unix(KALSHI_START_2025)
            end = _iso_to_unix(KALSHI_END_2025)
        elif close_time.startswith("2026"):
            start = _iso_to_unix(KALSHI_START_2026)
            end = _iso_to_unix(KALSHI_END_NOW)
        else:
            continue   

        try:
            candles = _fetch_kalshi_candles(ticker, start_ts=start, end_ts=end)
        except requests.RequestException as exc:
            continue

        if not candles:
            empty_count += 1
        else:
            for c in candles:
                c["market_ticker"] = ticker
                c["series"] = KALSHI_SERIES
            
            all_candles.extend(candles)
            # Only print when we actually capture data to avoid console spam
            print(f"    ✓ {ticker} -> Captured {len(candles)} candles")
            
        time.sleep(0.2)

    print(f"    Skipped {empty_count} markets with zero trades/volume.")
    output_path.write_text(json.dumps(all_candles, indent=2))
    print(f"  → {len(all_candles)} total candles saved to {output_path}")

# ─────────────────────────────────────────────────────────────────────────────
# STREAM 2 — THE ODDS API  (player props)
# ─────────────────────────────────────────────────────────────────────────────

def _get_live_events() -> list[dict]:
    """
    Return full event objects (id, home_team, away_team, commence_time) for
    all upcoming MLB games.  We need home_team here so it can be stored in
    the JSON and later joined with weather data in process_model.py.
    """
    if not ODDS_API_KEY:
        raise EnvironmentError("ODDS_API_KEY is not set in .env")
    url    = f"{ODDS_BASE_URL}/sports/{ODDS_SPORT}/events"
    params = {"apiKey": ODDS_API_KEY, "dateFormat": "iso"}
    resp   = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    events = resp.json()
    print(f"    Found {len(events)} upcoming MLB events")
    return events   # full dicts, not just IDs


def _fetch_event_player_props(event_id: str) -> tuple[list[dict], str, str]:
    """
    Fetch player props for a single event.
    Returns (bookmakers, home_team, away_team).
    The /events/{id}/odds endpoint includes home_team at the top level,
    so we capture it here rather than making a separate lookup.
    """
    url    = f"{ODDS_BASE_URL}/sports/{ODDS_SPORT}/events/{event_id}/odds"
    params = {
        "apiKey":      ODDS_API_KEY,
        "regions":     ODDS_REGIONS,
        "markets":     ODDS_MARKETS,
        "bookmakers":  ODDS_BOOKMAKERS,
        "dateFormat":  "iso",
        "oddsFormat":  "american",
    }
    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code == 422:
        # market not listed for this event — normal during pre-game window
        return [], "", ""
    resp.raise_for_status()
    body = resp.json()
    return (
        body.get("bookmakers", []),
        body.get("home_team",  ""),   # e.g. "Kansas City Royals"
        body.get("away_team",  ""),
    )


def fetch_odds(output_path: Path) -> None:
    """
    Pull live player props for all upcoming MLB games.
    Stores home_team + away_team alongside bookmakers so process_model.py
    can apply wind/park multipliers per stadium.
    Saves structured JSON to output_path.
    """
    print("[fetch_all_data] Stream 2 — The Odds API player props…")
    events    = _get_live_events()
    all_props: list[dict] = []

    for evt in events:
        eid = evt["id"]
        # Prefer the full team names from the /events endpoint (already have them)
        home_from_events = evt.get("home_team", "")
        away_from_events = evt.get("away_team", "")

        print(f"    {home_from_events} vs {away_from_events}", end=" ", flush=True)
        try:
            bookmakers, home_from_odds, away_from_odds = _fetch_event_player_props(eid)
        except requests.RequestException as exc:
            print(f"[ERR: {exc}]")
            continue

        # /events/odds also returns home_team; fall back to /events value if empty
        home_team = home_from_odds or home_from_events
        away_team = away_from_odds or away_from_events

        all_props.append({
            "event_id":       eid,
            "home_team":      home_team,   # full name, e.g. "Kansas City Royals"
            "away_team":      away_team,
            "commence_time":  evt.get("commence_time", ""),
            "bookmakers":     bookmakers,
        })
        print(f"({len(bookmakers)} bookmakers)")
        time.sleep(0.15)

    output_path.write_text(json.dumps(all_props, indent=2))
    print(f"  → Props for {len(all_props)} events saved to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# STREAM 3 — pybaseball  (team H2H schedule/game logs)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_team_schedule(team: str, season: int) -> pd.DataFrame:
    """
    Use pybaseball's schedule_and_record() to get a team's season log.
    Returns a DataFrame with one row per game.
    """
    try:
        import pybaseball as pb
    except ImportError:
        raise ImportError(
            "pybaseball is not installed.  Run:  pip install pybaseball"
        )

    pb.cache.enable()
    df = pb.schedule_and_record(season, team)
    df["team"]   = team
    df["season"] = season
    return df


def _build_h2h(season: int) -> pd.DataFrame:
    """
    Fetch schedule/record for all configured teams for a given season and
    produce an H2H matrix: each row = a game between two teams.
    """
    frames: list[pd.DataFrame] = []
    for team in PYBASEBALL_TEAMS:
        print(f"    {team}/{season}", end=" ", flush=True)
        try:
            df = _fetch_team_schedule(team, season)
            frames.append(df)
            print(f"({len(df)} games)")
            time.sleep(6.5)   # stay under Baseball-Reference rate limit
        except Exception as exc:
            print(f"[ERR: {exc}]")
            time.sleep(10)    # back off longer on a rate-limit or fetch error

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # normalise column names to lower snake_case
    combined.columns = (
        combined.columns.str.strip()
        .str.lower()
        .str.replace(r"[\s/]+", "_", regex=True)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )

    # keep only completed games
    if "w_l" in combined.columns:
        combined = combined[combined["w_l"].isin(["W", "L"])]

    return combined


def fetch_pybaseball(output_dir: Path) -> None:
    """
    Build and save H2H CSVs for 2025 and 2026 (YTD).
    """
    print("[fetch_all_data] Stream 3 — pybaseball H2H logs…")
    current_year = 2026

    for season in [2025, current_year]:
        print(f"  Season {season}:")
        df = _build_h2h(season)
        if df.empty:
            print(f"    No data returned for {season}.")
            continue
        out_path = output_dir / f"h2h_{season}.csv"
        df.to_csv(out_path, index=False)
        print(f"  → {len(df)} rows saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

STREAM_MAP = {
    "kalshi":     lambda: fetch_kalshi(RAW_DIR / "kalshi_candles.json"),
    "odds":       lambda: fetch_odds(RAW_DIR / "odds_props.json"),
    "pybaseball": lambda: fetch_pybaseball(RAW_DIR),
}


def main(streams: list[str]) -> None:
    errors: list[str] = []
    for name in streams:
        fn = STREAM_MAP.get(name)
        if fn is None:
            print(f"[fetch_all_data] Unknown stream '{name}' — skipping.")
            continue
        try:
            fn()
        except EnvironmentError as exc:
            print(f"[fetch_all_data] Config error in '{name}': {exc}")
            errors.append(name)
        except Exception as exc:
            print(f"[fetch_all_data] Unexpected error in '{name}': {exc}")
            errors.append(name)

    if errors:
        print(f"\n[fetch_all_data] ⚠  Streams with errors: {errors}")
    else:
        print("\n[fetch_all_data] ✓  All streams completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest MLB data from three streams.")
    parser.add_argument(
        "--streams",
        nargs="+",
        default=list(STREAM_MAP.keys()),
        choices=list(STREAM_MAP.keys()),
        help="Which streams to run (default: all three).",
    )
    args = parser.parse_args()
    main(args.streams)