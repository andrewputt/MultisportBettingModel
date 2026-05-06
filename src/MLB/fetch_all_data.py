"""
fetch_all_data.py
─────────────────────────────────────────────────────────────────────────────
Triple-stream MLB data ingestion.

  Stream 1 – Kalshi          : Today's KXMLBGAME open markets (yes_ask prob)
  Stream 2 – The Odds API    : Live player props (Outs, Ks, Total Bases, Hits)
  Stream 3 – pybaseball      : Team H2H schedule/results logs (2025 + 2026)

All outputs land in data/raw/:
  kalshi_today.json
  odds_props.json
  h2h_2025.csv  /  h2h_2026.csv

Usage
  python fetch_all_data.py
  python fetch_all_data.py --streams kalshi odds   # run specific streams
"""

import argparse
import io as _io
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv
import numpy as np
import pybaseball
from curl_cffi.requests import Session as CurlSession

# ── MONKEY PATCH 1: HTTPS enforcement + Cloudflare bypass ────────────────────
#
# TWO problems fixed here:
#
#   Problem A — Port 80 blocked:
#     pybaseball generates http:// URLs for Baseball-Reference (port 80).
#     Many networks block outbound HTTP; the OS raises errno 50 (ENETDOWN)
#     before any redirect to HTTPS can happen.  We must rewrite the URL to
#     https:// *before* the connection attempt.
#
#   Problem B — Wrong patch target (previous approach):
#     Setting `pybaseball.datasources.bref.requests = session_instance` only
#     replaces a module attribute that bref.py never reads after import.
#     The bref class already has `self.session = requests.Session()` baked in
#     at instantiation time.  The correct target is the `.session` attribute
#     on the already-created BRefSession instance:
#         pybaseball.datasources.bref.session.session  ← inner requests.Session
#
# Fix: replace the inner requests.Session with a thin wrapper that:
#   1. Rewrites http://www.baseball-reference.com → https://...
#   2. Uses curl_cffi (Chrome 110 impersonation) to bypass Cloudflare WAF

_cffi = CurlSession(impersonate="chrome110")
_cffi.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/110.0.0.0 Safari/537.36"
    )
})

class _BRefHttpsSession:
    """Drop-in replacement for requests.Session inside pybaseball's BRefSession.
    Forces https:// on every B-Ref request and routes through curl_cffi."""

    def get(self, url: str, **kwargs) -> object:
        url = url.replace(
            "http://www.baseball-reference.com",
            "https://www.baseball-reference.com",
            1,
        )
        kwargs.pop("verify", None)   # curl_cffi handles TLS natively
        return _cffi.get(url, **kwargs)

    def __getattr__(self, name: str) -> object:
        return getattr(_cffi, name)

# The inner requests.Session lives at:
#   pybaseball.team_results  (module)
#     .session               (BRefSession instance created at import time)
#       .session             (requests.Session — THIS is what we replace)
import pybaseball.team_results as _tr
_tr.session.session = _BRefHttpsSession()

# ── MONKEY PATCH 2: Pandas 3.0 Copy-on-Write compatibility ───────────────────
# pybaseball/team_results.py does:
#   df['Attendance'].replace(..., inplace=True)   ← silent NO-OP in Pandas 3.0
#   df['Attendance'] = df['Attendance'].astype(float)  ← ValueError: 'Unknown'
#
# Wrapping the return value doesn't help because the original RAISES before
# returning. We must replace the function entirely with a CoW-safe version.
# It uses the same _tr.session (already patched above for HTTPS + curl_cffi).

def _cow_safe_schedule_and_record(season: int, team: str) -> pd.DataFrame:
    """
    Drop-in replacement for pybaseball.team_results.schedule_and_record.

    Replicates B-Ref table parsing with CoW-safe attendance cleaning.
    Uses _tr.session so the HTTPS rewrite + curl_cffi bypass applies.
    """
    url = (
        f"http://www.baseball-reference.com/teams/{team}"
        f"/{season}-schedule-scores.shtml"
    )
    html_content = _tr.session.get(url).content
    tables = pd.read_html(_io.BytesIO(html_content), flavor=["lxml", "html.parser"])
    df = tables[0].copy()                    # own the buffer from the start

    # Strip asterisks from column names (B-Ref uses them for notes)
    df.columns = [str(c).replace("*", "") for c in df.columns]

    # Remove repeated header rows B-Ref embeds mid-table every ~20 rows.
    # The repeating row has "Gm#" in the Gm column (or "Gm#" column itself).
    for _gc in ("Gm", "Gm#"):
        if _gc in df.columns:
            df = df[df[_gc].astype(str) != "Gm#"].reset_index(drop=True)
            break

    # CoW-safe attendance cleaning — avoids inplace + astype(float) pattern
    if "Attendance" in df.columns:
        att = df["Attendance"].astype(str)            # detached copy
        att = att.where(att != "Unknown", other=pd.NA)
        att = att.str.replace(",", "", regex=False)
        df["Attendance"] = pd.to_numeric(att, errors="coerce")

    return df

_tr.schedule_and_record        = _cow_safe_schedule_and_record
pybaseball.schedule_and_record = _cow_safe_schedule_and_record

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

ODDS_SPORT         = "baseball_mlb"
ODDS_REGIONS       = "us"
# pitcher_outs_recorded is NOT a valid Odds API market key — it causes a 422
# for the entire request (the API rejects the call rather than ignoring the bad key).
# Valid MLB player-prop markets confirmed against the live API:
#   pitcher_strikeouts, batter_total_bases, batter_hits
# pitcher_outs is intentionally excluded: the API returns it inconsistently.
ODDS_MARKETS       = "pitcher_strikeouts,batter_total_bases,batter_hits"
ODDS_BOOKMAKERS    = "draftkings,fanduel,betmgm,caesars"

PYBASEBALL_TEAMS   = [
    # Standard abbreviations ──────────────────────────────────────────────────
    "NYY", "BOS", "LAD", "CHC", "SFG", "HOU", "ATL", "PHI", "COL", "MIN",
    "NYM", "CLE", "SEA", "TEX", "DET", "MIL", "STL", "CIN", "PIT", "ARI",
    "MIA", "WSN", "BAL", "TOR",
    # Baseball-Reference / pybaseball URL slugs (differ from ESPN/statsapi) ──
    "LAA",   # Los Angeles Angels
    "TBR",   # Tampa Bay Rays      (TB  on MLB Stats API; TBR on B-Ref)
    "SDP",   # San Diego Padres    (SD  on MLB Stats API; SDP on B-Ref)
    "KCR",   # Kansas City Royals  (KC  on MLB Stats API; KCR on B-Ref)
    "CWS",   # Chicago White Sox
    "OAK",   # Oakland Athletics
]

# ─────────────────────────────────────────────────────────────────────────────
# STREAM 1 — KALSHI  (today's games only)
# ─────────────────────────────────────────────────────────────────────────────
# Architecture note
# ─────────────────
# KXMLBGAME binary markets do NOT store OHLCV candlestick history — the
# candlestick endpoint returns 0 results for every market in this series.
# The correct signal is `yes_ask` (implied win probability) on the market
# object itself, which is returned directly from the /markets list call.
#
# We fetch only status=open markets and filter to those whose close_time
# falls on today's date so the pipeline only processes games being played
# today. One API call, no per-market requests, completes in ~1 second.

KALSHI_MARKETS_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"


def _kalshi_headers() -> dict:
    """Public read headers.  No auth required for market listing."""
    return {"Accept": "application/json"}


def fetch_kalshi(output_path: Path) -> None:
    today_str = date.today().isoformat()   # e.g. "2026-04-30"
    tomorrow_str = (date.today() + timedelta(days=1)).isoformat()
    print(f"[fetch_all_data] Stream 1 — Kalshi today's markets ({today_str})…")

    params = {
        "series_ticker": KALSHI_SERIES,
        "status":        "open",
        "limit":         200,   # generous ceiling; typical slate is 10–30 markets
    }

    try:
        resp = requests.get(KALSHI_MARKETS_URL, params=params,
                            headers=_kalshi_headers(), timeout=20)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  [Kalshi Error] market list fetch failed: {exc}")
        output_path.write_text(json.dumps([], indent=2))
        return

    all_open: list[dict] = resp.json().get("markets", [])
    print(f"    {len(all_open)} open KXMLBGAME markets total")

    # ── Debug: show the first market's time fields so we can see actual format
    if all_open:
        sample = all_open[0]
        time_fields = {k: sample[k] for k in sample if "time" in k.lower() or "date" in k.lower()}
        print(f"    [debug] sample market: ticker={sample.get('ticker','')}  time_fields={time_fields}")

    # ── Filter to today only ─────────────────────────────────────────────────
    # close_time is typically an ISO-8601 UTC string "2026-04-30T23:05:00Z"
    # but some markets use Unix timestamps (int).  Handle both.
    def _market_date(mkt: dict) -> str:
        """Return YYYY-MM-DD for a market's close_time, or '' on failure."""
        ct = mkt.get("close_time") or mkt.get("expiration_time") or mkt.get("end_time") or ""
        if isinstance(ct, (int, float)):
            # Unix timestamp → date string
            try:
                return pd.Timestamp(ct, unit="s", tz="UTC").strftime("%Y-%m-%d")
            except Exception:
                return ""
        if isinstance(ct, str) and len(ct) >= 10:
            return ct[:10]
        return ""

    today_markets = [m for m in all_open if _market_date(m) in [today_str, tomorrow_str]]
    print(f"    {len(today_markets)} markets closing today")

    # ── Extract implied probability from market object ────────────────────────
    # yes_ask / yes_bid are in cents (0–100); normalise to [0, 1].
    results: list[dict] = []
    for mkt in today_markets:
        raw_ask = mkt.get("yes_ask")
        raw_bid = mkt.get("yes_bid")
        yes_ask = round(raw_ask / 100, 4) if raw_ask is not None else None
        yes_bid = round(raw_bid / 100, 4) if raw_bid is not None else None

        results.append({
            "ticker":     mkt.get("ticker", ""),
            "title":      mkt.get("title", ""),
            "yes_ask":    yes_ask,
            "yes_bid":    yes_bid,
            "close_time": mkt.get("close_time", ""),
            "open_time":  mkt.get("open_time", ""),
            "status":     mkt.get("status", ""),
            "volume":     mkt.get("volume", 0),
        })
        ask_pct = f"{yes_ask:.1%}" if yes_ask is not None else "n/a"
        print(f"    {mkt.get('ticker','')}  yes_ask={ask_pct}")

    output_path.write_text(json.dumps(results, indent=2))
    print(f"  → {len(results)} today's markets saved to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# STREAM 2 — THE ODDS API  (player props)
# ─────────────────────────────────────────────────────────────────────────────
# Credit-optimisation notes
# ──────────────────────────
# The Odds API charges per /events/{id}/odds call (cost = markets × regions).
# With 4 markets × 1 region = 4 credits per game.  On a typical week the
# /events list returns 80–100 upcoming games; calling props for all of them
# wastes ~320–400 credits on games that aren't being played today.
#
# Fixes applied:
#   1. commenceTimeFrom / commenceTimeTo — only events starting today (UTC).
#      Window is today 00:00 UTC → tomorrow 07:00 UTC so west-coast night
#      games (start ~02:00 UTC next day local) are never cut off.
#   2. Log x-requests-remaining from every response header so you can track
#      burn rate across runs.
#   3. No artificial sleep — 10–15 today-only calls don't need throttling.

def _log_credits(resp: requests.Response, label: str = "") -> None:
    """Print remaining API credits from response headers (if present)."""
    remaining = resp.headers.get("x-requests-remaining")
    used      = resp.headers.get("x-requests-used")
    if remaining is not None:
        tag = f"  [{label}]" if label else ""
        print(f"    [Odds API credits]{tag} used={used}  remaining={remaining}")


def _get_today_events() -> list[dict]:
    """
    Return full event objects for MLB games starting today (local day in UTC).

    Uses commenceTimeFrom / commenceTimeTo to avoid fetching the full
    multi-week event list and burning credits on games not played today.
    Window: today 00:00:00Z → tomorrow 07:00:00Z (catches all US timezones).
    """
    if not ODDS_API_KEY:
        raise EnvironmentError("ODDS_API_KEY is not set in .env")

    from datetime import datetime, timezone, timedelta
    today_utc    = datetime.now(timezone.utc).replace(
                       hour=0, minute=0, second=0, microsecond=0)
    window_end   = today_utc + timedelta(hours=31)   # 31 h covers overnight PT games

    url    = f"{ODDS_BASE_URL}/sports/{ODDS_SPORT}/events"
    params = {
        "apiKey":             ODDS_API_KEY,
        "dateFormat":         "iso",
        "commenceTimeFrom":   today_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commenceTimeTo":     window_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    _log_credits(resp, "events")

    events = resp.json()
    print(f"    Found {len(events)} MLB events today")
    return events


def _fetch_event_player_props(event_id: str) -> tuple[list[dict], str, str, int, int]:
    """
    Fetch player props for a single event.
    Returns (bookmakers, home_team, away_team, credits_used, credits_remaining).

    NOTE: Do NOT pass 'bookmakers' here.
    The Odds API treats 'bookmakers' and 'regions' as mutually exclusive filters.
    If you specify bookmakers=draftkings,fanduel,...  the API returns ONLY those
    four books — and if none of them have posted lines yet (which is common early
    in the day) you get an empty bookmakers list for every event.
    Using regions=us returns any US-region book that has posted props, so you get
    results as soon as even one book opens lines.
    """
    url    = f"{ODDS_BASE_URL}/sports/{ODDS_SPORT}/events/{event_id}/odds"
    params = {
        "apiKey":      ODDS_API_KEY,
        "regions":     ODDS_REGIONS,   # 'us' — any US-region bookmaker
        "markets":     ODDS_MARKETS,
        "dateFormat":  "iso",
        "oddsFormat":  "decimal",      # confirmed format from live API (1.94, 1.8, etc.)
    }
    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code == 422:
        # props not yet listed for this event (pre-game window) — not an error
        return [], "", "", 0, 0
    resp.raise_for_status()

    used      = int(resp.headers.get("x-requests-used",      0))
    remaining = int(resp.headers.get("x-requests-remaining", 0))
    body      = resp.json()
    return (
        body.get("bookmakers", []),
        body.get("home_team",  ""),
        body.get("away_team",  ""),
        used,
        remaining,
    )


def fetch_odds(output_path: Path) -> None:
    """
    Pull live player props for today's MLB games only.
    Stores home_team + away_team alongside bookmakers so process_model.py
    can apply wind/park multipliers per stadium.
    Saves structured JSON to output_path.
    """
    print("[fetch_all_data] Stream 2 — The Odds API player props…")
    events    = _get_today_events()
    all_props: list[dict] = []
    credits_remaining: Optional[int] = None

    for evt in events:
        eid              = evt["id"]
        home_from_events = evt.get("home_team", "")
        away_from_events = evt.get("away_team", "")

        print(f"    {home_from_events} vs {away_from_events}", end=" ", flush=True)
        try:
            bookmakers, home_from_odds, away_from_odds, used, remaining = (
                _fetch_event_player_props(eid)
            )
        except requests.RequestException as exc:
            print(f"[ERR: {exc}]")
            continue

        if remaining:
            credits_remaining = remaining

        home_team = home_from_odds or home_from_events
        away_team = away_from_odds or away_from_events

        all_props.append({
            "event_id":      eid,
            "home_team":     home_team,
            "away_team":     away_team,
            "commence_time": evt.get("commence_time", ""),
            "bookmakers":    bookmakers,
        })
        bk_count = len(bookmakers)
        print(f"({bk_count} bookmakers, {used} credits used)")

    output_path.write_text(json.dumps(all_props, indent=2))
    cr_str = f"  {credits_remaining} credits remaining" if credits_remaining else ""
    print(f"  → Props for {len(all_props)} events saved to {output_path}.{cr_str}")


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


def _update_h2h_2026(output_dir):
    import pandas as pd
    import numpy as np
    import time
    import io
    from curl_cffi.requests import Session as CurlSession

    teams = ['NYY','BOS','LAD','CHC','SFG','HOU','ATL','PHI','COL','MIN','NYM','CLE',
             'SEA','TEX','DET','MIL','STL','CIN','PIT','ARI','MIA','WSN','BAL','TOR',
             'LAA','TBR','SDP','KCR','CHW','ATH'] # <-- Changed OAK to ATH
    
    all_teams_data = []
    print(f"[pybaseball] Fluidly fetching 2026 season for {len(teams)} teams (Bypass Mode)...")

    # Use curl_cffi to bypass Cloudflare blockades
    session = CurlSession(impersonate="chrome110")
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    })

    for team in teams:
        try:
            print(f"  Scraping {team}...")
            url = f"https://www.baseball-reference.com/teams/{team}/2026-schedule-scores.shtml"
            resp = session.get(url, timeout=15)
            
            # Parse HTML directly with Pandas to avoid pybaseball bugs
            tables = pd.read_html(io.StringIO(resp.text))
            df = tables[0].copy()
            
            # Clean up the table headers
            df.columns = [str(c).replace('*', '') for c in df.columns]
            if 'Gm#' in df.columns:
                df = df[df['Gm#'] != 'Gm#'].copy()  # Remove mid-table repeating headers
            
            # Safely handle the dreaded 'Unknown' attendance
            if 'Attendance' in df.columns:
                df['Attendance'] = pd.to_numeric(
                    df['Attendance'].astype(str).str.replace(',', '').replace('Unknown', np.nan), 
                    errors='coerce'
                )
            
            # Standardize column names for your model
            df.columns = [str(c).lower().replace('.', '').replace('/', '_') for c in df.columns]
            df['team'] = team
            df['season'] = 2026
            
            all_teams_data.append(df)
            time.sleep(1.5) # Be polite to the server
            
        except Exception as e:
            print(f"  Failed for {team}: {e}")

    if all_teams_data:
        full_2026 = pd.concat(all_teams_data, ignore_index=True)
        # Deduplicate
        if 'date' in full_2026.columns and 'opp' in full_2026.columns:
            full_2026 = full_2026.drop_duplicates(subset=['date', 'team', 'opp'])
        
        file_path = output_dir / "h2h_2026.csv"
        full_2026.to_csv(file_path, index=False)
        print(f"\n[pybaseball] ✓ Fluid Sync Complete. Total rows: {len(full_2026)}")


def fetch_pybaseball(output_dir: Path) -> None:
    """
    H2H log strategy:
      • 2025 — fetch once and cache; skip if h2h_2025.csv already exists.
      • 2026 — fetch all data gracefully handling bad B-Ref data
    """
    print("[fetch_all_data] Stream 3 — pybaseball H2H logs…")
    pybaseball.cache.enable()

    # ── 2025: skip if already on disk ────────────────────────────────────────
    path_2025 = output_dir / "h2h_2025.csv"
    if path_2025.exists():
        print(f"  2025: h2h_2025.csv already exists ({path_2025.stat().st_size // 1024} KB) — skipping.")
    else:
        print("  2025: file not found — fetching full season (one-time)…")
        df_2025 = _build_h2h(2025)
        if df_2025.empty:
            print("    No data returned for 2025.")
        else:
            df_2025.to_csv(path_2025, index=False)
            print(f"  → {len(df_2025)} rows saved to {path_2025}")

    # ── 2026: self-healing full sync ─────────────────────────────────────────
    _update_h2h_2026(output_dir)

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

STREAM_MAP = {
    "kalshi":     lambda: fetch_kalshi(RAW_DIR / "kalshi_today.json"),
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