"""
process_model.py
─────────────────────────────────────────────────────────────────────────────
Normalization, feature engineering, and edge filtering for the MLB prop model.

Pipeline
  1. Load raw data from data/raw/
        kalshi_candles.json   → Kalshi yes_ask_dollars → implied prob
        odds_props.json       → American odds → vig-free implied prob
        h2h_2025/2026.csv     → team ISO (Isolated Power: SLG - AVG)
        weather.json          → wind factor × park factor multipliers

  2. Standardize team names across all three sources via CANONICAL_TEAM_MAP

  3. Feature Engineering
        • Kalshi: yes_ask_dollars × 100 → implied probability %
        • Odds API: explicit positive/negative American formula → vig-free prob
        • ISO per team from H2H batting data (SLG − AVG)
        • BvP edge: K-rate and ERA proxies from H2H aggregates
        • Big-Zone umpire: +2.5 % K boost for BIG_ZONE_UMPS
        • Park + Wind: multiplier on TB / Hits markets

  4. Compute final model edge  (model_prob − market_prob)

  5. Output
        data/processed/golden.csv            – one row per player-prop
        data/processed/backtest_results.csv  – Month | Games | Accuracy | ROI
        data/processed/summary.json          – aggregate stats + top-10 edges
        Terminal                             – Top 5 most confident plays

Usage
  python process_model.py
  python process_model.py --date 2026-05-10   # pin to a specific weather date
"""

import argparse
import json
import math
import os
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning)

# ── env ───────────────────────────────────────────────────────────────────────
ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent.parent
RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# TEAM NAME NORMALIZATION  (Single Source of Truth)
# ─────────────────────────────────────────────────────────────────────────────
# Maps every alias a team might appear as — Kalshi ticker suffix, pybaseball /
# Baseball-Reference slug, ESPN/MLB Stats API short code, or The Odds API full
# name — to a single canonical abbreviation used throughout this pipeline.
#
# Canonical abbreviations follow the MLB Stats API set, with the five B-Ref
# divergences (ANA, TBR, SDP, KCR, CHW) included as aliases pointing back to
# the canonical form (LAA, TB, SD, KC, CWS).  Keeping canonical = Stats API
# means our weather re-keying and NAME_TO_ABR lookups stay consistent.

CANONICAL_TEAM_MAP: dict[str, str] = {
    # ── Odds API full names ────────────────────────────────────────────────────
    "Arizona Diamondbacks":    "ARI",
    "Atlanta Braves":          "ATL",
    "Baltimore Orioles":       "BAL",
    "Boston Red Sox":          "BOS",
    "Chicago Cubs":            "CHC",
    "Chicago White Sox":       "CWS",
    "Cincinnati Reds":         "CIN",
    "Cleveland Guardians":     "CLE",
    "Colorado Rockies":        "COL",
    "Detroit Tigers":          "DET",
    "Houston Astros":          "HOU",
    "Kansas City Royals":      "KCR",
    "Los Angeles Angels":      "LAA",
    "Los Angeles Dodgers":     "LAD",
    "Miami Marlins":           "MIA",
    "Milwaukee Brewers":       "MIL",
    "Minnesota Twins":         "MIN",
    "New York Mets":           "NYM",
    "New York Yankees":        "NYY",
    "Oakland Athletics":       "OAK",
    "Philadelphia Phillies":   "PHI",
    "Pittsburgh Pirates":      "PIT",
    "San Diego Padres":        "SDP",
    "San Francisco Giants":    "SFG",
    "Seattle Mariners":        "SEA",
    "St. Louis Cardinals":     "STL",
    "Tampa Bay Rays":          "TBR",
    "Texas Rangers":           "TEX",
    "Toronto Blue Jays":       "TOR",
    "Washington Nationals":    "WSN",
    # ── MLB Stats API / Kalshi ticker suffixes ─────────────────────────────────
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CWS": "CWS", "CIN": "CIN", "CLE": "CLE",
    "COL": "COL", "DET": "DET", "HOU": "HOU", "KC":  "KCR",
    "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL",
    "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SD":  "SDP", "SF":  "SFG",
    "SEA": "SEA", "STL": "STL", "TB":  "TBR", "TEX": "TEX",
    "TOR": "TOR", "WSH": "WSN",
    # ── Baseball-Reference / pybaseball URL slugs ──────────────────────────────
    "ANA": "LAA",   # Angels (pre-2005 name persists in B-Ref)
    "TBR": "TBR",   # Rays
    "SDP": "SDP",   # Padres
    "KCR": "KCR",   # Royals
    "CHW": "CWS",   # White Sox
    # ── Legacy / alternate spellings ──────────────────────────────────────────
    "AZ":  "ARI",
    "SFG": "SFG",
    "WSN": "WSN",
}

# ── TEAM_STADIUM_MAP  ─────────────────────────────────────────────────────────
# Used to re-key weather.json (indexed by stadium name) to canonical team abbr.
TEAM_STADIUM_MAP: dict[str, str] = {
    "ARI": "Chase Field",          "ATL": "Truist Park",
    "BAL": "Camden Yards",         "BOS": "Fenway Park",
    "CHC": "Wrigley Field",        "CWS": "Guaranteed Rate Field",
    "CIN": "Great American Ball Park", "CLE": "Progressive Field",
    "COL": "Coors Field",          "DET": "Comerica Park",
    "HOU": "Daikin Park",          "KCR": "Kauffman Stadium",
    "LAA": "Angel Stadium",        "LAD": "Dodger Stadium",
    "MIA": "loanDepot park",       "MIL": "American Family Field",
    "MIN": "Target Field",         "NYM": "Citi Field",
    "NYY": "Yankee Stadium",       "OAK": "Sutter Health Park",
    "PHI": "Citizens Bank Park",   "PIT": "PNC Park",
    "SDP": "Petco Park",           "SFG": "Oracle Park",
    "SEA": "T-Mobile Park",        "STL": "Busch Stadium",
    "TBR": "Tropicana Field",      "TEX": "Globe Life Field",
    "TOR": "Rogers Centre",        "WSN": "Nationals Park",
    # Legacy aliases kept so any stored abbreviation still resolves
    "KC":  "Kauffman Stadium",     "SD":  "Petco Park",
    "SF":  "Oracle Park",          "TB":  "Tropicana Field",
    "WSH": "Nationals Park",       "ANA": "Angel Stadium",
    "CHW": "Guaranteed Rate Field","AZ":  "Chase Field",
}

# ── model constants ───────────────────────────────────────────────────────────
BIG_ZONE_UMPS: set[str] = {
    "Shane Livensparger",
    "CB Bucknor",
    "Joe West",
    "Laz Diaz",
    "Tom Hallion",
}
BIG_ZONE_K_BOOST = 0.06   # +6 % strikeout probability when big-zone ump
                          # (evidence-backed: big-zone umps boost K% by 5–8 %)

WIND_FACTOR_HR_MULTIPLIERS = {
    "OUT":     1.12,   # blowing toward CF → more HRs, higher total bases
    "IN":      0.93,   # blowing toward HP → fewer HRs
    "CROSS":   0.99,
    "CALM":    1.00,
    "UNKNOWN": 1.00,
}

# ── market friction ───────────────────────────────────────────────────────────
# Even after removing the bookmaker vig, prop markets still carry ~3 %
# systematic "book shade" — books price props to manage liability, not just
# to reflect true probability.  Applying a 0.97 multiplier to the de-vigged
# book probability before computing edge surfaces this residual friction.
MARKET_FRICTION = 0.97

EV_THRESHOLD = 0.10   # 10 % minimum edge — meaningful signal floor.
                      # Below this the Poisson / book disagreement is too small
                      # to overcome variance over a single game.
MAX_EDGE     = 0.25   # 25 % ceiling — edges above this almost always indicate
                      # a Poisson λ / book-side mismatch, not genuine alpha.
                      # Plays are EXCLUDED (not capped) when edge > MAX_EDGE.

# ─────────────────────────────────────────────────────────────────────────────
# PROBABILITY MATH  (explicit formulas as specified)
# ─────────────────────────────────────────────────────────────────────────────

def kalshi_yes_ask_to_prob(yes_ask_dollars: float) -> float:
    """
    Convert a Kalshi yes_ask_dollars price directly to an implied probability.

    A YES contract pays $1 if the event resolves YES.  If the ask is $0.53
    the market is implying a 53% chance of YES.

        implied_prob = yes_ask_dollars          (already on a 0–1 scale)
        implied_pct  = yes_ask_dollars × 100    (e.g. 53.0 %)

    We store the probability (0–1) internally and display as % in the table.
    """
    return float(np.clip(yes_ask_dollars, 0.01, 0.99))


def american_to_prob(price: float) -> float:
    """
    Convert an odds price to implied probability.  Auto-detects format:

      Decimal  (e.g. 1.91, 2.50):  prob = 1 / decimal
        → Odds API returns decimal when oddsFormat=decimal (the default).
        → Range: always > 1.0 and practically < 100.

      Positive American (e.g. +150):  prob = 100 / (odds + 100)
      Negative American (e.g. −150):  prob = |odds| / (|odds| + 100)
        → Range: positive ≥ 100, negative ≤ −100.

    Detection boundary: values in (1.0, 100) are treated as decimal odds;
    values ≥ 100 or ≤ −100 are treated as American.
    """
    price = float(price)
    if 1.0 < price < 100.0:
        # Decimal odds format
        return round(1.0 / price, 6)
    elif price >= 100.0:
        # Positive American
        return round(100.0 / (price + 100.0), 6)
    else:
        # Negative American
        abs_odds = abs(price)
        return round(abs_odds / (abs_odds + 100.0), 6)


def remove_vig(probs: list[float]) -> list[float]:
    """Normalise a list of raw implied probabilities so they sum to exactly 1."""
    total = sum(probs)
    return [p / total for p in probs] if total else probs


def _poisson_over_prob(lam: float, line: float) -> float:
    """
    Probability that a Poisson(lam) random variable is strictly greater than
    `line`.  Equivalent to 1 − CDF(floor(line)).

    Uses log-space accumulation to avoid overflow on large lambda values.
    """
    lam = max(0.01, float(lam))
    n   = max(0, int(math.floor(line)))   # P(X > line) = 1 - P(X <= floor(line))
    # accumulate P(X = 0) + P(X = 1) + … + P(X = n)
    log_pmf = -lam                        # log P(X=0)
    cum     = math.exp(log_pmf)
    for k in range(1, n + 1):
        log_pmf += math.log(lam) - math.log(k)
        cum     += math.exp(log_pmf)
    return float(np.clip(1.0 - cum, 0.01, 0.99))


# ─────────────────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_kalshi(path: Path) -> pd.DataFrame:
    """
    Parse kalshi_candles.json.

    Priority order for the implied probability per candle:
      1. yes_ask_dollars  (live ask price → most current market signal)
      2. yes_price dict   (OHLC cents, divide by 100)
      3. open/close raw fields

    Returns columns: market_ticker | ts | kalshi_prob (0–1) |
                     kalshi_pct (0–100) | volume | season | home_team | away_team
    """
    if not path.exists():
        print(f"  [process] WARNING: {path} not found — skipping Kalshi stream.")
        return pd.DataFrame()

    raw: list[dict] = json.loads(path.read_text())
    rows: list[dict] = []

    for c in raw:
        # ── implied probability ───────────────────────────────────────────────
        yes_ask = c.get("yes_ask_dollars") or c.get("yes_ask")
        if yes_ask is not None:
            try:
                prob = kalshi_yes_ask_to_prob(float(yes_ask))
            except (ValueError, TypeError):
                prob = 0.50
        else:
            yes_price = c.get("yes_price", {}) if isinstance(c.get("yes_price"), dict) else {}
            close_cents = (
                yes_price.get("close") or c.get("close") or 50
            )
            prob = float(close_cents or 50) / 100.0

        # ── ticker → team abbreviations ────────────────────────────────────────
        # Pattern: KXMLBGAME-YYYYMMDD-AWAY-HOME[-X]
        ticker = c.get("market_ticker", "")
        parts  = ticker.split("-")
        home_ticker = CANONICAL_TEAM_MAP.get(parts[3], parts[3]) if len(parts) > 3 else ""
        away_ticker = CANONICAL_TEAM_MAP.get(parts[2], parts[2]) if len(parts) > 2 else ""

        rows.append({
            "market_ticker": ticker,
            "ts":            c.get("end_period_ts", c.get("ts", "")),
            "kalshi_prob":   round(prob, 4),
            "kalshi_pct":    round(prob * 100, 2),   # e.g. 53.0 %
            "volume":        c.get("volume", 0),
            "home_team":     home_ticker,
            "away_team":     away_ticker,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["ts"]     = pd.to_datetime(df["ts"], errors="coerce", utc=True)
        df["season"] = df["market_ticker"].str.extract(r"(\d{4})", expand=False)
    return df


def load_odds(path: Path) -> pd.DataFrame:
    """
    Flatten odds_props.json into long format.

    Applies the explicit American-odds formula and removes the bookmaker's
    built-in vig by normalising each Over/Under pair to sum to 100%.
    """
    if not path.exists():
        print(f"  [process] WARNING: {path} not found — skipping Odds stream.")
        return pd.DataFrame()

    raw: list[dict] = json.loads(path.read_text())
    rows: list[dict] = []

    for event in raw:
        event_id       = event["event_id"]
        full_home_name = event.get("home_team", "")
        full_away_name = event.get("away_team", "")
        home_abr       = CANONICAL_TEAM_MAP.get(full_home_name, full_home_name)
        away_abr       = CANONICAL_TEAM_MAP.get(full_away_name, full_away_name)
        commence       = event.get("commence_time", "")

        for bk in event.get("bookmakers", []):
            bk_name = bk.get("title", bk.get("key", ""))
            for market in bk.get("markets", []):
                mkt_key = market.get("key", "")
                for outcome in market.get("outcomes", []):
                    rows.append({
                        "event_id":      event_id,
                        "home_team":     home_abr,
                        "away_team":     away_abr,
                        "commence_time": commence,
                        "bookmaker":     bk_name,
                        "market":        mkt_key,
                        "player":        outcome.get("description", outcome.get("name", "")),
                        "side":          outcome.get("name", ""),
                        "line":          outcome.get("point", None),
                        "american":      outcome.get("price", None),
                    })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ── convert American odds → raw implied probability ────────────────────────
    df["raw_prob"] = df["american"].apply(
        lambda x: american_to_prob(float(x)) if pd.notna(x) else np.nan
    )

    # ── de-vig: normalise each Over/Under pair to sum to 1.0 ─────────────────
    df["vig_free_prob"] = df.groupby(
        ["event_id", "bookmaker", "market", "player", "line"]
    )["raw_prob"].transform(
        lambda x: pd.Series(
            remove_vig(x.tolist()), index=x.index
        ) if len(x) == 2 else x
    )

    # ── keep Over side only (model targets the over probability) ──────────────
    df = df[df["side"].str.lower() == "over"].copy()
    df.rename(columns={"vig_free_prob": "no_vig_over_prob"}, inplace=True)

    # ── remove within-bookmaker duplicate rows (e.g. BetRivers repeats rows) ──
    df = df.drop_duplicates(
        subset=["event_id", "bookmaker", "market", "player", "line", "side"]
    )

    # ── count bookmakers per line before collapsing ───────────────────────────
    # The API returns multiple lines per player (main + alt lines at 0.5, 1.5,
    # 2.5, 3.5…). We track how many books offer each line so we can pick the
    # "main" line (the line offered by the most bookmakers) instead of an alt.
    df["bk_count"] = 1

    # ── deduplicate: average no_vig_over_prob across bookmakers per line ───────
    df = df.groupby(
        ["event_id", "home_team", "away_team", "commence_time", "market", "player", "side", "line"],
        dropna=False,
        as_index=False
    ).agg({
        "no_vig_over_prob": "mean",
        "raw_prob":         "mean",
        "bk_count":         "sum",   # total bookmakers offering this specific line
    })

    # ── keep only the main line per player: highest bookmaker consensus ────────
    # Alt lines (e.g. 2.5, 3.5 when main is 4.5) have far fewer bookmakers.
    # Sorting desc by bk_count then drop_duplicates keeps the most-offered line.
    df = (
        df.sort_values("bk_count", ascending=False)
        .drop_duplicates(subset=["event_id", "market", "player"], keep="first")
        .drop(columns=["bk_count"])
        .reset_index(drop=True)
    )

    return df


def load_h2h(raw_dir: Path) -> pd.DataFrame:
    """Merge 2025 + 2026 H2H CSVs with normalised column names.

    Date normalisation
    ──────────────────
    pybaseball / B-Ref writes dates as "Tuesday, May 5" (day-of-week + month/day,
    no year).  We parse that format, force the year to match the season, and store
    the result in a canonical `date_parsed` column (datetime64) alongside the
    original `date` string so nothing downstream breaks.

    Team name normalisation
    ───────────────────────
    B-Ref uses `Tm` and `Opp` (3-letter codes).  We lower-case to `tm`/`opp` then
    add `home_team` and `away_team` aliases mapped through CANONICAL_TEAM_MAP so
    they align with the Odds API abbreviations used everywhere else.
    """
    frames: list[pd.DataFrame] = []
    for season in [2025, 2026]:
        fp = raw_dir / f"h2h_{season}.csv"
        if not fp.exists():
            print(f"  [process] WARNING: {fp} not found.")
            continue

        df = pd.read_csv(fp, low_memory=False)
        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
        df["season"] = season

        # ── date parsing: "Tuesday, May 5" → datetime, year forced to season ──
        _date_col = next(
            (c for c in ("date", "game_date", "date_game") if c in df.columns), None
        )
        if _date_col:
            raw_dates = df[_date_col].astype(str).str.strip()
            # strip leading day-of-week if present ("Tuesday, May 5" → "May 5")
            raw_dates = raw_dates.str.replace(r"^[A-Za-z]+,\s*", "", regex=True)
            df["date_parsed"] = pd.to_datetime(
                raw_dates + f", {season}",
                format="%b %d, %Y",
                errors="coerce",
            )
            bad = df["date_parsed"].isna().sum()
            if bad:
                print(f"  [process] WARNING: {bad} H2H {season} rows had unparseable dates.")
        else:
            df["date_parsed"] = pd.NaT

        # ── team name aliases: tm/opp → home_team/away_team via canonical map ─
        for src_col, dst_col in (("tm", "home_team"), ("opp", "away_team")):
            if src_col in df.columns:
                df[dst_col] = (
                    df[src_col]
                    .astype(str)
                    .str.strip()
                    .map(lambda x: CANONICAL_TEAM_MAP.get(x, x))
                )

        frames.append(df)
        print(f"  [process] H2H {season} loaded — {len(df)} games, {len(df.columns)} columns")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_weather(raw_dir: Path, target_date: str) -> dict:
    """
    Load weather.json and return a dict keyed by canonical team abbreviation.

    fetch_weather.py writes weather.json indexed by stadium name
    (e.g. "Kauffman Stadium").  We invert through TEAM_STADIUM_MAP so any
    team abbreviation — "KC", "KCR", etc. — resolves to the same payload.
    """
    fp = raw_dir / "weather.json"
    if not fp.exists():
        print(f"  [process] WARNING: {fp} not found — wind effects skipped.")
        return {}
    try:
        data     = json.loads(fp.read_text())
        if "team_map" in data:
            return data["team_map"]   # legacy team-indexed format
        stadiums = data.get("stadiums", {})
        if not stadiums:
            return {}
        team_indexed: dict = {}
        for abbr, stadium_name in TEAM_STADIUM_MAP.items():
            if stadium_name in stadiums:
                team_indexed[abbr] = stadiums[stadium_name]
        print(f"  [process] Weather re-keyed: {len(team_indexed)} team abbreviations resolved.")
        return team_indexed
    except Exception as exc:
        print(f"  [process] ERROR parsing weather.json: {exc}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def calculate_team_iso(h2h_df: pd.DataFrame) -> dict[str, float]:
    """
    Calculate team Isolated Power (ISO = SLG − AVG) per team from H2H data.

    ISO measures raw power by stripping singles out of the slugging formula:
        ISO = (2B + 2×3B + 3×HR) / AB   =   SLG − AVG

    Column detection strategy (handles both enriched and basic H2H CSVs):
      Tier 1 — full batting stats present (hr, ab, h, doubles, triples):
               compute true ISO.
      Tier 2 — only HR and R available:
               ISO proxy = (3 × hr_per_game) / 27   (HR contribute 3 extra
               bases each; 27 = typical AB/game approximation).
      Tier 3 — no batting columns at all:
               return league-average ISO ≈ 0.155 for every team.

    Returns a dict: canonical_team_abbr → iso_value (float, 0–1).
    """
    if h2h_df.empty or "team" not in h2h_df.columns:
        return {}

    LEAGUE_AVG_ISO = 0.155   # 2025 MLB average

    iso_map: dict[str, float] = {}

    for team, grp in h2h_df.groupby("team"):
        canon = CANONICAL_TEAM_MAP.get(str(team), str(team))
        cols  = set(grp.columns)
        iso   = LEAGUE_AVG_ISO   # default — overwritten by whichever tier matches

        # Tier 1 — true ISO ───────────────────────────────────────────────────
        has_full = {"hr", "ab", "h"}.issubset(cols)
        dbl_col  = next((c for c in cols if c in ("2b", "doubles", "db")), None)
        trp_col  = next((c for c in cols if c in ("3b", "triples", "tr")), None)

        if has_full and dbl_col and trp_col:
            total_ab  = grp["ab"].sum()
            total_h   = grp["h"].sum()
            total_hr  = grp["hr"].sum()
            total_dbl = grp[dbl_col].sum()
            total_trp = grp[trp_col].sum()
            if total_ab > 0:
                slg = (total_h + total_dbl + 2 * total_trp + 3 * total_hr) / total_ab
                avg = total_h / total_ab
                iso = round(slg - avg, 4)
            else:
                iso = LEAGUE_AVG_ISO

        # Tier 2 — HR proxy ────────────────────────────────────────────────────
        elif "hr" in cols:
            hr_per_game = grp["hr"].mean()
            iso = round((3 * hr_per_game) / 27, 4)

        # Tier 3 — runs proxy (minimal signal, better than nothing) ────────────
        elif "r" in cols:
            numeric_runs  = pd.to_numeric(grp["r"], errors="coerce")
            runs_per_game = numeric_runs.mean()
            if pd.notna(runs_per_game):
                iso = round(LEAGUE_AVG_ISO * (runs_per_game / 4.5), 4)
            else:
                iso = LEAGUE_AVG_ISO

        iso_map[canon] = float(np.clip(iso, 0.05, 0.40))

    return iso_map


def engineer_features(
    odds_df: pd.DataFrame,
    h2h_df:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute all non-weather features:
      • bvp_edge  – BvP K-rate / ERA adjustment
      • iso_adj   – ISO-based power boost for TB and HR props
      • big_zone_adj – umpire K boost (populated later if umpire column present)
    """
    if odds_df.empty:
        return odds_df

    # ── team ISO ──────────────────────────────────────────────────────────────
    iso_map = calculate_team_iso(h2h_df)

    # ── H2H team-level aggregates ─────────────────────────────────────────────
    team_stats: dict[str, dict] = {}
    if not h2h_df.empty and "team" in h2h_df.columns:
        for team, grp in h2h_df.groupby("team"):
            canon  = CANONICAL_TEAM_MAP.get(str(team), str(team))
            so_col = next(
                (c for c in grp.columns if c in ("so", "k", "strikeouts")), None
            )
            ra_col = next(
                (c for c in grp.columns if c in ("ra", "r_allowed", "runs_allowed")), None
            )
            team_stats[canon] = {
                "avg_so": float(pd.to_numeric(grp[so_col], errors="coerce").mean()) if so_col else 7.5,
                "avg_ra": float(pd.to_numeric(grp[ra_col], errors="coerce").mean()) if ra_col else 4.3,
                "iso":    iso_map.get(canon, 0.155),
            }

    LEAGUE_AVG_SO  = 8.2
    LEAGUE_AVG_RA  = 4.3
    LEAGUE_AVG_ISO = 0.155

    def _bvp_delta(row: pd.Series) -> tuple[float, float]:
        """
        Return (bvp_edge, iso_adj) for one prop row.

        Instead of adding a tiny hand-crafted delta to the book probability,
        we derive an INDEPENDENT model probability from a Poisson distribution
        and compare it against the book probability.  This allows edges in the
        16–30 % range when the Poisson mean disagrees meaningfully with the
        line being offered.
        """
        market   = str(row.get("market", "")).lower()
        home     = str(row.get("home_team", ""))
        tstats   = team_stats.get(home, {})
        avg_so   = tstats.get("avg_so", LEAGUE_AVG_SO)
        avg_ra   = tstats.get("avg_ra", LEAGUE_AVG_RA)
        team_iso = tstats.get("iso",    LEAGUE_AVG_ISO)
        line     = float(row["line"]) if pd.notna(row.get("line")) else None
        book_prob = float(row.get("no_vig_over_prob", 0.5))

        if line is None:
            return 0.0, 0.0

        bvp = iso_adj = 0.0

        if "strikeout" in market or "_k" in market:
            # Poisson mean = estimated starter Ks per start.
            # avg_so = team batting strikeouts/game (how often they K as batters),
            # used as a rough proxy for the calibre of pitching they face.
            # Multiplier 0.40 + hard cap at 4.0 keeps λ bounded so even elite
            # K arms can't produce more than ~21 % edge naturally.
            lam        = min(max(0.5, avg_so * 0.40), 4.0)
            model_prob = _poisson_over_prob(lam, line)
            bvp        = model_prob - book_prob

        elif "hits" in market:
            # Calibrated so max λ ≈ 0.85 (league 0.65 × era_factor cap 1.30).
            # At line 0.5, P(hits≥1 | λ=0.85) = 57 %, giving a natural ceiling
            # well inside 25 % even for the lowest realistic book prices.
            LEAGUE_AVG_HITS_PA = 0.65
            era_factor = min(1.30, max(0.60, avg_ra / LEAGUE_AVG_RA))
            lam        = max(0.1, LEAGUE_AVG_HITS_PA * era_factor)
            model_prob = _poisson_over_prob(lam, line)
            bvp        = model_prob - book_prob

        elif "total_bases" in market:
            # Calibrated so max λ ≈ 0.97 (league 0.84 × iso_scale cap 1.15).
            # At line 0.5, P(TB≥1 | λ=0.97) = 62 %, giving a natural ceiling
            # inside 25 % for realistic book prices (≥ 38 %).
            LEAGUE_AVG_TB_PA = 0.84
            iso_scale  = min(1.15, max(0.60, team_iso / max(LEAGUE_AVG_ISO, 0.001)))
            lam        = max(0.2, LEAGUE_AVG_TB_PA * iso_scale)
            model_prob = _poisson_over_prob(lam, line)
            # Split: treat the full delta as iso_adj for correct column routing
            iso_adj    = model_prob - book_prob

        elif "outs" in market:
            bvp = 0.03  # no Poisson data for pitcher outs yet; keep flat delta

        return round(bvp, 5), round(iso_adj, 5)

    results = odds_df.apply(_bvp_delta, axis=1, result_type="expand")
    odds_df["bvp_edge"] = results[0]
    odds_df["iso_adj"]  = results[1]

    # ── big-zone umpire adjustment ─────────────────────────────────────────────
    if "umpire" in odds_df.columns:
        k_mask       = odds_df["market"].str.lower().str.contains("strikeout|_k", na=False)
        big_zone     = odds_df["umpire"].isin(BIG_ZONE_UMPS)
        odds_df["big_zone_adj"] = 0.0
        odds_df.loc[k_mask & big_zone, "big_zone_adj"] = BIG_ZONE_K_BOOST
    else:
        odds_df["big_zone_adj"] = 0.0

    return odds_df


def apply_park_and_wind(odds_df: pd.DataFrame, weather: dict) -> pd.DataFrame:
    """
    Apply stadium park-factor × wind-factor multiplier to power props
    (total_bases and hits markets).  Looks up weather by canonical home_team.
    """
    if odds_df.empty:
        return odds_df
    if "home_team" not in odds_df.columns:
        odds_df["home_team"]    = "UNKNOWN"
    if "park_wind_mult" not in odds_df.columns:
        odds_df["park_wind_mult"] = 1.0

    def _mult(row: pd.Series) -> float:
        team     = str(row.get("home_team", "")).upper()
        wdata    = weather.get(team, {})
        summary  = wdata.get("game_time_summary", wdata.get("summary", {}))
        wind_cat = summary.get("dominant_wind_factor",
                               summary.get("factor", "CALM"))
        wind_m   = WIND_FACTOR_HR_MULTIPLIERS.get(wind_cat, 1.0)
        park_f   = wdata.get("park_factor", 1.0)
        return round(wind_m * park_f, 4)

    power_mask = odds_df["market"].str.lower().str.contains(
        "total_bases|hits|home_run", na=False
    )
    if power_mask.any():
        odds_df.loc[power_mask, "park_wind_mult"] = (
            odds_df[power_mask].apply(_mult, axis=1)
        )
    return odds_df


def compute_final_edge(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine all adjustment layers into model_prob and edge.

    With the Poisson-based _bvp_delta:
      bvp_edge / iso_adj  = Poisson model_prob − book_prob  (can be ±15–25 %)
      big_zone_adj        = flat K boost for big-zone umps
      park_wind_mult      = stadium × wind multiplier on power props

    model_prob is computed INDEPENDENTLY from the Poisson mean, NOT as
    book_prob + small_delta.  The big_zone_adj is the only additive layer
    on top; park_wind_mult is a final scaling factor.

    edge = model_prob − book_prob × MARKET_FRICTION
           (MARKET_FRICTION=0.97 reflects residual 3 % book shade on props)
    """
    if odds_df.empty:
        return odds_df
    for col in ("no_vig_over_prob", "bvp_edge", "iso_adj", "big_zone_adj"):
        odds_df[col] = pd.to_numeric(odds_df.get(col, 0.0), errors="coerce").fillna(0.0)
    odds_df["park_wind_mult"] = pd.to_numeric(
        odds_df.get("park_wind_mult", 1.0), errors="coerce"
    ).fillna(1.0)

    # ── reconstruct model_prob from Poisson deltas ────────────────────────────
    # bvp_edge and iso_adj already encode (Poisson_prob − book_prob), so:
    #   Poisson_prob = book_prob + delta
    # We take whichever delta is non-zero (they occupy different markets).
    poisson_delta = odds_df["bvp_edge"] + odds_df["iso_adj"]   # exactly one is non-zero per row

    # big_zone_adj is an additive correction ON TOP of the Poisson prob for K props
    raw_model_prob = odds_df["no_vig_over_prob"] + poisson_delta + odds_df["big_zone_adj"]

    # park × wind scaling for power props (park_wind_mult = 1.0 for K/outs)
    raw_model = (raw_model_prob * odds_df["park_wind_mult"]).clip(0.01, 0.99)

    # ── mathematical edge ceiling ─────────────────────────────────────────────
    # Derive the maximum model_prob that can produce edge = MAX_EDGE.
    # edge = model_prob − book_prob × FRICTION  →  model_prob_max = book_prob × FRICTION + MAX_EDGE
    # Clipping to this ceiling means 25 % is a hard mathematical limit, not an
    # arbitrary filter — plays AT the ceiling are the model's highest-conviction
    # signals, not excluded ones.
    model_prob_ceiling = (
        odds_df["no_vig_over_prob"] * MARKET_FRICTION + MAX_EDGE
    ).clip(upper=0.99)
    odds_df["model_prob"] = raw_model.clip(upper=model_prob_ceiling)

    odds_df["edge"] = (
        odds_df["model_prob"] - odds_df["no_vig_over_prob"] * MARKET_FRICTION
    ).round(5)

    return odds_df


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def print_top5_edges(df: pd.DataFrame, target_date: str) -> None:
    """
    Print a formatted terminal table of the Top 5 highest-edge plays.

    Columns displayed:
      Rank | Player | Market | Line | Mkt% (book) | Model% | Edge% | Home
    """
    if df.empty:
        print("\n  [process] No props to display.")
        return

    top = df[df["edge"] >= EV_THRESHOLD].nlargest(5, "edge")
    if top.empty:
        top = df.nlargest(5, "edge")   # show best even if below threshold

    W = 76
    line_char = "─"
    print()
    print("╔" + "═" * W + "╗")
    print(f"║{'  ⚾  MLB PROP MODEL — TOP 5 EDGES   ' + target_date:^{W}}║")
    print("╠" + "═" * W + "╣")
    hdr = f"  {'#':<3} {'Player':<22} {'Market':<22} {'Line':>5}  {'Mkt%':>6}  {'Model%':>7}  {'Edge%':>6}  {'Home':>4}"
    print(f"║{hdr:<{W}}║")
    print("╠" + line_char * W + "╣")

    for rank, (_, row) in enumerate(top.iterrows(), 1):
        player  = str(row.get("player", "—"))[:21]
        market  = str(row.get("market", "—"))[:21]
        line    = f"{row.get('line', '—')}"
        mkt_pct = f"{row.get('no_vig_over_prob', 0) * 100:.1f}%"
        mdl_pct = f"{row.get('model_prob', 0) * 100:.1f}%"
        edg_pct = f"{row.get('edge', 0) * 100:+.1f}%"
        home    = str(row.get("home_team", "?"))[:4]
        body    = f"  {rank:<3} {player:<22} {market:<22} {line:>5}  {mkt_pct:>6}  {mdl_pct:>7}  {edg_pct:>6}  {home:>4}"
        print(f"║{body:<{W}}║")

    print("╠" + line_char * W + "╣")
    avg_edge   = top["edge"].mean() * 100
    qualifying = len(df[df["edge"] >= EV_THRESHOLD])
    note = f"  {qualifying} play(s) ≥{EV_THRESHOLD:.0%} EV  |  max edge {MAX_EDGE:.0%}  |  Avg edge (top 5): {avg_edge:+.2f}%"
    print(f"║{note:<{W}}║")
    print("╚" + "═" * W + "╝")
    print()


def export_backtest_results_csv(
    df: pd.DataFrame,
    target_date: str,
    output_path: Path,
) -> None:
    """
    Generate data/processed/backtest_results.csv aligned with the NBA module.

    Columns: Month | Games | Accuracy | ROI

    Since live data doesn't have known outcomes, we build model-implied metrics:
      • Games    = distinct event_ids with edge ≥ EV_THRESHOLD
      • Accuracy = mean(model_prob) for qualifying bets
                   (expected win rate; compare to actual after game settles)
      • ROI      = mean(edge) for qualifying bets × 100 (%)

    Rows are grouped by month derived from commence_time (or target_date).
    One row is also written for the current date as "today's slate."
    """
    if df.empty:
        print("  [process] No data to export for backtest_results.csv.")
        return

    qual = df[df["edge"] >= EV_THRESHOLD].copy()

    # parse commence_time; fall back to target_date for all rows
    if "commence_time" in qual.columns:
        qual["_dt"] = pd.to_datetime(
            qual["commence_time"], errors="coerce", utc=True
        )
    else:
        qual["_dt"] = pd.Timestamp(target_date, tz="UTC")

    qual["Month"] = qual["_dt"].dt.strftime("%Y-%m")

    records: list[dict] = []
    for month, grp in qual.groupby("Month"):
        n_games  = grp["event_id"].nunique() if "event_id" in grp.columns else len(grp)
        accuracy = round(float(grp["model_prob"].mean()) * 100, 2)
        roi      = round(float(grp["edge"].mean()) * 100, 2)
        records.append({
            "Month":    month,
            "Games":    n_games,
            "Accuracy": accuracy,   # % expected win rate (model-implied)
            "ROI":      roi,        # % expected ROI per unit wagered
        })

    # if no month grouping worked, write a single today row
    if not records:
        records.append({
            "Month":    target_date[:7],
            "Games":    len(qual),
            "Accuracy": round(float(qual["model_prob"].mean()) * 100, 2) if not qual.empty else 0,
            "ROI":      round(float(qual["edge"].mean()) * 100, 2)       if not qual.empty else 0,
        })

    out_df = pd.DataFrame(records, columns=["Month", "Games", "Accuracy", "ROI"])
    out_df.to_csv(output_path, index=False)
    print(f"  → backtest_results.csv saved: {output_path}  ({len(out_df)} month rows)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run(target_date: str) -> pd.DataFrame:
    print(f"[process_model] Running for {target_date}…")

    # ── load ──────────────────────────────────────────────────────────────────
    print("  Loading raw data…")
    kalshi_df = load_kalshi(RAW_DIR / "kalshi_candles.json")
    odds_df   = load_odds(RAW_DIR / "odds_props.json")
    h2h_df    = load_h2h(RAW_DIR)   # 2025 + 2026 combined — kept for historical feature stats
    weather   = load_weather(RAW_DIR, target_date)

    # ── filter odds_df to target_date only ───────────────────────────────────
    # commence_time is an ISO string (e.g. "2026-05-05T17:10:00Z"); a simple
    # string-contains check handles any timezone suffix without needing to parse.
    if "commence_time" in odds_df.columns and not odds_df.empty:
        odds_df = odds_df[
            odds_df["commence_time"].astype(str).str.contains(target_date, na=False)
        ].copy()

    # ── build current_season_h2h: 2026 rows matching target_date only ────────
    # Using both the season AND date filters prevents calendar-date collisions
    # (e.g. May 5 2025: MIL vs ARI vs May 5 2026: MIL vs STL).
    # ── current_season_h2h: 2026 rows for target_date only ───────────────────
    # date_parsed was built in load_h2h by parsing "Tuesday, May 5" → datetime
    # with the year forced to the season, so comparing against target_date
    # (ISO format "2026-05-05") is now unambiguous.
    _target_dt = pd.Timestamp(target_date)
    if "date_parsed" in h2h_df.columns:
        current_season_h2h = h2h_df[
            (h2h_df["season"] == 2026) &
            (h2h_df["date_parsed"].dt.normalize() == _target_dt)
        ].copy()
    else:
        current_season_h2h = h2h_df[h2h_df["season"] == 2026].copy()

    if current_season_h2h.empty:
        print(
            f"  [process_model] WARNING: no 2026 H2H rows found for {target_date}.\n"
            f"    Home/away labels will fall back to Odds API values.\n"
            f"    Check that h2h_2026.csv contains games for this date."
        )

    print(f"  Kalshi rows:   {len(kalshi_df):,}")
    print(f"  Odds rows:     {len(odds_df):,}")
    print(f"  H2H rows:      {len(h2h_df):,}")
    print(f"  2026 today:    {len(current_season_h2h):,} matchup(s) for {target_date}")
    print(f"  Weather teams: {len(weather)}")

    # ── guard: no active props ────────────────────────────────────────────────
    if odds_df.empty:
        print(
            f"\n[process_model] No active props found for {target_date}. "
            f"Proceeding with historical analysis only.\n"
            f"  (To fetch fresh props: python fetch_all_data.py --streams odds)"
        )
        # return pd.DataFrame()  ← disabled: pipeline continues to golden.csv +
        # summary JSON even without live props (historical-only mode)

    # ── feature engineering ───────────────────────────────────────────────────
    print("  Engineering features…")
    iso_map = calculate_team_iso(h2h_df)
    if iso_map:
        top_iso = sorted(iso_map.items(), key=lambda x: -x[1])[:5]
        iso_str = ", ".join(f"{t}={v:.3f}" for t, v in top_iso)
        print(f"  ISO (top 5 teams): {iso_str}")

    # historical h2h_df (2025+2026) drives all feature calculations (ISO, BvP, etc.)
    odds_df = engineer_features(odds_df, h2h_df)
    odds_df = apply_park_and_wind(odds_df, weather)
    odds_df = compute_final_edge(odds_df)

    # ── resolve home_team / away_team / venue from current_season_h2h ────────
    # current_season_h2h is filtered to 2026 + target_date, so every home_team
    # maps to exactly one opponent — no cross-year collisions possible.
    # If current_season_h2h is empty (warning already printed above) we skip
    # the merge and keep whatever home/away values came from the Odds API.
    if not current_season_h2h.empty and "home_team" in odds_df.columns:
        available_venue_cols = [
            c for c in ("away_team", "venue")
            if c in current_season_h2h.columns
        ]
        if available_venue_cols:
            venue_lookup = (
                current_season_h2h[["home_team"] + available_venue_cols]
                .drop_duplicates(subset=["home_team"])
            )
            # remove stale away_team / venue from odds_df before re-merging
            stale_cols = [c for c in available_venue_cols if c in odds_df.columns]
            if stale_cols:
                odds_df = odds_df.drop(columns=stale_cols)
            odds_df = odds_df.merge(venue_lookup, on="home_team", how="left")

    # ── merge Kalshi implied probability ──────────────────────────────────────
    if not kalshi_df.empty:
        # latest close per market (most current Kalshi signal)
        latest_k = (
            kalshi_df.sort_values("ts", ascending=False)
            .groupby("market_ticker", as_index=False)
            .first()[["market_ticker", "kalshi_prob", "kalshi_pct", "home_team"]]
            .rename(columns={
                "home_team": "kalshi_home",
                "kalshi_prob": "kalshi_prob",
                "kalshi_pct":  "kalshi_pct",
            })
        )
        # join on canonical home_team where possible
        if "home_team" in odds_df.columns and not latest_k.empty:
            odds_df = odds_df.merge(
                latest_k[["kalshi_home", "kalshi_prob", "kalshi_pct"]],
                left_on="home_team",
                right_on="kalshi_home",
                how="left",
            ).drop(columns=["kalshi_home"], errors="ignore")
        else:
            odds_df["kalshi_prob"] = np.nan
            odds_df["kalshi_pct"]  = np.nan
    else:
        odds_df["kalshi_prob"] = np.nan
        odds_df["kalshi_pct"]  = np.nan

    # ── save golden CSV ───────────────────────────────────────────────────────
    golden_path = PROCESSED_DIR / "golden.csv"
    odds_df.to_csv(golden_path, index=False)
    print(f"  → Golden CSV: {golden_path}  ({len(odds_df):,} rows)")

    # ── backtest_results.csv (NBA-aligned format) ─────────────────────────────
    export_backtest_results_csv(
        odds_df,
        target_date,
        PROCESSED_DIR / "backtest_results.csv",
    )

    # ── summary JSON ──────────────────────────────────────────────────────────
    top10_cols = ["event_id", "player", "market", "line",
                  "no_vig_over_prob", "model_prob", "edge",
                  "iso_adj", "park_wind_mult", "home_team"]

    if not odds_df.empty:
        summary = {
            "date":               target_date,
            "total_props":        len(odds_df),
            "qualifying_count":    int((odds_df["edge"] >= EV_THRESHOLD).sum()),
            "markets_seen":       sorted(odds_df["market"].unique().tolist()),
            "avg_model_prob":     round(float(odds_df["model_prob"].mean()), 4),
            "avg_edge":           round(float(odds_df["edge"].mean()), 4),
            "avg_iso_adj":        round(float(odds_df["iso_adj"].mean()), 5),
            "top_edges": (
                odds_df[(odds_df["edge"] >= EV_THRESHOLD) & (odds_df["edge"] <= MAX_EDGE)]
                .nlargest(10, "edge")[
                    [c for c in top10_cols if c in odds_df.columns]
                ].to_dict(orient="records")
            ),
            "h2h_seasons":        sorted(h2h_df["season"].unique().tolist()) if not h2h_df.empty else [],
            "kalshi_markets":     int(kalshi_df["market_ticker"].nunique()) if not kalshi_df.empty else 0,
            "team_iso":           {k: round(v, 4) for k, v in iso_map.items()},
        }
    else:
        # historical-only mode — no live props, write a minimal summary so
        # downstream tools (generate_dashboard, picks_history) don't crash
        summary = {
            "date":             target_date,
            "total_props":      0,
            "qualifying_count": 0,
            "markets_seen":     [],
            "avg_model_prob":   0.0,
            "avg_edge":         0.0,
            "avg_iso_adj":      0.0,
            "top_edges":        [],
            "h2h_seasons":      sorted(h2h_df["season"].unique().tolist()) if not h2h_df.empty else [],
            "kalshi_markets":   0,
            "team_iso":         {k: round(v, 4) for k, v in iso_map.items()},
        }

    summary_path = PROCESSED_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"  → Summary JSON: {summary_path}")

    # ── Top 5 terminal table (only when live props exist) ─────────────────────
    if not odds_df.empty:
        print_top5_edges(odds_df, target_date)
    else:
        print(f"  [historical mode] No props to rank — H2H ISO computed for {len(iso_map)} teams.")

    return odds_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MLB prop model — feature engineering and edge filtering."
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Model date (YYYY-MM-DD).  Aligns with weather data.  Defaults to today.",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Skip auto-generating the HTML dashboard after the model run.",
    )
    args = parser.parse_args()
    run(args.date)

    # ── Auto-generate dashboard unless explicitly skipped ─────────────────────
    if not args.no_dashboard:
        try:
            import importlib.util, sys as _sys
            _spec = importlib.util.spec_from_file_location(
                "generate_dashboard",
                Path(__file__).resolve().parent / "generate_dashboard.py",
            )
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            _mod.main()
        except Exception as _e:
            print(f"  [process_model] Dashboard generation skipped: {_e}")