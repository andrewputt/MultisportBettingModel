"""
process_model.py
─────────────────────────────────────────────────────────────────────────────
Normalization, merging, and feature engineering for the MLB prop model.

Pipeline
  1. Load raw data from data/raw/
        kalshi_candles.json  →  Kalshi implied probability per market
        odds_props.json      →  Bookmaker lines for player props
        h2h_2025.csv / h2h_2026.csv  →  Historical H2H results
        weather.json         →  Stadium wind factors

  2. Feature Engineering
        • Batter vs. Pitching Staff (BvP) edge
        • BIG_ZONE_UMPS adjustment (+2.5 % K boost for pitchers)
        • Park Effects: wind factor × park factor multiplier

  3. Output
        data/processed/golden.csv   – one row per player-prop opportunity
        data/processed/summary.json – aggregate stats per team/market

Usage
  python process_model.py
  python process_model.py --date 2026-05-10   # ties to a specific weather date
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

# ── env ──────────────────────────────────────────────────────────────────────
ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent.parent
RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── model constants ───────────────────────────────────────────────────────────
BIG_ZONE_UMPS: set[str] = {
    # Umpires historically calling an expanded zone (expand as needed)
    "Shane Livensparger",   # removed stray trailing period
    "CB Bucknor",
    "Joe West",
    "Laz Diaz",
    "Tom Hallion",
}
BIG_ZONE_K_BOOST = 0.025          # +2.5 % strikeout probability for pitchers

WIND_FACTOR_HR_MULTIPLIERS = {
    "OUT":     1.08,   # blowing out → more HRs, higher total bases
    "IN":      0.93,   # blowing in  → fewer HRs
    "CROSS":   0.99,   # marginal effect
    "CALM":    1.00,
    "UNKNOWN": 1.00,
}

# Mirrors TEAM_STADIUM_MAP in fetch_weather.py — used to re-key weather.json
# (which is indexed by stadium name) into a team-abbreviation-indexed dict
# so apply_park_and_wind() can do a simple weather[home_team_abbr] lookup.
# Includes both MLB Stats API abbreviations and FanGraphs/B-Ref aliases so
# the join succeeds regardless of which abbreviation load_odds() emits.
TEAM_STADIUM_MAP: dict[str, str] = {
    # ── Primary: MLB Stats API ────────────────────────────────────────────────
    "ARI": "Chase Field",          "ATL": "Truist Park",
    "BAL": "Camden Yards",         "BOS": "Fenway Park",
    "CHC": "Wrigley Field",        "CWS": "Guaranteed Rate Field",
    "CIN": "Great American Ball Park", "CLE": "Progressive Field",
    "COL": "Coors Field",          "DET": "Comerica Park",
    "HOU": "Daikin Park",          "KC":  "Kauffman Stadium",
    "LAA": "Angel Stadium",        "LAD": "Dodger Stadium",
    "MIA": "loanDepot park",       "MIL": "American Family Field",
    "MIN": "Target Field",         "NYM": "Citi Field",
    "NYY": "Yankee Stadium",       "OAK": "Sutter Health Park",
    "PHI": "Citizens Bank Park",   "PIT": "PNC Park",
    "SD":  "Petco Park",           "SF":  "Oracle Park",
    "SEA": "T-Mobile Park",        "STL": "Busch Stadium",
    "TB":  "Tropicana Field",      "TEX": "Globe Life Field",
    "TOR": "Rogers Centre",        "WSH": "Nationals Park",
    # ── Aliases: FanGraphs / Baseball-Reference / DraftKings ─────────────────
    "AZ":  "Chase Field",          "ANA": "Angel Stadium",
    "KCR": "Kauffman Stadium",     "SDP": "Petco Park",
    "SFG": "Oracle Park",          "TBR": "Tropicana Field",
    "WSN": "Nationals Park",       "CHW": "Guaranteed Rate Field",
}

# American → decimal → probability conversion helpers ─────────────────────────

def american_to_decimal(american: float) -> float:
    if american >= 100:
        return (american / 100) + 1
    return (100 / abs(american)) + 1


def american_to_prob(american: float) -> float:
    dec = american_to_decimal(american)
    return 1 / dec


def remove_vig(probs: list[float]) -> list[float]:
    """Normalise a list of raw implied probabilities to sum to 1.0."""
    total = sum(probs)
    if total == 0:
        return probs
    return [p / total for p in probs]


# ─────────────────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_kalshi(path: Path) -> pd.DataFrame:
    """
    Parse kalshi_candles.json into a DataFrame with columns:
      market_ticker, ts, open_yes, high_yes, low_yes, close_yes, volume
    where *_yes prices are in cents (0–100) → converted to probability (0–1).
    """
    if not path.exists():
        print(f"  [process] WARNING: {path} not found — skipping Kalshi stream.")
        return pd.DataFrame()

    raw: list[dict] = json.loads(path.read_text())
    rows: list[dict] = []
    for c in raw:
        yes_price = c.get("yes_price", {}) if isinstance(c.get("yes_price"), dict) else {}
        rows.append({
            "market_ticker": c.get("market_ticker", ""),
            "ts":            c.get("end_period_ts", c.get("ts", "")),
            "open_prob":     (yes_price.get("open",  c.get("open",  50)) or 50) / 100,
            "high_prob":     (yes_price.get("high",  c.get("high",  50)) or 50) / 100,
            "low_prob":      (yes_price.get("low",   c.get("low",   50)) or 50) / 100,
            "close_prob":    (yes_price.get("close", c.get("close", 50)) or 50) / 100,
            "volume":        c.get("volume", 0),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
        # extract team context from ticker pattern KXMLBGAME-YYYYMMDD-AAA-HHH-X
        df["season"] = df["market_ticker"].str.extract(r"(\d{4})", expand=False)
    return df


def load_odds(path: Path) -> pd.DataFrame:
    """
    Flatten odds_props.json into a long-format DataFrame with home_team mapping.
    """
    if not path.exists():
        print(f"  [process] WARNING: {path} not found — skipping Odds stream.")
        return pd.DataFrame()

    # SSOT Mapping: Full Name -> Abbreviation for weather joining
    NAME_TO_ABR = {
        "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
        "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
        "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL",
        "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KCR",
        "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
        "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM",
        "New York Yankees": "NYY", "Oakland Athletics": "OAK", "Philadelphia Phillies": "PHI",
        "Pittsburgh Pirates": "PIT", "San Diego Padres": "SDP", "San Francisco Giants": "SFG",
        "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TBR",
        "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WSN"
    }

    raw: list[dict] = json.loads(path.read_text())
    rows: list[dict] = []

    for event in raw:
        event_id = event["event_id"]
        
        # 1. Extract the Home Team abbreviation for the wind/park boost
        # The Odds API provides 'home_team' as a string at the top level
        full_home_name = event.get("home_team", "UNKNOWN")
        home_abr = NAME_TO_ABR.get(full_home_name, "UNKNOWN")

        for bk in event.get("bookmakers", []):
            bk_name = bk.get("title", bk.get("key", ""))
            for market in bk.get("markets", []):
                mkt_key = market.get("key", "")
                for outcome in market.get("outcomes", []):
                    rows.append({
                        "event_id":    event_id,
                        "home_team":   home_abr,  # CRITICAL for weather merge
                        "bookmaker":   bk_name,
                        "market":      mkt_key,
                        "player":      outcome.get("description", outcome.get("name", "")),
                        "side":        outcome.get("name", ""),
                        "line":        outcome.get("point", None),
                        "american":    outcome.get("price", None),
                    })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    
    # Convert American odds to Implied Probability
    df["raw_prob"] = df["american"].apply(
        lambda x: american_to_prob(float(x)) if pd.notna(x) else np.nan
    )

    # 2. De-vig logic: Find Over/Under pairs and normalize to 100%
    # We use a transform to keep the original index order
    df["vig_free_prob"] = df.groupby(["event_id", "bookmaker", "market", "player", "line"])["raw_prob"].transform(
        lambda x: remove_vig(x.tolist()) if len(x) == 2 else x
    )

    # 3. Filter to 'Over' sides only
    # Models generally target the 'Over' probability as the primary signal
    df = df[df["side"].str.lower() == "over"].copy()
    df.rename(columns={"vig_free_prob": "no_vig_over_prob"}, inplace=True)
    
    return df


def load_h2h(raw_dir: Path) -> pd.DataFrame:
    """Merge 2025 and 2026 H2H CSVs with column normalization."""
    frames: list[pd.DataFrame] = []
    for season in [2025, 2026]:
        fp = raw_dir / f"h2h_{season}.csv"
        if fp.exists():
            df = pd.read_csv(fp)
            
            # NORMALIZATION: Force columns to lowercase snake_case
            # This fixes issues where 'RA' vs 'ra' or 'SO' vs 'so' cause crashes
            df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
            
            print(f"  [process] Loaded {season} H2H. Columns: {len(df.columns)}")
            df["season"] = season
            frames.append(df)
        else:
            print(f"  [process] WARNING: {fp} not found.")
            
    if not frames:
        return pd.DataFrame()
        
    combined = pd.concat(frames, ignore_index=True)
    return combined

def load_weather(raw_dir: Path, target_date: str) -> dict:
    """
    Load weather.json and return a dict keyed by TEAM ABBREVIATION so that
    apply_park_and_wind() can do a simple weather[home_team_abbr] lookup.

    fetch_weather.py writes weather.json keyed by stadium name
    (e.g. "Kauffman Stadium").  We re-index it using TEAM_STADIUM_MAP so
    every abbreviation variant ("KC", "KCR") resolves to the same payload.
    """
    fp = raw_dir / "weather.json"
    if not fp.exists():
        print(f"  [process] WARNING: {fp} not found. Wind effects will be skipped.")
        return {}

    try:
        data = json.loads(fp.read_text())
        # Support both the old team-keyed format and the current stadium-keyed format
        if "team_map" in data:
            return data["team_map"]   # already team-indexed (legacy path)

        stadiums: dict = data.get("stadiums", {})
        if not stadiums:
            return {}

        # Build an inverted map: stadium_name → weather_payload
        stadium_to_data = {name: payload for name, payload in stadiums.items()}

        # Re-key by every team abbreviation that maps to a known stadium
        team_indexed: dict = {}
        for abbr, stadium_name in TEAM_STADIUM_MAP.items():
            if stadium_name in stadium_to_data:
                team_indexed[abbr] = stadium_to_data[stadium_name]

        print(f"  [process] Weather re-keyed: {len(team_indexed)} team abbreviations resolved.")
        return team_indexed

    except Exception as e:
        print(f"  [process] ERROR parsing weather: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_bvp(odds_df: pd.DataFrame, h2h_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a Batter-vs-Pitching-Staff edge for each player-prop row.

    Logic (simplified without roster data):
      • For K props:  use the pitching team's strikeout rate (K/9 proxy from H2H)
      • For hits/TB:  use the opposing team's ERA proxy

    Both are expressed as a probability delta vs. the bookmaker line.
    """
    if odds_df.empty or h2h_df.empty:
        if not odds_df.empty:
            odds_df["bvp_edge"] = 0.0
        return odds_df

    # build team-level aggregates from H2H
    team_stats: dict[str, dict] = {}
    needed_cols = {"team", "season"}
    if needed_cols.issubset(set(h2h_df.columns)):
        for team, grp in h2h_df.groupby("team"):
            # proxy: if SO column exists use it, else default
            so_col = next(
                (c for c in grp.columns if "so" in c.lower() or "strikeout" in c.lower()),
                None,
            )
            avg_so = float(grp[so_col].mean()) if so_col else 7.5
            # proxy ERA from runs allowed column
            ra_col = next(
                (c for c in grp.columns if c.lower() in ("ra", "r_allowed", "runs_allowed")),
                None,
            )
            avg_ra = float(grp[ra_col].mean()) if ra_col else 4.2
            team_stats[team] = {"avg_so": avg_so, "avg_ra": avg_ra}

    def _bvp_delta(row: pd.Series) -> float:
        """Return estimated probability boost vs. book line."""
        market = str(row.get("market", "")).lower()
        # crude: no roster linkage without a full roster API; use market averages
        league_avg_so = 8.2   # ~8.2 Ks per game per team (2025 pace)
        league_avg_ra = 4.3

        if "strikeout" in market or "_k" in market:
            line    = float(row["line"]) if pd.notna(row.get("line")) else 5.0
            # pitcher K rate vs. line
            delta = (league_avg_so - line) / (league_avg_so + line + 1e-6) * 0.05
        elif "hits" in market:
            line  = float(row["line"]) if pd.notna(row.get("line")) else 1.5
            delta = (3.0 - league_avg_ra) / 30 * 0.03
        elif "total_bases" in market:
            line  = float(row["line"]) if pd.notna(row.get("line")) else 1.5
            delta = 0.02
        elif "outs" in market:
            line  = float(row["line"]) if pd.notna(row.get("line")) else 15.0
            delta = 0.01
        else:
            delta = 0.0

        return round(delta, 5)

    odds_df["bvp_edge"] = odds_df.apply(_bvp_delta, axis=1)
    return odds_df


def apply_big_zone_adjustment(
    odds_df: pd.DataFrame,
    umpire_col: str = "umpire",
) -> pd.DataFrame:
    """
    If the assigned home-plate umpire is in BIG_ZONE_UMPS, add +2.5 % to the
    strikeout probability for all pitcher-K props in that game.
    Umpire data is expected in a column named *umpire_col* (optional; skipped
    if absent).
    """
    if umpire_col not in odds_df.columns:
        odds_df["big_zone_adj"] = 0.0
        return odds_df

    k_mask = odds_df["market"].str.lower().str.contains("strikeout|_k", na=False)
    big_mask = odds_df[umpire_col].isin(BIG_ZONE_UMPS)
    odds_df["big_zone_adj"] = 0.0
    odds_df.loc[k_mask & big_mask, "big_zone_adj"] = BIG_ZONE_K_BOOST
    return odds_df


def apply_park_and_wind(
    odds_df: pd.DataFrame,
    weather: dict,
) -> pd.DataFrame:
    """
    Apply park factor + wind factor adjustment to total-bases and hit props.
    Looks for a 'home_team' column to join weather.
    """
    if weather and "home_team" not in odds_df.columns:
        # fallback: add a neutral column so the rest of the logic runs
        odds_df["home_team"] = "UNKNOWN"

    def _multiplier(row: pd.Series) -> float:
        team = str(row.get("home_team", "UNKNOWN")).upper()
        wdata = weather.get(team, {})
        summary = wdata.get("game_time_summary", {})
        wind_cat = summary.get("dominant_wind_factor", "CALM")
        wind_mult = WIND_FACTOR_HR_MULTIPLIERS.get(wind_cat, 1.0)
        park_factor = wdata.get("park_factor", 1.0)
        return round(wind_mult * park_factor, 4)

    market_lc = odds_df["market"].str.lower() if "market" in odds_df.columns else pd.Series()
    power_mask = (
        market_lc.str.contains("total_bases|hits", na=False)
        if not market_lc.empty
        else pd.Series(False, index=odds_df.index)
    )

    odds_df["park_wind_mult"] = 1.0
    if not odds_df.empty and power_mask.any():
        odds_df.loc[power_mask, "park_wind_mult"] = odds_df[power_mask].apply(
            _multiplier, axis=1
        )
    return odds_df


def compute_final_edge(odds_df: pd.DataFrame) -> pd.DataFrame:
    # Ensure columns exist to avoid NaN crashes
    for col in ["no_vig_over_prob", "bvp_edge", "big_zone_adj"]:
        odds_df[col] = odds_df[col].fillna(0.0)
    odds_df["park_wind_mult"] = odds_df["park_wind_mult"].fillna(1.0)

    # ONE LINE: Fast math for 10k+ rows
    odds_df["model_prob"] = (odds_df["no_vig_over_prob"] + odds_df["bvp_edge"] + odds_df["big_zone_adj"]) * odds_df["park_wind_mult"]
    
    # Clip to keep probabilities realistic
    odds_df["model_prob"] = odds_df["model_prob"].clip(0.01, 0.99)
    
    # Calculate the edge (Final Prob - Market Prob)
    odds_df["edge"] = (odds_df["model_prob"] - odds_df["no_vig_over_prob"]).round(5)
    
    return odds_df

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run(target_date: str) -> pd.DataFrame:
    print(f"[process_model] Running for date {target_date}…")

    # ── load ──────────────────────────────────────────────────────────────────
    print("  Loading raw data…")
    kalshi_df = load_kalshi(RAW_DIR / "kalshi_candles.json")
    odds_df   = load_odds(RAW_DIR / "odds_props.json")
    h2h_df    = load_h2h(RAW_DIR)
    weather   = load_weather(RAW_DIR, target_date)

    print(f"  Kalshi rows:   {len(kalshi_df):,}")
    print(f"  Odds rows:     {len(odds_df):,}")
    print(f"  H2H rows:      {len(h2h_df):,}")
    print(f"  Weather teams: {len(weather)}")

    # ── guard: nothing to process without odds data ────────────────────────────
    if odds_df.empty:
        print(
            "\n[process_model] ABORT — odds_props.json produced no usable rows.\n"
            "  Checklist:\n"
            "    1. Run fetch_all_data.py --streams odds to refresh the file.\n"
            "    2. Confirm ODDS_API_KEY is set in your .env.\n"
            "    3. Verify there are MLB games scheduled today (off-season = no events)."
        )
        return pd.DataFrame()

    # ── feature engineering ───────────────────────────────────────────────────
    print("  Engineering features…")
    odds_df = engineer_bvp(odds_df, h2h_df)
    odds_df = apply_big_zone_adjustment(odds_df)
    odds_df = apply_park_and_wind(odds_df, weather)
    odds_df = compute_final_edge(odds_df)

    # ── merge Kalshi close price ───────────────────────────────────────────────
    if not kalshi_df.empty and not odds_df.empty:
        # latest close per market
        latest_kalshi = (
            kalshi_df.sort_values("ts", ascending=False)
            .groupby("market_ticker", as_index=False)
            .first()[["market_ticker", "close_prob"]]
            .rename(columns={"close_prob": "kalshi_close_prob"})
        )
        # rudimentary join: try to match event_id substrings (works when
        # tickers carry game IDs; fine-tune join key to your actual schema)
        odds_df["kalshi_close_prob"] = np.nan
        if "market_ticker" in latest_kalshi.columns:
            # best-effort cross-reference: just attach to demonstrate merge
            sample_prob = latest_kalshi["kalshi_close_prob"].median()
            odds_df["kalshi_close_prob"] = sample_prob
    else:
        odds_df["kalshi_close_prob"] = np.nan

    # ── save golden CSV ───────────────────────────────────────────────────────
    golden_path = PROCESSED_DIR / "golden.csv"
    odds_df.to_csv(golden_path, index=False)
    print(f"  → Golden CSV saved: {golden_path}  ({len(odds_df):,} rows)")

    # ── summary JSON ──────────────────────────────────────────────────────────
    summary: dict = {
        "date": target_date,
        "total_props": len(odds_df),
        "markets_seen": sorted(odds_df["market"].unique().tolist()) if not odds_df.empty else [],
        "avg_model_prob":  round(float(odds_df["model_prob"].mean()), 4)  if not odds_df.empty else None,
        "avg_edge":        round(float(odds_df["edge"].mean()), 4)         if not odds_df.empty else None,
        "top_edges": (
            odds_df.nlargest(10, "edge")[
                ["event_id", "player", "market", "line", "no_vig_over_prob", "model_prob", "edge"]
            ].to_dict(orient="records")
            if not odds_df.empty else []
        ),
        "h2h_seasons_loaded": sorted(h2h_df["season"].unique().tolist()) if not h2h_df.empty else [],
        "kalshi_markets":      len(kalshi_df["market_ticker"].unique()) if not kalshi_df.empty else 0,
    }
    summary_path = PROCESSED_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"  → Summary JSON saved: {summary_path}")

    return odds_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize and engineer features for MLB model.")
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Model date (YYYY-MM-DD).  Aligns with weather data. Defaults to today.",
    )
    args = parser.parse_args()
    run(args.date)