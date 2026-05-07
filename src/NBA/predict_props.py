"""
predict_props.py
────────────────────────────────────────────────────────────────────────────
Fetches today's NBA player prop and combo markets from Kalshi, builds real
game-context features (rest days, opp def rating, home/away, game-in-series),
applies lineup context adjustments for absent players, runs the trained XGBoost
models with per-player residual std, applies half-Kelly sizing, and logs all
predictions.

Individual props:  PTS, REB, AST, FG3M, STL, BLK
Combo props:       Double-Double (KXNBA2D), Triple-Double (KXNBA3D)

Lineup context (Kalshi-only):
  Players who appeared in recent series games but have no props today are
  flagged as absent. Their average stat load is redistributed evenly across
  active teammates, giving a usage-bump to listed players.

Usage:
    python src/NBA/predict_props.py
    python src/NBA/predict_props.py --game 26MAY03TORCLE
    python src/NBA/predict_props.py --min-edge 0.05
"""

import argparse
import json
import os
import pickle
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm
from nba_api.stats.endpoints import leaguegamelog, playergamelog
from nba_api.stats.static import players as nba_players

# ── Load .env ──────────────────────────────────────────────────────────────────
def _load_env():
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
_load_env()

MODEL_DIR   = Path("src/NBA/models")
DATA_DIR    = Path("src/NBA/data")
PROPS_LOG   = DATA_DIR / "props_log.csv"
PROPS_TODAY = DATA_DIR / "props_edges_today.csv"

KALSHI_BASE   = "https://api.elections.kalshi.com/trade-api/v2"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_CACHE    = DATA_DIR / "odds_cache.json"
ODDS_API_KEY  = os.getenv("ODDS_API_KEY", "")

# Sportsbook preference order (sharpest books first)
BOOKS_PRIORITY = ["pinnacle", "draftkings", "fanduel", "betmgm", "caesars"]

ODDS_STAT_MAP = {
    "player_points":   "PTS",
    "player_rebounds": "REB",
    "player_assists":  "AST",
    "player_threes":   "FG3M",
    "player_blocks":   "BLK",
}

TEAM_FULL_TO_ABB = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

PROP_STATS = ["PTS", "REB", "AST", "FG3M", "BLK"]

COMBO_SERIES = {
    "DD": "KXNBA2D",
    "TD": "KXNBA3D",
}

CURRENT_SEASON = "2025-26"
SEASON_NUM     = 5

MIN_SERIES_GAMES_FOR_ADJUSTMENT = 2


# ── Game-context helpers ───────────────────────────────────────────────────────

def fetch_current_season_team_logs() -> pd.DataFrame:
    print("Fetching 2025-26 team logs for game context ...")
    frames = []
    for stype in ["Regular Season", "Playoffs"]:
        try:
            df = leaguegamelog.LeagueGameLog(
                season=CURRENT_SEASON,
                season_type_all_star=stype,
                player_or_team_abbreviation="T",
                timeout=60,
            ).get_data_frames()[0]
            df["IS_PLAYOFF"] = int(stype == "Playoffs")
            frames.append(df)
            time.sleep(1.2)
        except Exception as e:
            print(f"  Warning: {stype} team logs: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)


def build_team_context(team_logs: pd.DataFrame) -> dict:
    if team_logs.empty:
        return {}
    # Self-join to get opponent stats per game (Fix 3: add AST + REB allowed)
    opp = team_logs[["GAME_ID", "TEAM_ABBREVIATION", "PTS", "AST", "REB"]].copy()
    opp.columns = ["GAME_ID", "OPP_TEAM", "OPP_PTS", "OPP_AST", "OPP_REB"]
    tl = team_logs.merge(opp, on="GAME_ID", how="left")
    tl = tl[tl["TEAM_ABBREVIATION"] != tl["OPP_TEAM"]]
    for col in ["OPP_PTS", "OPP_AST", "OPP_REB"]:
        tl[f"{col}_ROLL"] = (
            tl.groupby("TEAM_ABBREVIATION")[col]
            .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
        )
    ctx = {}
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    for team, grp in tl.groupby("TEAM_ABBREVIATION"):
        grp     = grp.sort_values("GAME_DATE")
        last    = grp.iloc[-1]
        rest    = max(0, (today - last["GAME_DATE"]).days)
        opp_def = float(last.get("OPP_PTS_ROLL", np.nan))
        opp_ast = float(last.get("OPP_AST_ROLL", np.nan))
        opp_reb = float(last.get("OPP_REB_ROLL", np.nan))
        if pd.isna(opp_def): opp_def = 110.0
        if pd.isna(opp_ast): opp_ast = 25.0
        if pd.isna(opp_reb): opp_reb = 44.0
        po_grp  = grp[grp["IS_PLAYOFF"] == 1]
        game_in_s  = 0
        series_w   = 0
        if not po_grp.empty:
            last_opp     = last.get("OPP_TEAM", "")
            series_games = po_grp[po_grp["OPP_TEAM"] == last_opp]
            game_in_s    = len(series_games) + 1
            if "WL" in series_games.columns:
                series_w = int((series_games["WL"] == "W").sum())
        ctx[team] = {
            "rest_days":      float(rest),
            "opp_def_l10":    opp_def,
            "opp_ast_l10":    opp_ast,
            "opp_reb_l10":    opp_reb,
            "game_in_series": int(game_in_s),
            "series_wins":    series_w,
        }
    return ctx


def get_series_form(hist_df: pd.DataFrame, opp_abb: str) -> dict | None:
    """
    Fix 1: Extract per-stat averages from current-series games vs this opponent.
    Returns None if fewer than 2 series games found.
    Blend weight grows with sample size: 2 games=40%, 3=60%, 4=75%, 5+=80%.
    """
    if hist_df.empty or not opp_abb:
        return None
    mask = hist_df["MATCHUP"].str.contains(opp_abb, case=False, na=False)
    series_games = hist_df[mask]
    if len(series_games) < 2:
        return None
    n = len(series_games)
    form: dict = {"n_games": n, "weight": min(0.8, n * 0.2)}
    for col in ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "MIN", "FGA", "TOV"]:
        if col in series_games.columns:
            form[col] = float(pd.to_numeric(series_games[col], errors="coerce").mean())
    return form


def apply_opp_stat_correction(pred: float, stat: str, opp_ctx: dict) -> float:
    """
    Fix 3: Damped correction for stat-specific opponent defense.
    Only applied to AST and REB — PTS is already captured by OPP_DEF_RATING_L10
    in the model features, so we skip it to avoid double-counting.
    Damping via sqrt: a team 20% better at suppressing AST reduces prediction ~10%.
    """
    league_avgs = {"AST": 25.0, "REB": 44.0}
    opp_keys    = {"AST": "opp_ast_l10", "REB": "opp_reb_l10"}
    if stat not in opp_keys:
        return pred
    opp_val = opp_ctx.get(opp_keys[stat])
    if opp_val is None or pd.isna(opp_val):
        return pred
    correction = (opp_val / league_avgs[stat]) ** 0.5
    return pred * correction


def parse_game_teams(game_id: str) -> tuple[str, str]:
    teams_part = re.sub(r"^\d{2}[A-Z]{3}\d{2}", "", game_id)
    if len(teams_part) == 6:
        return teams_part[:3], teams_part[3:]
    return "", ""


def extract_player_team(ticker: str) -> str:
    parts = ticker.split("-")
    return parts[2][:3] if len(parts) >= 3 else ""


# ── Kalshi market helpers ──────────────────────────────────────────────────────

def fetch_kalshi_props(game_id: str | None = None) -> pd.DataFrame:
    rows = []
    prop_series_map = {"PTS":"KXNBAPTS","REB":"KXNBAREB","AST":"KXNBAAST","FG3M":"KXNBA3PT","STL":"KXNBASTL","BLK":"KXNBABLK"}
    for stat, series in prop_series_map.items():
        r = requests.get(
            f"{KALSHI_BASE}/markets",
            params={"limit": 200, "series_ticker": series},
        )
        for m in r.json().get("markets", []):
            ticker = m["ticker"]
            if game_id and game_id not in ticker:
                continue
            yes_ask = m.get("yes_ask_dollars")
            if yes_ask is None:
                continue
            match = re.match(r"^(.+?):\s*(\d+)\+", m.get("title", ""))
            if not match:
                continue
            t_parts = ticker.split("-")
            rows.append({
                "ticker":      ticker,
                "game_id":     t_parts[1] if len(t_parts) > 1 else "",
                "player_team": extract_player_team(ticker),
                "stat":        stat,
                "player":      match.group(1).strip(),
                "threshold":   float(match.group(2)),
                "yes_ask":     float(yes_ask),
            })
        time.sleep(0.3)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["player", "stat", "threshold"])


def fetch_kalshi_combos(game_id: str | None = None) -> pd.DataFrame:
    rows = []
    for combo_type, series in COMBO_SERIES.items():
        r = requests.get(
            f"{KALSHI_BASE}/markets",
            params={"limit": 200, "series_ticker": series},
        )
        for m in r.json().get("markets", []):
            ticker = m["ticker"]
            if game_id and game_id not in ticker:
                continue
            yes_ask = m.get("yes_ask_dollars")
            if yes_ask is None:
                continue
            # title: "Scottie Barnes: Double Double" or "James Harden: Triple Double"
            match = re.match(r"^(.+?):\s*(Double Double|Triple Double)", m.get("title", ""))
            if not match:
                continue
            t_parts = ticker.split("-")
            rows.append({
                "ticker":      ticker,
                "game_id":     t_parts[1] if len(t_parts) > 1 else "",
                "player_team": extract_player_team(ticker),
                "combo_type":  combo_type,
                "player":      match.group(1).strip(),
                "yes_ask":     float(yes_ask),
            })
        time.sleep(0.3)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ── Odds API (sportsbook lines for individual props) ──────────────────────────

def american_to_prob(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def remove_vig(over_p: float, under_p: float) -> tuple[float, float]:
    total = over_p + under_p
    return over_p / total, under_p / total


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation variants for fuzzy player name matching."""
    return re.sub(r"[.\s]+", " ", name).strip().lower()


def build_player_team_map() -> dict[str, str]:
    """Return {normalized_player_name: team_abb} from most recent game in player_features.csv."""
    feat = DATA_DIR / "player_features.csv"
    if not feat.exists():
        return {}
    df = pd.read_csv(feat, usecols=["PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_DATE"])
    df = df.sort_values("GAME_DATE")
    raw = df.groupby("PLAYER_NAME")["TEAM_ABBREVIATION"].last().to_dict()
    return {normalize_name(k): v for k, v in raw.items()}


def lookup_team(player_name: str, player_team_map: dict, fallback: str) -> str:
    return player_team_map.get(normalize_name(player_name), fallback)


def fetch_odds_api_props(game_id: str | None = None,
                         use_cache: bool = False) -> pd.DataFrame:
    """
    Fetch player prop lines from The Odds API using the event-level endpoint.
    Uses no-vig implied probability so edges are real, not spread artifacts.
    Costs 2 requests: 1 for events list + 1 per matched game.
    """
    if use_cache and ODDS_CACHE.exists():
        print("  Loading props from cache ...")
        raw_events = json.loads(ODDS_CACHE.read_text())
        return _parse_odds_events(raw_events, game_id)

    if not ODDS_API_KEY:
        print("  Warning: ODDS_API_KEY not set — add it to .env")
        return pd.DataFrame()

    away_abb, home_abb = parse_game_teams(game_id or "")

    # Step 1: get today's events to find the event ID
    r = requests.get(
        f"{ODDS_API_BASE}/sports/basketball_nba/events",
        params={"apiKey": ODDS_API_KEY},
        timeout=30,
    )
    remaining = r.headers.get("x-requests-remaining", "?")
    print(f"  Odds API requests remaining: {remaining}")
    events = r.json()

    # Step 2: find matching game(s) and fetch player prop odds per event
    markets = ",".join(ODDS_STAT_MAP.keys())
    all_event_data = []
    for event in events:
        if not isinstance(event, dict):
            continue
        g_home = TEAM_FULL_TO_ABB.get(event.get("home_team", ""), "")
        g_away = TEAM_FULL_TO_ABB.get(event.get("away_team", ""), "")
        if game_id and (g_home != home_abb or g_away != away_abb):
            continue
        event_id = event["id"]
        r2 = requests.get(
            f"{ODDS_API_BASE}/sports/basketball_nba/events/{event_id}/odds",
            params={
                "apiKey":     ODDS_API_KEY,
                "regions":    "us",
                "markets":    markets,
                "oddsFormat": "american",
                "bookmakers": ",".join(BOOKS_PRIORITY),
            },
            timeout=30,
        )
        remaining = r2.headers.get("x-requests-remaining", "?")
        print(f"  Fetched props for {event.get('away_team')} @ {event.get('home_team')} "
              f"({remaining} requests left)")
        all_event_data.append(r2.json())
        time.sleep(0.5)

    ODDS_CACHE.write_text(json.dumps(all_event_data))
    return _parse_odds_events(all_event_data, game_id)


def _parse_odds_events(event_data: list, game_id: str | None) -> pd.DataFrame:
    """Parse raw Odds API event response into props DataFrame."""
    away_abb, home_abb = parse_game_teams(game_id or "")
    player_team_map    = build_player_team_map()
    rows = []
    used_book = ""

    for event in event_data:
        if not isinstance(event, dict) or "bookmakers" not in event:
            continue
        g_home = TEAM_FULL_TO_ABB.get(event.get("home_team", ""), "")
        g_away = TEAM_FULL_TO_ABB.get(event.get("away_team", ""), "")

        # Pick sharpest available book
        bookmaker = None
        for bk in BOOKS_PRIORITY:
            bookmaker = next((b for b in event["bookmakers"] if b["key"] == bk), None)
            if bookmaker:
                used_book = bookmaker.get("title", bk)
                break
        if not bookmaker:
            continue

        for market in bookmaker.get("markets", []):
            stat = ODDS_STAT_MAP.get(market["key"])
            if not stat:
                continue

            # Odds API format: name="Over"/"Under", description=player_name
            sides: dict[tuple, dict] = defaultdict(dict)
            for o in market.get("outcomes", []):
                player_name = o.get("description", "")
                direction   = o.get("name", "")      # "Over" or "Under"
                sides[(player_name, float(o["point"]))][direction] = o["price"]

            for (player_name, point), book_sides in sides.items():
                if "Over" not in book_sides or "Under" not in book_sides:
                    continue
                over_raw  = american_to_prob(book_sides["Over"])
                under_raw = american_to_prob(book_sides["Under"])
                over_prob, _ = remove_vig(over_raw, under_raw)

                p_team = lookup_team(player_name, player_team_map, g_away)

                rows.append({
                    "ticker":      f"{stat}-{game_id or g_away+g_home}-{player_name}-{point}",
                    "game_id":     game_id or f"{g_away}{g_home}",
                    "player_team": p_team,
                    "stat":        stat,
                    "player":      player_name,
                    "threshold":   point,
                    "yes_ask":     round(over_prob, 4),
                })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(["player", "stat", "threshold"])
    print(f"  {len(df)} lines / {df['player'].nunique()} players  (source: {used_book})")
    return df


# ── Lineup context (Kalshi-only) ───────────────────────────────────────────────

def build_lineup_context(props: pd.DataFrame, game_id: str | None) -> dict:
    """
    Flags genuinely significant players as absent using a Kalshi-only signal:
    a player is absent only if they were a meaningful contributor in this series
    (L5 avg > MIN_ABSENT_PTS pts AND > MIN_ABSENT_MIN min) but have no prop
    markets today. Bench players that Kalshi never posts props for are excluded
    to prevent false load-redistribution inflation.

    Returns a dict keyed by team:
      {
        "absent_pts": float,
        "absent_ast": float,
        "absent_reb": float,
        "active_count": int,
        "absent_players": [str],
      }
    """
    # Thresholds: only flag players averaging this much in the series
    MIN_ABSENT_PTS = 12.0
    MIN_ABSENT_MIN = 20.0

    if props.empty or not game_id:
        return {}

    away_abb, home_abb = parse_game_teams(game_id)
    both_teams = {away_abb, home_abb} - {""}

    active_by_team: dict[str, set] = {}
    for _, row in props.iterrows():
        t = row["player_team"]
        if t not in active_by_team:
            active_by_team[t] = set()
        active_by_team[t].add(row["player"])

    feat_path = DATA_DIR / "player_features.csv"
    if not feat_path.exists():
        return {}
    feats = pd.read_csv(feat_path, parse_dates=["GAME_DATE"])
    playoff_feats = feats[(feats["IS_PLAYOFF"] == 1) & (feats["SEASON"] == CURRENT_SEASON)].copy()

    lineup_ctx: dict[str, dict] = {}

    for team in both_teams:
        active_players = active_by_team.get(team, set())
        season_po = playoff_feats[playoff_feats["TEAM_ABBREVIATION"] == team]

        # Only consider players who appeared >= MIN_SERIES_GAMES_FOR_ADJUSTMENT times
        player_counts = season_po.groupby("PLAYER_NAME").size()
        series_players = set(player_counts[player_counts >= MIN_SERIES_GAMES_FOR_ADJUSTMENT].index)

        absent = series_players - active_players
        if not absent:
            lineup_ctx[team] = {
                "absent_pts": 0.0, "absent_ast": 0.0, "absent_reb": 0.0,
                "active_count": len(active_players), "absent_players": [],
            }
            continue

        absent_pts = absent_ast = absent_reb = 0.0
        confirmed_absent = []
        for p in absent:
            p_rows = season_po[season_po["PLAYER_NAME"] == p].sort_values("GAME_DATE")
            if p_rows.empty:
                continue
            last = p_rows.iloc[-1]
            pts_avg = float(last.get("PTS_L5") or last.get("PTS_L10") or 0)
            min_avg = float(last.get("MIN_L5") or last.get("MIN_L10") or 0)

            # Skip bench players Kalshi never posts props for
            if pts_avg < MIN_ABSENT_PTS or min_avg < MIN_ABSENT_MIN:
                continue

            ast = float(last.get("AST_L5") or last.get("AST_L10") or 0)
            reb = float(last.get("REB_L5") or last.get("REB_L10") or 0)
            absent_pts += pts_avg
            absent_ast += ast
            absent_reb += reb
            confirmed_absent.append(p)

        lineup_ctx[team] = {
            "absent_pts":     absent_pts,
            "absent_ast":     absent_ast,
            "absent_reb":     absent_reb,
            "active_count":   len(active_players),
            "absent_players": confirmed_absent,
        }

    return lineup_ctx


def apply_lineup_adjustment(pred_stat: float, stat: str, team: str,
                             lineup_ctx: dict) -> float:
    """
    Distribute absent players' avg stats evenly across active players.
    Capped at +30% to avoid runaway adjustments.
    """
    ctx = lineup_ctx.get(team)
    if not ctx or not ctx["absent_players"]:
        return pred_stat
    n_active = max(ctx["active_count"], 1)
    if stat == "PTS":
        adj = ctx["absent_pts"] / n_active
    elif stat == "AST":
        adj = ctx["absent_ast"] / n_active
    elif stat == "REB":
        adj = ctx["absent_reb"] / n_active
    else:
        return pred_stat
    cap = pred_stat * 0.30
    return pred_stat + min(adj, cap)


# ── Player log helpers ─────────────────────────────────────────────────────────

def get_player_id(name: str) -> int | None:
    results = nba_players.find_players_by_full_name(name)
    if results:
        return results[0]["id"]
    last = name.split()[-1]
    results = nba_players.find_players_by_last_name(last)
    active = [p for p in results if p["is_active"]]
    if len(active) == 1:
        return active[0]["id"]
    return None


def fetch_recent_logs(player_id: int, n: int = 25) -> pd.DataFrame:
    frames = []
    for stype in ["Playoffs", "Regular Season"]:
        try:
            df = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=CURRENT_SEASON,
                season_type_all_star=stype,
                timeout=30,
            ).get_data_frames()[0]
            if not df.empty:
                frames.append(df)
        except Exception:
            pass
        time.sleep(0.5)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["GAME_DATE"] = pd.to_datetime(combined["GAME_DATE"], format="mixed")
    return combined.sort_values("GAME_DATE").tail(n)


def build_live_features(is_home: int, is_playoff: int, rest_days: float,
                        opp_def_l10: float, game_in_series: int,
                        hist_df: pd.DataFrame, feat_cols: list,
                        series_form: dict | None = None,
                        series_wins: int = 0) -> dict:
    row = {
        "IS_HOME":            is_home,
        "IS_PLAYOFF":         is_playoff,
        "GAME_IN_SERIES":     game_in_series,
        "SERIES_WINS":        series_wins,
        "REST_DAYS":          rest_days,
        "SEASON_NUM":         SEASON_NUM,
        "OPP_DEF_RATING_L10": opp_def_l10,
    }
    if hist_df.empty:
        return row
    col_map = {
        "PTS": "PTS", "REB": "REB", "AST": "AST",
        "FG3M": "FG3M", "STL": "STL", "BLK": "BLK",
        "MIN": "MIN", "FGA": "FGA", "TOV": "TOV", "FTA": "FTA",
    }
    for api_col, feat_prefix in col_map.items():
        if api_col not in hist_df.columns:
            continue
        series = pd.to_numeric(hist_df[api_col], errors="coerce")
        for w in [5, 10, 20]:
            key = f"{feat_prefix}_L{w}"
            if key in feat_cols:
                row[key] = float(series.tail(w).mean()) if len(series) >= max(1, w // 2) else np.nan

    # Fix 1: Blend series-specific form into short-window rolling features.
    # Only L5 and L10 are blended — L20 stays season-long for stability.
    if series_form:
        w = series_form["weight"]
        for api_col, feat_prefix in col_map.items():
            if api_col not in series_form:
                continue
            s_avg = series_form[api_col]
            for window in [5, 10]:
                key = f"{feat_prefix}_L{window}"
                if key in row and not pd.isna(row.get(key, float("nan"))):
                    row[key] = w * s_avg + (1 - w) * row[key]

    # Derive USG_L10 from components (ball-handling burden feature)
    if "USG_L10" in feat_cols:
        fga = row.get("FGA_L10") or 0.0
        fta = row.get("FTA_L10") or 0.0
        tov = row.get("TOV_L10") or 0.0
        row["USG_L10"] = fga + 0.44 * fta + tov

    return row


# ── Model helpers ──────────────────────────────────────────────────────────────

def load_bias_corrections() -> dict[str, float]:
    """
    Per-stat additive bias from scored predictions: mean(actual - pred_stat).
    Applied before probability calculation to remove systematic over/under-prediction.
    """
    scored_path = DATA_DIR / "props_scored.csv"
    if not scored_path.exists():
        return {}
    df = pd.read_csv(scored_path)
    df["actual_stat"] = pd.to_numeric(df["actual_stat"], errors="coerce")
    df["pred_stat"]   = pd.to_numeric(df["pred_stat"],   errors="coerce")
    df = df.dropna(subset=["actual_stat", "pred_stat"])
    corrections = {}
    for stat, grp in df.groupby("stat"):
        if len(grp) >= 20:
            corrections[stat] = float((grp["actual_stat"] - grp["pred_stat"]).mean())
    if corrections:
        print(f"  Bias corrections loaded: { {k: round(v,2) for k,v in corrections.items()} }")
    return corrections


def load_model(stat: str):
    path = MODEL_DIR / f"props_{stat}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def get_std(mdl: dict, player_name: str, is_playoff: bool = False) -> float:
    """
    Blended std: 40% per-player, 60% global residual.
    Per-player stds are computed from as few as 8 holdout games and are
    unreliable when small — blending prevents extreme overconfidence.
    Playoff multiplier set to 3.5×: tighter defense, scheme adjustments,
    role changes between series games, and small sample sizes in a 7-game
    series all create substantially higher real-world variance than the
    residual std from regular-season training captures.
    """
    player_std  = mdl["player_std"].get(player_name, mdl["residual_std"])
    global_std  = mdl["residual_std"]
    std = 0.4 * player_std + 0.6 * global_std
    if is_playoff:
        std *= 3.5
    return std


def kelly_fraction(edge: float, implied: float) -> float:
    if implied <= 0 or implied >= 1 or edge <= 0:
        return 0.0
    return round(min((edge / (1.0 - implied)) * 0.5, 0.25), 4)


# ── Platt scaling calibration ──────────────────────────────────────────────────

def _fit_platt_calibration() -> dict[str, tuple[float, float]]:
    """
    Fit per-stat logistic calibration on scored props data.
    For YES bets: calibrate P(OVER) vs actual OVER hit rate.
    For NO bets:  calibrate (1-P(OVER)) vs actual UNDER hit rate.
    Returns {stat: (a, b)} for calibrated = sigmoid(a + b * logit(raw_prob)).
    Falls back to identity (0, 1) if not enough data.
    """
    from scipy.special import logit, expit
    from scipy.optimize import minimize

    scored_path = DATA_DIR / "props_scored.csv"
    if not scored_path.exists():
        return {}

    df = pd.read_csv(scored_path)
    df = df[df["hit"].notna() & df["model_prob"].notna()].copy()
    df["model_prob"] = pd.to_numeric(df["model_prob"], errors="coerce")
    df["hit"]        = pd.to_numeric(df["hit"],        errors="coerce")
    df = df.dropna(subset=["model_prob", "hit"])

    # For YES bets, raw = model_prob; for NO bets, raw = 1 - model_prob
    yes = df[df["direction"] == "YES"].copy()
    yes["raw"] = yes["model_prob"]
    yes["act"] = yes["hit"]
    no  = df[df["direction"] == "NO"].copy()
    no["raw"] = 1.0 - no["model_prob"]
    no["act"] = 1.0 - no["hit"]   # under hit = stat < threshold
    combined = pd.concat([yes, no], ignore_index=True)

    params: dict[str, tuple[float, float]] = {}
    for stat, grp in combined.groupby("stat"):
        if len(grp) < 20:
            params[stat] = (0.0, 1.0)
            continue
        raw = np.clip(grp["raw"].values, 1e-6, 1 - 1e-6)
        act = grp["act"].values

        def neg_ll(ab):
            a, b = ab
            p = expit(a + b * logit(raw))
            p = np.clip(p, 1e-9, 1 - 1e-9)
            return -np.mean(act * np.log(p) + (1 - act) * np.log(1 - p))

        res = minimize(neg_ll, x0=[0.0, 1.0], method="Nelder-Mead",
                       options={"xatol": 1e-5, "fatol": 1e-5})
        params[stat] = tuple(res.x[:2])

    return params


_PLATT_PARAMS: dict[str, tuple[float, float]] = {}

def load_calibration():
    global _PLATT_PARAMS
    _PLATT_PARAMS = _fit_platt_calibration()
    if _PLATT_PARAMS:
        print(f"  Loaded Platt calibration for {list(_PLATT_PARAMS.keys())}")


def calibrate_prob(raw_prob: float, stat: str) -> float:
    """Apply Platt scaling to convert raw model probability to calibrated probability."""
    from scipy.special import logit, expit
    if stat not in _PLATT_PARAMS:
        return raw_prob
    a, b = _PLATT_PARAMS[stat]
    raw_clipped = np.clip(raw_prob, 1e-6, 1 - 1e-6)
    return float(expit(a + b * logit(raw_clipped)))


# ── Probability helpers ────────────────────────────────────────────────────────

def prob_exceed(threshold: float, pred: float, std: float) -> float:
    return float(1 - norm.cdf(threshold, pred, std))


def prob_double_double(pred_pts: float, pred_reb: float, pred_ast: float,
                       std_pts: float, std_reb: float, std_ast: float) -> float:
    """
    P(at least 2 of [PTS≥10, REB≥10, AST≥10]) assuming independence.
    Uses inclusion-exclusion: P(A∩B) + P(A∩C) + P(B∩C) - 2·P(A∩B∩C)
    """
    p  = prob_exceed(10, pred_pts, std_pts)
    r  = prob_exceed(10, pred_reb, std_reb)
    a  = prob_exceed(10, pred_ast, std_ast)
    return p * r + p * a + r * a - 2 * p * r * a


def prob_triple_double(pred_pts: float, pred_reb: float, pred_ast: float,
                       std_pts: float, std_reb: float, std_ast: float) -> float:
    """P(PTS≥10 AND REB≥10 AND AST≥10) assuming independence."""
    p = prob_exceed(10, pred_pts, std_pts)
    r = prob_exceed(10, pred_reb, std_reb)
    a = prob_exceed(10, pred_ast, std_ast)
    return p * r * a


# ── Main ───────────────────────────────────────────────────────────────────────

def run(game_id: str | None, min_edge: float, use_cache: bool = False):
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ── Team context
    team_logs = fetch_current_season_team_logs()
    team_ctx  = build_team_context(team_logs)

    # ── Sportsbook props (Odds API) + Kalshi combos
    print("Fetching sportsbook prop lines (Odds API) ...")
    props  = fetch_odds_api_props(game_id, use_cache=use_cache)
    print("Fetching Kalshi combo markets (DD/TD) ...")
    combos = fetch_kalshi_combos(game_id)

    if props.empty:
        print("No prop lines found. Check ODDS_API_KEY in .env or run with --use-cache.")
        return

    # Keep only the primary line per player+stat (closest to 50/50 = main line, not alt lines)
    props = (
        props
        .assign(_dist=lambda d: (d["yes_ask"] - 0.5).abs())
        .sort_values("_dist")
        .drop_duplicates(subset=["player", "stat"], keep="first")
        .drop(columns=["_dist"])
        .reset_index(drop=True)
    )
    print(f"  {len(props)} primary lines after deduplication")

    print(f"  {len(combos)} Kalshi combo markets\n")

    # ── Lineup context (Kalshi-only absent player detection)
    lineup_ctx = build_lineup_context(props, game_id)
    for team, ctx in lineup_ctx.items():
        if ctx["absent_players"]:
            print(f"  Lineup [{team}]: absent={ctx['absent_players']}  "
                  f"load redistributed across {ctx['active_count']} active")

    # ── Game date from ID
    game_date = run_date
    if game_id:
        m = re.match(r"(\d{2})([A-Z]{3})(\d{2})", game_id)
        if m:
            mon_map = {"JAN":"01","FEB":"02","MAR":"03","APR":"04","MAY":"05",
                       "JUN":"06","JUL":"07","AUG":"08","SEP":"09","OCT":"10",
                       "NOV":"11","DEC":"12"}
            game_date = f"20{m.group(1)}-{mon_map.get(m.group(2),'01')}-{m.group(3)}"

    # ── Load models + bias corrections
    models  = {stat: load_model(stat) for stat in PROP_STATS}
    missing = [s for s, mdl in models.items() if mdl is None]
    if missing:
        print(f"Warning: no model for {missing} — run train_props_model.py first")
    bias_corrections = load_bias_corrections()

    # ── Player log cache
    log_cache: dict[str, pd.DataFrame] = {}
    pid_cache: dict[str, int | None]   = {}

    def get_logs(player_name: str) -> pd.DataFrame:
        if player_name not in pid_cache:
            pid_cache[player_name] = get_player_id(player_name)
        pid = pid_cache[player_name]
        if pid is None:
            return pd.DataFrame()
        if player_name not in log_cache:
            print(f"  Fetching logs: {player_name}")
            log_cache[player_name] = fetch_recent_logs(pid)
        return log_cache[player_name]

    # ── Per-player predictions cache (stat → {player: (pred, prob_10)})
    # Used later by combo calculations
    player_preds: dict[str, dict[str, tuple[float, float, float]]] = {
        s: {} for s in ["PTS", "REB", "AST"]
    }

    results = []

    # ─── Individual props ───────────────────────────────────────────────────
    for player_name in props["player"].unique():
        hist = get_logs(player_name)
        player_rows  = props[props["player"] == player_name]
        sample       = player_rows.iloc[0]
        p_team       = sample["player_team"]
        p_game_id    = sample["game_id"]
        away_abb, home_abb = parse_game_teams(p_game_id)
        is_home      = 1 if p_team == home_abb else 0
        opp_team     = home_abb if p_team == away_abb else away_abb
        p_ctx        = team_ctx.get(p_team, {})
        opp_ctx      = team_ctx.get(opp_team, {})
        rest_days    = p_ctx.get("rest_days", 2.0)
        opp_def      = opp_ctx.get("opp_def_l10", 110.0)
        game_in_s    = p_ctx.get("game_in_series", 1)
        series_w     = p_ctx.get("series_wins", 0)
        is_playoff   = 1

        # Fix 1: series-specific form for this opponent
        series_form = get_series_form(hist, opp_team)
        if series_form:
            print(f"  Series form [{player_name} vs {opp_team}]: "
                  f"{series_form['n_games']}g  "
                  f"PTS={series_form.get('PTS',float('nan')):.1f}  "
                  f"REB={series_form.get('REB',float('nan')):.1f}  "
                  f"AST={series_form.get('AST',float('nan')):.1f}")

        for _, mkt in player_rows.iterrows():
            stat      = mkt["stat"]
            threshold = mkt["threshold"]
            yes_ask   = mkt["yes_ask"]
            mdl       = models.get(stat)
            if mdl is None:
                continue

            feat_cols = mdl["features"]
            feat_vec  = build_live_features(
                is_home, is_playoff, rest_days, opp_def, game_in_s,
                hist, feat_cols, series_form, series_wins=series_w,
            )
            X = pd.DataFrame([feat_vec]).reindex(columns=feat_cols)
            if X.isnull().any().any():
                continue

            pred_raw   = float(mdl["model"].predict(X.values)[0])
            # Apply bias correction only to the display value, not the probability.
            # Applying it to the probability inflates edges artificially (a -1.5 AST
            # correction shifts the predicted mean far from the line, which the normal
            # CDF converts into massive "edges" that don't hold up in practice).
            pred_display = pred_raw + bias_corrections.get(stat, 0.0)
            # Fix 3: opponent stat-specific defense (AST/REB only)
            pred_raw   = apply_opp_stat_correction(pred_raw, stat, opp_ctx)
            pred_stat  = apply_lineup_adjustment(pred_raw, stat, p_team, lineup_ctx)
            pred_display = apply_lineup_adjustment(pred_display, stat, p_team, lineup_ctx)
            # Fix 2: widen std in playoffs
            std        = get_std(mdl, player_name, is_playoff=bool(is_playoff))
            model_prob = prob_exceed(threshold, pred_stat, std)

            # Cache PTS/REB/AST predictions for combo use (at threshold=10)
            if stat in player_preds and threshold == 10.0:
                player_preds[stat][player_name] = (pred_stat, std, model_prob)

            edge_yes  = model_prob - yes_ask
            edge_no   = yes_ask - model_prob
            best_edge = max(edge_yes, edge_no)
            direction = "YES" if edge_yes >= edge_no else "NO"
            kelly     = kelly_fraction(best_edge,
                                       yes_ask if direction == "YES" else 1.0 - yes_ask)

            results.append({
                "game_date":      game_date,
                "game_id":        p_game_id,
                "player":         player_name,
                "team":           p_team,
                "opp":            opp_team,
                "is_home":        is_home,
                "market_type":    "PROP",
                "stat":           stat,
                "threshold":      threshold,
                "yes_ask":        yes_ask,
                "model_prob":     round(model_prob, 3),
                "pred_stat":      round(pred_display, 1),
                "lineup_adj":     round(pred_display - pred_raw, 2),
                "edge":           round(best_edge, 3),
                "direction":      direction,
                "kelly":          kelly,
                "rest_days":      rest_days,
                "opp_def_l10":    round(opp_def, 1),
                "game_in_series": game_in_s,
                "ticker":         mkt["ticker"],
                "run_date":       run_date,
            })

    # ─── Combo props (DD / TD) ──────────────────────────────────────────────
    if not combos.empty:
        # We need PTS/REB/AST predictions for each player — fetch if not cached
        all_combo_players = set(combos["player"].unique())
        for player_name in all_combo_players:
            hist = get_logs(player_name)
            sample   = combos[combos["player"] == player_name].iloc[0]
            p_team   = sample["player_team"]
            p_gid    = sample["game_id"]
            away_abb, home_abb = parse_game_teams(p_gid)
            is_home  = 1 if p_team == home_abb else 0
            opp_team = home_abb if p_team == away_abb else away_abb
            p_ctx    = team_ctx.get(p_team, {})
            opp_ctx  = team_ctx.get(opp_team, {})
            rest     = p_ctx.get("rest_days", 2.0)
            opp_def  = opp_ctx.get("opp_def_l10", 110.0)
            gis      = p_ctx.get("game_in_series", 1)

            series_form_combo = get_series_form(hist, opp_team)
            for stat in ["PTS", "REB", "AST"]:
                if player_name in player_preds[stat]:
                    continue  # already computed above
                mdl = models.get(stat)
                if mdl is None:
                    continue
                feat_vec  = build_live_features(
                    is_home, 1, rest, opp_def, gis, hist, mdl["features"], series_form_combo,
                )
                X = pd.DataFrame([feat_vec])[mdl["features"]]
                if X.isnull().any().any():
                    continue
                pred_raw  = float(mdl["model"].predict(X.values)[0])
                pred_raw  = apply_opp_stat_correction(pred_raw, stat, opp_ctx)
                pred_stat = apply_lineup_adjustment(pred_raw, stat, p_team, lineup_ctx)
                std       = get_std(mdl, player_name, is_playoff=True)
                p10       = prob_exceed(10, pred_stat, std)
                player_preds[stat][player_name] = (pred_stat, std, p10)

        for _, mkt in combos.iterrows():
            player_name  = mkt["player"]
            combo_type   = mkt["combo_type"]
            yes_ask      = mkt["yes_ask"]
            p_team       = mkt["player_team"]
            p_gid        = mkt["game_id"]
            away_abb, home_abb = parse_game_teams(p_gid)
            opp_team     = home_abb if p_team == away_abb else away_abb
            p_ctx        = team_ctx.get(p_team, {})
            opp_ctx      = team_ctx.get(opp_team, {})

            pts_data = player_preds["PTS"].get(player_name)
            reb_data = player_preds["REB"].get(player_name)
            ast_data = player_preds["AST"].get(player_name)
            if not all([pts_data, reb_data, ast_data]):
                continue

            pred_pts, std_pts, _ = pts_data
            pred_reb, std_reb, _ = reb_data
            pred_ast, std_ast, _ = ast_data

            if combo_type == "DD":
                model_prob = prob_double_double(pred_pts, pred_reb, pred_ast,
                                               std_pts, std_reb, std_ast)
                stat_label = "DD"
                threshold  = 0.0
            else:  # TD
                model_prob = prob_triple_double(pred_pts, pred_reb, pred_ast,
                                               std_pts, std_reb, std_ast)
                stat_label = "TD"
                threshold  = 0.0

            edge_yes  = model_prob - yes_ask
            edge_no   = yes_ask - model_prob
            best_edge = max(edge_yes, edge_no)
            direction = "YES" if edge_yes >= edge_no else "NO"
            kelly     = kelly_fraction(best_edge,
                                       yes_ask if direction == "YES" else 1.0 - yes_ask)

            # Readable predicted stat summary
            pred_summary = f"{pred_pts:.0f}p/{pred_reb:.0f}r/{pred_ast:.0f}a"

            results.append({
                "game_date":      game_date,
                "game_id":        p_gid,
                "player":         player_name,
                "team":           p_team,
                "opp":            opp_team,
                "is_home":        1 if p_team == home_abb else 0,
                "market_type":    "COMBO",
                "stat":           stat_label,
                "threshold":      threshold,
                "yes_ask":        yes_ask,
                "model_prob":     round(model_prob, 3),
                "pred_stat":      pred_summary,
                "lineup_adj":     0.0,
                "edge":           round(best_edge, 3),
                "direction":      direction,
                "kelly":          kelly,
                "rest_days":      p_ctx.get("rest_days", 2.0),
                "opp_def_l10":    round(opp_ctx.get("opp_def_l10", 110.0), 1),
                "game_in_series": p_ctx.get("game_in_series", 1),
                "ticker":         mkt["ticker"],
                "run_date":       run_date,
            })

    if not results:
        print("No predictions generated.")
        return

    all_results = (
        pd.DataFrame(results)
        .sort_values("edge", ascending=False)
        .reset_index(drop=True)
    )
    show = all_results[all_results["edge"] >= min_edge].copy()

    # ── Print output ─────────────────────────────────────────────────────────
    away_abb, home_abb = parse_game_teams(game_id or "")
    print("\n" + "═" * 92)
    print(f"{'PLAYER':<22} {'TYPE':<5} {'STAT':<5} {'LINE':>5} {'YES_ASK':>8} "
          f"{'MODEL':>7} {'PRED':>8} {'EDGE':>7} {'DIR':<4} {'KELLY':>7}")
    print("─" * 92)

    for mtype in ["PROP", "COMBO"]:
        section = show[show["market_type"] == mtype]
        if section.empty:
            continue
        print(f"  ── {'Individual Props' if mtype=='PROP' else 'Combo Props (DD/TD)'} ──")
        for _, r in section.iterrows():
            if r["stat"] in ("DD", "TD"):
                line_str = "—"
            else:
                t = float(r["threshold"])
                line_str = f"{t + 0.5:.1f}" if t == int(t) else f"{t}"
            print(
                f"  {r['player']:<20} {mtype:<5} {r['stat']:<5} {line_str:>5} "
                f"{r['yes_ask']:>8.3f} {r['model_prob']:>7.3f} {str(r['pred_stat']):>8} "
                f"{r['edge']:>7.3f} {r['direction']:<4} {r['kelly']:>7.4f}"
            )

    print("═" * 92)

    # Show absent players
    for team, ctx in lineup_ctx.items():
        if ctx["absent_players"]:
            print(f"\n  ⚠ [{team}] absent: {', '.join(ctx['absent_players'])} "
                  f"— load spread across {ctx['active_count']} active")

    print(f"\n  Context: {away_abb} @ {home_abb}  |  "
          f"rest={team_ctx.get(away_abb,{}).get('rest_days','?')}d/{team_ctx.get(home_abb,{}).get('rest_days','?')}d  "
          f"|  series game #{team_ctx.get(home_abb,{}).get('game_in_series','?')}")

    # ── Save ─────────────────────────────────────────────────────────────────
    show.to_csv(PROPS_TODAY, index=False)
    print(f"\nSaved edges → {PROPS_TODAY}")

    write_header = not PROPS_LOG.exists()
    all_results.to_csv(PROPS_LOG, mode="a", header=write_header, index=False)
    print(f"Appended {len(all_results)} rows → {PROPS_LOG}")


def list_tonight_games():
    """Fetch and display tonight's NBA games with their game IDs."""
    if not ODDS_API_KEY:
        print("ODDS_API_KEY not set in .env")
        return
    r = requests.get(
        f"{ODDS_API_BASE}/sports/basketball_nba/events",
        params={"apiKey": ODDS_API_KEY},
        timeout=30,
    )
    events = r.json()
    if not isinstance(events, list) or not events:
        print("No games found tonight.")
        return

    mon_map = {"01":"JAN","02":"FEB","03":"MAR","04":"APR","05":"MAY","06":"JUN",
               "07":"JUL","08":"AUG","09":"SEP","10":"OCT","11":"NOV","12":"DEC"}
    print("\nTonight's NBA games:")
    print("─" * 55)
    for e in events:
        home = e.get("home_team","")
        away = e.get("away_team","")
        tip  = e.get("commence_time","")
        home_abb = TEAM_FULL_TO_ABB.get(home, home[:3].upper())
        away_abb = TEAM_FULL_TO_ABB.get(away, away[:3].upper())
        # Parse ISO timestamp → YYMONDD
        if tip:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(tip.replace("Z","+00:00"))
            dt_local = dt.astimezone()
            tag = f"{str(dt_local.year)[2:]}{mon_map[f'{dt_local.month:02d}']}{dt_local.day:02d}"
            game_id = f"{tag}{away_abb}{home_abb}"
            tip_str = dt_local.strftime("%-I:%M %p")
        else:
            game_id = f"{away_abb}{home_abb}"
            tip_str = "TBD"
        print(f"  {away:<28} @ {home:<28}")
        print(f"  Tip-off: {tip_str:<10}  --game {game_id}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default=None,
                        help="Game ID, e.g. 26MAY04PHINYK  (use --list to see options)")
    parser.add_argument("--list", action="store_true",
                        help="Show tonight's games and their --game IDs then exit")
    parser.add_argument("--min-edge", type=float, default=0.05,
                        help="Minimum edge to display (default 5%%)")
    parser.add_argument("--use-cache", action="store_true",
                        help="Load props from local cache — safe for demos, no API call")
    parser.add_argument("--cache-only", action="store_true",
                        help="Fetch and cache tonight's props then exit — run this before demo")
    args = parser.parse_args()

    if args.list:
        list_tonight_games()
    elif args.cache_only:
        print("Pre-fetching and caching props for demo ...")
        fetch_odds_api_props(args.game, use_cache=False)
        print(f"Cached → {ODDS_CACHE}")
    else:
        run(args.game, args.min_edge, use_cache=args.use_cache)
