"""
predict_today.py
Fetches today's NBA markets from Kalshi, builds fresh team features from the
current season, runs the trained model, and compares model win probability
against Kalshi implied probability to surface edges.
"""

import pickle
import re
from datetime import datetime, timezone

import pandas as pd
import requests
from nba_api.stats.endpoints import leaguegamelog

# ── Feature columns expected by the model ─────────────────────────────────────
FEATURE_COLS = [
    "IS_HOME", "IS_PLAYOFF", "REST_DAYS", "WIN_PCT_L10",
    "OFF_RATING_L10", "DEF_RATING_L10", "PACE_PROXY_L10", "PM_TREND_L10",
    "OPP_WIN_PCT_L10", "OPP_OFF_RATING_L10", "OPP_DEF_RATING_L10",
    "OPP_PACE_PROXY_L10", "OPP_PM_TREND_L10",
]

# ── Abbreviation → display name ────────────────────────────────────────────────
ABB_TO_NAME = {
    "ATL": "Atlanta", "BKN": "Brooklyn", "BOS": "Boston", "CHA": "Charlotte",
    "CHI": "Chicago", "CLE": "Cleveland", "DAL": "Dallas", "DEN": "Denver",
    "DET": "Detroit", "GSW": "Golden State", "HOU": "Houston", "IND": "Indiana",
    "LAC": "LA Clippers", "LAL": "LA Lakers", "MEM": "Memphis", "MIA": "Miami",
    "MIL": "Milwaukee", "MIN": "Minnesota", "NOP": "New Orleans", "NYK": "New York",
    "OKC": "Oklahoma City", "ORL": "Orlando", "PHI": "Philadelphia", "PHX": "Phoenix",
    "POR": "Portland", "SAC": "Sacramento", "SAS": "San Antonio", "TOR": "Toronto",
    "UTA": "Utah", "WAS": "Washington",
}


# ── 1. Fetch current season game logs and compute rolling features ─────────────
def get_current_season_features():
    print("Fetching current season (2025-26) game logs (regular + playoffs)...")
    fields = ["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "MATCHUP",
              "WL", "PTS", "FG_PCT", "REB", "AST", "PLUS_MINUS"]
    frames = []
    for season_type, label in [("Regular Season", 0), ("Playoffs", 1)]:
        logs = leaguegamelog.LeagueGameLog(season="2025-26", season_type_all_star=season_type)
        part = logs.get_data_frames()[0][fields].copy()
        part["IS_PLAYOFF"] = label
        frames.append(part)
        import time; time.sleep(1)
    df = pd.concat(frames, ignore_index=True)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["WIN"] = df["WL"].map({"W": 1, "L": 0})
    df = df.dropna(subset=["PTS", "REB", "AST", "PLUS_MINUS"])
    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)
    df["REST_DAYS"] = df.groupby("TEAM_ABBREVIATION")["GAME_DATE"].diff().dt.days.fillna(0)
    df["IS_HOME"] = df["MATCHUP"].apply(lambda x: 1 if "vs." in x else 0)

    def roll(col, window=10):
        return df.groupby("TEAM_ABBREVIATION")[col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=3).mean()
        )

    df["WIN_PCT_L10"] = roll("WIN")
    df["OFF_RATING_L10"] = roll("PTS")

    opp = df[["GAME_ID", "TEAM_ABBREVIATION", "PTS"]].copy()
    opp.columns = ["GAME_ID", "OPP_TEAM", "OPP_PTS"]
    df = df.merge(opp, on="GAME_ID", how="left")
    df = df[df["TEAM_ABBREVIATION"] != df["OPP_TEAM"]]
    df["DEF_RATING_L10"] = roll("OPP_PTS")
    df["PACE_PROXY_L10"] = roll("REB") + roll("AST")
    df["PM_TREND_L10"] = roll("PLUS_MINUS")

    # Most recent row per team
    latest = (
        df.dropna(subset=["WIN_PCT_L10"])
        .sort_values("GAME_DATE")
        .groupby("TEAM_ABBREVIATION")
        .last()
        .reset_index()
    )
    print(f"  Got stats for {len(latest)} teams (last game: {latest['GAME_DATE'].max().date()})")
    return latest


# ── 2. Fetch Kalshi NBA markets ────────────────────────────────────────────────
def fetch_kalshi_markets():
    resp = requests.get(
        "https://api.elections.kalshi.com/trade-api/v2/markets",
        params={"limit": 200, "status": "open", "series_ticker": "KXNBAGAME"},
    )
    resp.raise_for_status()
    markets = resp.json().get("markets", [])
    print(f"Fetched {len(markets)} open Kalshi NBA markets")
    return markets


def parse_teams_from_event_ticker(event_ticker):
    """
    Extract (away_abb, home_abb) from event ticker.
    Format: KXNBAGAME-26APR17GSWPHX -> away=GSW, home=PHX (last 6 chars = 3+3)
    """
    # Strip prefix and date: KXNBAGAME-26APR17GSWPHX -> GSWPHX
    match = re.search(r"\d{2}[A-Z]{3}\d{2}([A-Z]{6})$", event_ticker)
    if match:
        code = match.group(1)
        return code[:3], code[3:]
    return None, None


def parse_markets(markets):
    events = {}
    for m in markets:
        et = m.get("event_ticker", "")
        if et not in events:
            events[et] = []
        events[et].append(m)

    parsed = []
    for event, sides in events.items():
        if len(sides) != 2:
            continue

        away_abb, home_abb = parse_teams_from_event_ticker(event)
        if not away_abb or not home_abb:
            print(f"  [WARN] Could not parse teams from ticker: {event}")
            continue
        if away_abb not in ABB_TO_NAME or home_abb not in ABB_TO_NAME:
            print(f"  [WARN] Unknown abbreviation in {event}: {away_abb} or {home_abb}")
            continue

        # Map yes_sub_title → implied probability using the ticker suffix (last 3 chars)
        implied = {}
        for side in sides:
            ticker = side.get("ticker", "")
            team_abb = ticker.split("-")[-1]  # e.g. KXNBAGAME-26APR17GSWPHX-PHX -> PHX
            if team_abb in ABB_TO_NAME:
                implied[team_abb] = float(side.get("yes_ask_dollars", 0))

        game_date = sides[0].get("expected_expiration_time", "")[:10]

        parsed.append({
            "event": event,
            "game_date": game_date,
            "away_abb": away_abb,
            "home_abb": home_abb,
            "away": ABB_TO_NAME[away_abb],
            "home": ABB_TO_NAME[home_abb],
            "away_implied": implied.get(away_abb),
            "home_implied": implied.get(home_abb),
        })

    return parsed


# ── 3. Build feature rows and run model ───────────────────────────────────────
def get_team_stats(team_stats, abb):
    row = team_stats[team_stats["TEAM_ABBREVIATION"] == abb]
    return row.iloc[0] if not row.empty else None


def build_row(team_abb, opp_abb, is_home, is_playoff, team_stats):
    t = get_team_stats(team_stats, team_abb)
    o = get_team_stats(team_stats, opp_abb)
    if t is None or o is None:
        return None
    return {
        "IS_HOME": is_home,
        "IS_PLAYOFF": is_playoff,
        "REST_DAYS": t["REST_DAYS"],
        "WIN_PCT_L10": t["WIN_PCT_L10"],
        "OFF_RATING_L10": t["OFF_RATING_L10"],
        "DEF_RATING_L10": t["DEF_RATING_L10"],
        "PACE_PROXY_L10": t["PACE_PROXY_L10"],
        "PM_TREND_L10": t["PM_TREND_L10"],
        "OPP_WIN_PCT_L10": o["WIN_PCT_L10"],
        "OPP_OFF_RATING_L10": o["OFF_RATING_L10"],
        "OPP_DEF_RATING_L10": o["DEF_RATING_L10"],
        "OPP_PACE_PROXY_L10": o["PACE_PROXY_L10"],
        "OPP_PM_TREND_L10": o["PM_TREND_L10"],
    }


def run_predictions(games, team_stats, model):
    results = []
    for game in games:
        away_row = build_row(game["away_abb"], game["home_abb"], 0, 1, team_stats)
        home_row = build_row(game["home_abb"], game["away_abb"], 1, 1, team_stats)

        if away_row is None or home_row is None:
            print(f"  [WARN] Missing stats for {game['away_abb']} or {game['home_abb']} — skipping")
            continue

        df_input = pd.DataFrame([away_row, home_row])[FEATURE_COLS]
        probs = model.predict_proba(df_input)[:, 1]

        away_model_prob = round(float(probs[0]), 4)
        home_model_prob = round(float(probs[1]), 4)

        away_implied = game["away_implied"]
        home_implied = game["home_implied"]

        away_edge = round(away_model_prob - away_implied, 4) if away_implied is not None else None
        home_edge = round(home_model_prob - home_implied, 4) if home_implied is not None else None

        matchup = f"{game['away']} @ {game['home']}"
        for abb, name, is_home, model_prob, implied, edge in [
            (game["away_abb"], game["away"], 0, away_model_prob, away_implied, away_edge),
            (game["home_abb"], game["home"], 1, home_model_prob, home_implied, home_edge),
        ]:
            results.append({
                "GAME_DATE": game["game_date"],
                "MATCHUP": matchup,
                "TEAM": name,
                "ABB": abb,
                "IS_HOME": is_home,
                "MODEL_PROB": model_prob,
                "KALSHI_IMPLIED": implied,
                "EDGE": edge,
            })

    return pd.DataFrame(results)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open("src/NBA/models/nba_model.pkl", "rb") as f:
        model = pickle.load(f)

    team_stats = get_current_season_features()
    markets = fetch_kalshi_markets()
    games = parse_markets(markets)
    print(f"Parsed {len(games)} unique games from Kalshi")

    df = run_predictions(games, team_stats, model)

    if df.empty:
        print("No predictions generated.")
    else:
        df = df.sort_values("EDGE", ascending=False).reset_index(drop=True)

        print("\n" + "=" * 80)
        print("NBA PREDICTIONS vs KALSHI IMPLIED PROBABILITY")
        print(f"Run at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        print("=" * 80)
        print(df.to_string(index=False))

        edges = df[df["EDGE"].notna() & (df["EDGE"] >= 0.05)]
        if not edges.empty:
            print("\n--- EDGES >= 5% (potential value bets) ---")
            print(edges[["GAME_DATE", "MATCHUP", "TEAM", "MODEL_PROB", "KALSHI_IMPLIED", "EDGE"]].to_string(index=False))
        else:
            print("\nNo edges >= 5% found.")

        df.to_csv("src/NBA/data/predictions_today.csv", index=False)
        print("\nSaved to src/NBA/data/predictions_today.csv")
