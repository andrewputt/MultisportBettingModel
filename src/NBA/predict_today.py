"""
predict_today.py
Fetches today's NBA markets from Kalshi, builds fresh team features from the
current season, runs the trained XGBoost model, applies an injury adjustment
for missing star players, and compares model win probability against Kalshi
implied probability to surface edges.
"""

import os
import pickle
import re
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from nba_api.stats.endpoints import leaguegamelog, leaguedashplayerstats, playergamelog, boxscoretraditionalv3

# ── Feature columns expected by the model ─────────────────────────────────────
FEATURE_COLS = [
    "IS_HOME",
    "IS_PLAYOFF",
    "REST_DAYS",
    "WIN_PCT_L10",
    "OFF_RATING_L10",
    "DEF_RATING_L10",
    "PACE_PROXY_L10",
    "PM_TREND_L10",
    "OPP_WIN_PCT_L10",
    "OPP_OFF_RATING_L10",
    "OPP_DEF_RATING_L10",
    "OPP_PACE_PROXY_L10",
    "OPP_PM_TREND_L10",
]

# ── Abbreviation → display name ────────────────────────────────────────────────
ABB_TO_NAME = {
    "ATL": "Atlanta",
    "BKN": "Brooklyn",
    "BOS": "Boston",
    "CHA": "Charlotte",
    "CHI": "Chicago",
    "CLE": "Cleveland",
    "DAL": "Dallas",
    "DEN": "Denver",
    "DET": "Detroit",
    "GSW": "Golden State",
    "HOU": "Houston",
    "IND": "Indiana",
    "LAC": "LA Clippers",
    "LAL": "LA Lakers",
    "MEM": "Memphis",
    "MIA": "Miami",
    "MIL": "Milwaukee",
    "MIN": "Minnesota",
    "NOP": "New Orleans",
    "NYK": "New York",
    "OKC": "Oklahoma City",
    "ORL": "Orlando",
    "PHI": "Philadelphia",
    "PHX": "Phoenix",
    "POR": "Portland",
    "SAC": "Sacramento",
    "SAS": "San Antonio",
    "TOR": "Toronto",
    "UTA": "Utah",
    "WAS": "Washington",
}

# Base injury penalty for a player averaging REFERENCE_STAR_PPG points.
# Scales proportionally — a 30ppg player costs ~10%, a 12ppg player ~4%.
BASE_INJURY_PENALTY = 0.06
REFERENCE_STAR_PPG = 18.0


# ── 1. Fetch current season game logs and compute rolling features ─────────────
def get_current_season_features():
    print("Fetching current season (2025-26) game logs (regular + playoffs)...")
    fields = [
        "GAME_ID",
        "GAME_DATE",
        "TEAM_ABBREVIATION",
        "MATCHUP",
        "WL",
        "PTS",
        "FG_PCT",
        "REB",
        "AST",
        "PLUS_MINUS",
    ]
    frames = []
    for season_type, label in [("Regular Season", 0), ("Playoffs", 1)]:
        logs = leaguegamelog.LeagueGameLog(
            season="2025-26", season_type_all_star=season_type
        )
        part = logs.get_data_frames()[0][fields].copy()
        part["IS_PLAYOFF"] = label
        frames.append(part)
        time.sleep(1)
    df = pd.concat(frames, ignore_index=True)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["WIN"] = df["WL"].map({"W": 1, "L": 0})
    df = df.dropna(subset=["PTS", "REB", "AST", "PLUS_MINUS"])
    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)
    df["REST_DAYS"] = (
        df.groupby("TEAM_ABBREVIATION")["GAME_DATE"].diff().dt.days.fillna(0)
    )
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

    latest = (
        df.dropna(subset=["WIN_PCT_L10"])
        .sort_values("GAME_DATE")
        .groupby("TEAM_ABBREVIATION")
        .last()
        .reset_index()
    )
    print(
        f"  Got stats for {len(latest)} teams (last game: {latest['GAME_DATE'].max().date()})"
    )

    # Build series context for each (team, opponent) pair currently in playoffs.
    # Values represent totals from games already played — used as context for the next game.
    series_context = {}
    playoffs_rest_advantage = {}
    playoff_games = df[df["IS_PLAYOFF"] == 1].copy()
    if not playoff_games.empty:
        for (team, opp), grp in playoff_games.groupby(
            ["TEAM_ABBREVIATION", "OPP_TEAM"]
        ):
            grp = grp.sort_values("GAME_DATE")
            n_played = len(grp)
            n_wins = int(grp["WIN"].sum())
            series_context[(team, opp)] = {
                "SERIES_GAME_NUM": n_played + 1,
                "SERIES_WINS": n_wins,
                "SERIES_LOSSES": n_played - n_wins,
            }
        print(
            f"  Computed series context for {len(series_context)} team-opponent pairs"
        )

    return latest, series_context, playoffs_rest_advantage


# ── 2. Injury detection ────────────────────────────────────────────────────────
def get_injury_report(team_abbs, team_stats):
    """
    For each team, check their most recent playoff game box score to see which
    top-5 players (by season minutes) actually played. If a player is absent
    from the box score or logged 0 minutes, they are flagged as out.

    Falls back to the 6-day threshold for teams whose last game box score
    cannot be fetched.

    Returns: dict of {team_abb: [{"name": str, "ppg": float}]}
    """
    print("Fetching player stats for injury detection...")

    # Get top 5 players per team by playoff minutes, including PPG for penalty scaling
    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season="2025-26",
        season_type_all_star="Playoffs",
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]
    time.sleep(1)

    stars_by_team = {}
    for abb in team_abbs:
        team_players = stats[stats["TEAM_ABBREVIATION"] == abb].copy()
        if team_players.empty:
            stars_by_team[abb] = []
            continue
        top = team_players.nlargest(5, "MIN")[["PLAYER_ID", "PLAYER_NAME", "MIN", "PTS"]]
        stars_by_team[abb] = top.to_dict("records")

    # Get the most recent game ID per team from the season game logs
    recent_game_id = {}
    if team_stats is not None and "GAME_ID" in team_stats.columns:
        for abb in team_abbs:
            row = team_stats[team_stats["TEAM_ABBREVIATION"] == abb]
            if not row.empty and pd.notna(row.iloc[0]["GAME_ID"]):
                recent_game_id[abb] = str(row.iloc[0]["GAME_ID"]).zfill(10)

    missing = {abb: [] for abb in team_abbs}
    today = datetime.now(timezone.utc).date()

    for abb in team_abbs:
        players = stars_by_team.get(abb, [])
        if not players:
            continue

        game_id = recent_game_id.get(abb)
        if game_id:
            # Primary: check box score of team's most recent game
            try:
                box = boxscoretraditionalv3.BoxScoreTraditionalV3(
                    game_id=game_id
                ).get_data_frames()[0]
                time.sleep(0.6)
                # Players who appeared in the box score with minutes played
                played_ids = set(
                    box.loc[box["minutes"].notna() & (box["minutes"] != "0:00"), "personId"]
                )
                for player in players:
                    if player["PLAYER_ID"] not in played_ids:
                        missing[abb].append({"name": player["PLAYER_NAME"], "ppg": player.get("PTS", REFERENCE_STAR_PPG)})
                continue  # box score check succeeded, skip fallback
            except Exception:
                pass  # fall through to fallback below

        # Fallback: check if player's last game log is 6+ days old
        for player in players:
            pid = player["PLAYER_ID"]
            name = player["PLAYER_NAME"]
            try:
                logs = playergamelog.PlayerGameLog(
                    player_id=pid, season="2025-26", season_type_all_star="Playoffs"
                ).get_data_frames()[0]
                time.sleep(0.6)
                ppg = next((p.get("PTS", REFERENCE_STAR_PPG) for p in stars_by_team.get(abb, []) if p["PLAYER_ID"] == pid), REFERENCE_STAR_PPG)
                if logs.empty:
                    missing[abb].append({"name": name, "ppg": ppg})
                    continue
                last_game = pd.to_datetime(logs["GAME_DATE"].iloc[0])
                if (pd.Timestamp(today) - last_game).days >= 6:
                    missing[abb].append({"name": name, "ppg": ppg})
            except Exception:
                pass

    for abb in team_abbs:
        if missing[abb]:
            names = ", ".join(f"{p['name']} ({p['ppg']:.1f}ppg)" for p in missing[abb])
            print(f"  {abb} missing stars: {names}")
        else:
            print(f"  {abb}: all stars active")

    return missing


# ── 3. Fetch Kalshi NBA markets ────────────────────────────────────────────────
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

        implied = {}
        for side in sides:
            ticker = side.get("ticker", "")
            team_abb = ticker.split("-")[-1]
            if team_abb in ABB_TO_NAME:
                implied[team_abb] = float(side.get("yes_ask_dollars", 0))

        game_date = sides[0].get("expected_expiration_time", "")[:10]

        parsed.append(
            {
                "event": event,
                "game_date": game_date,
                "away_abb": away_abb,
                "home_abb": home_abb,
                "away": ABB_TO_NAME[away_abb],
                "home": ABB_TO_NAME[home_abb],
                "away_implied": implied.get(away_abb),
                "home_implied": implied.get(home_abb),
            }
        )

    return parsed


# ── 4. Build feature rows, run model, apply injury adjustment ─────────────────
def get_team_stats(team_stats, abb):
    row = team_stats[team_stats["TEAM_ABBREVIATION"] == abb]
    return row.iloc[0] if not row.empty else None


def build_row(team_abb, opp_abb, is_home, is_playoff, team_stats, series_context=None):
    t = get_team_stats(team_stats, team_abb)
    o = get_team_stats(team_stats, opp_abb)
    if t is None or o is None:
        return None
    sc = (series_context or {}).get((team_abb, opp_abb), {})
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
        "SERIES_GAME_NUM": sc.get("SERIES_GAME_NUM", 0),
        "SERIES_WINS": sc.get("SERIES_WINS", 0),
        "SERIES_LOSSES": sc.get("SERIES_LOSSES", 0),
    }


def compute_injury_penalty(missing_players):
    """Sum scaled penalties for all missing players based on their PPG."""
    total = 0.0
    for p in missing_players:
        ppg = p.get("ppg", REFERENCE_STAR_PPG)
        total += BASE_INJURY_PENALTY * (ppg / REFERENCE_STAR_PPG)
    return min(total, 0.25)


def run_predictions(games, team_stats, model, missing_stars, series_context=None, playoffs_rest_advantage=None):
    results = []
    for game in games:
        away_row = build_row(
            game["away_abb"], game["home_abb"], 0, 1, team_stats, series_context
        )
        home_row = build_row(
            game["home_abb"], game["away_abb"], 1, 1, team_stats, series_context
        )

        if away_row is None or home_row is None:
            print(
                f"  [WARN] Missing stats for {game['away_abb']} or {game['home_abb']} — skipping"
            )
            continue

        df_input = pd.DataFrame([away_row, home_row])[FEATURE_COLS]
        probs = model.predict_proba(df_input)[:, 1]

        away_base = round(float(probs[0]), 4)
        home_base = round(float(probs[1]), 4)

        away_penalty = compute_injury_penalty(missing_stars.get(game["away_abb"], []))
        home_penalty = compute_injury_penalty(missing_stars.get(game["home_abb"], []))

        # Injured team loses probability; opponent gains the same amount (zero-sum transfer)
        away_adj = away_base - away_penalty + home_penalty
        home_adj = home_base - home_penalty + away_penalty

        if playoffs_rest_advantage:
            away_adj += playoffs_rest_advantage.get(game["away_abb"], 0.0)
            home_adj += playoffs_rest_advantage.get(game["home_abb"], 0.0)

        away_adj = round(max(0.01, min(0.99, away_adj)), 4)
        home_adj = round(max(0.01, min(0.99, home_adj)), 4)

        away_implied = game["away_implied"]
        home_implied = game["home_implied"]

        matchup = f"{game['away']} @ {game['home']}"
        for abb, name, is_home, base_prob, adj_prob, implied in [
            (game["away_abb"], game["away"], 0, away_base, away_adj, away_implied),
            (game["home_abb"], game["home"], 1, home_base, home_adj, home_implied),
        ]:
            n_missing = len(missing_stars.get(abb, []))
            edge = round(adj_prob - implied, 4) if implied is not None else None
            # Half Kelly: conservative sizing that accounts for model uncertainty.
            # Capped at 25% of bankroll.
            if edge is not None and edge > 0 and implied is not None and implied < 1:
                kelly = round((edge / (1 - implied)) * 0.5, 4)
                kelly = min(kelly, 0.25)
            else:
                kelly = None
            results.append(
                {
                    "GAME_DATE": game["game_date"],
                    "MATCHUP": matchup,
                    "TEAM": name,
                    "ABB": abb,
                    "IS_HOME": is_home,
                    "MODEL_PROB": base_prob,
                    "INJ_ADJUSTED_PROB": adj_prob,
                    "STARS_OUT": n_missing,
                    "KALSHI_IMPLIED": implied,
                    "EDGE": edge,
                    "KELLY_FRACTION": kelly,
                }
            )

    return pd.DataFrame(results)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open("src/NBA/models/nba_model.pkl", "rb") as f:
        model = pickle.load(f)

    team_stats, series_context, playoffs_rest_advantage = get_current_season_features()
    markets = fetch_kalshi_markets()
    games = parse_markets(markets)
    print(f"Parsed {len(games)} unique games from Kalshi")

    # Get all unique team abbreviations across today's games
    all_teams = list({abb for g in games for abb in [g["away_abb"], g["home_abb"]]})
    missing_stars = get_injury_report(all_teams, team_stats)

    df = run_predictions(
        games, team_stats, model, missing_stars, series_context, playoffs_rest_advantage
    )

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
            print(
                edges[
                    [
                        "GAME_DATE",
                        "MATCHUP",
                        "TEAM",
                        "INJ_ADJUSTED_PROB",
                        "STARS_OUT",
                        "KALSHI_IMPLIED",
                        "EDGE",
                        "KELLY_FRACTION",
                    ]
                ].to_string(index=False)
            )
        else:
            print("\nNo edges >= 5% found.")

        df.to_csv("src/NBA/data/predictions_today.csv", index=False)
        print("\nSaved to src/NBA/data/predictions_today.csv")

        # Append to running predictions log with timestamp
        log_path = "src/NBA/data/predictions_log.csv"
        df["RUN_DATE"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        write_header = not os.path.exists(log_path)
        df.to_csv(log_path, mode="a", header=write_header, index=False)
        print(f"Appended {len(df)} predictions to {log_path}")
