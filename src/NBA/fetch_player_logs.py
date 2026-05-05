"""
fetch_player_logs.py
────────────────────────────────────────────────────────────────────────────
Pulls 5 seasons of per-player per-game logs from nba_api (regular season +
playoffs) and saves them to src/NBA/data/player_logs_raw.csv.

Also pulls team defensive context (pts allowed per game, rolling) so the
prop model can see opponent quality.

Usage:
    python src/NBA/fetch_player_logs.py
    python src/NBA/fetch_player_logs.py --seasons 2023-24 2024-25

Rate limit note: nba_api has ~1 req/sec limits; we sleep between calls.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

OUT_DIR = Path("src/NBA/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
SEASON_TYPES = ["Regular Season", "Playoffs"]

PLAYER_COLS = [
    "SEASON_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
    "GAME_ID", "GAME_DATE", "MATCHUP", "WL",
    "MIN", "FGM", "FGA", "FG_PCT",
    "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF",
    "PTS", "PLUS_MINUS",
]

TEAM_COLS = [
    "SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION",
    "GAME_ID", "GAME_DATE", "MATCHUP", "WL",
    "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "PLUS_MINUS",
]


def fetch_season(season: str, season_type: str, retries: int = 3) -> pd.DataFrame:
    label = f"{season} {season_type}"
    for attempt in range(1, retries + 1):
        try:
            print(f"  Fetching player logs: {label} (attempt {attempt})")
            df = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                player_or_team_abbreviation="P",
                timeout=60,
            ).get_data_frames()[0]
            keep = [c for c in PLAYER_COLS if c in df.columns]
            df = df[keep].copy()
            df["SEASON"] = season
            df["SEASON_TYPE"] = season_type
            print(f"    → {len(df):,} player-game rows")
            return df
        except Exception as e:
            print(f"    ERROR: {e}")
            if attempt < retries:
                time.sleep(5 * attempt)
    return pd.DataFrame()


def fetch_team_logs(season: str, season_type: str) -> pd.DataFrame:
    label = f"{season} {season_type}"
    for attempt in range(1, 4):
        try:
            print(f"  Fetching team logs:   {label} (attempt {attempt})")
            df = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                player_or_team_abbreviation="T",
                timeout=60,
            ).get_data_frames()[0]
            keep = [c for c in TEAM_COLS if c in df.columns]
            df = df[keep].copy()
            df["SEASON"] = season
            df["SEASON_TYPE"] = season_type
            return df
        except Exception as e:
            print(f"    ERROR: {e}")
            if attempt < 3:
                time.sleep(5 * attempt)
    return pd.DataFrame()


def build_opp_defense(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each team-game, compute what the *opponent* allowed that night so we
    can attach it as a defensive-strength feature to each player row.

    Output columns added (per GAME_ID + defending TEAM_ABBREVIATION):
        OPP_PTS_ALLOWED   – how many points the opponent gave up that game
    """
    df = team_df[["GAME_ID", "TEAM_ABBREVIATION", "PTS", "MATCHUP"]].copy()
    df.columns = ["GAME_ID", "OPP_TEAM", "OPP_PTS_SCORED", "OPP_MATCHUP"]
    # pts scored by team X = pts allowed by opponent
    # we want: for player on team Y in GAME_ID, what did Y's opponent allow?
    # join on GAME_ID where team != opp_team
    base = team_df[["GAME_ID", "TEAM_ABBREVIATION", "PTS"]].copy()
    # merge to get opponent's pts scored (= this team's pts allowed)
    merged = base.merge(df[["GAME_ID", "OPP_TEAM", "OPP_PTS_SCORED"]], on="GAME_ID")
    merged = merged[merged["TEAM_ABBREVIATION"] != merged["OPP_TEAM"]]
    # OPP_PTS_SCORED is what the opponent scored = pts this team allowed
    opp_def = merged[["GAME_ID", "TEAM_ABBREVIATION", "OPP_PTS_SCORED"]].copy()
    opp_def.rename(columns={"OPP_PTS_SCORED": "OPP_PTS_ALLOWED"}, inplace=True)
    return opp_def


def main(seasons=None):
    seasons = seasons or SEASONS
    player_frames = []
    team_frames = []

    for season in seasons:
        for stype in SEASON_TYPES:
            pf = fetch_season(season, stype)
            tf = fetch_team_logs(season, stype)
            if not pf.empty:
                player_frames.append(pf)
            if not tf.empty:
                team_frames.append(tf)
            time.sleep(1.5)

    if not player_frames:
        print("No data fetched — exiting.")
        return

    players = pd.concat(player_frames, ignore_index=True)
    teams = pd.concat(team_frames, ignore_index=True)

    # Parse dates
    players["GAME_DATE"] = pd.to_datetime(players["GAME_DATE"])
    teams["GAME_DATE"] = pd.to_datetime(teams["GAME_DATE"])

    # Derive IS_HOME from MATCHUP (e.g. "CLE vs. TOR" = home, "CLE @ TOR" = away)
    players["IS_HOME"] = players["MATCHUP"].str.contains(" vs\\.").astype(int)
    players["IS_PLAYOFF"] = (players["SEASON_TYPE"] == "Playoffs").astype(int)

    # Attach opponent defensive context
    opp_def = build_opp_defense(teams)
    players = players.merge(opp_def, on=["GAME_ID", "TEAM_ABBREVIATION"], how="left")

    # Sort for rolling calculations later
    players.sort_values(["PLAYER_ID", "GAME_DATE"], inplace=True)
    players.reset_index(drop=True, inplace=True)

    out_path = OUT_DIR / "player_logs_raw.csv"
    players.to_csv(out_path, index=False)
    print(f"\nSaved {len(players):,} rows → {out_path}")

    # Also save team logs for downstream use
    team_path = OUT_DIR / "team_logs_raw.csv"
    teams.to_csv(team_path, index=False)
    print(f"Saved {len(teams):,} team rows → {team_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", default=None,
                        help="e.g. --seasons 2023-24 2024-25")
    args = parser.parse_args()
    main(args.seasons)
