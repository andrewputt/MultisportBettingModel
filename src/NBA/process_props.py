"""
process_props.py
────────────────────────────────────────────────────────────────────────────
Reads player_logs_raw.csv and engineers features for the prop model.

For each player-game row we build:
  - Rolling averages for PTS / REB / AST / FG3M / STL / BLK / MIN
    over the last 5, 10, and 20 games (shift-1 so no leakage)
  - OPP_DEF_RATING_L10  – opponent's pts-allowed rolling avg (last 10 games)
  - REST_DAYS           – days since last game
  - IS_HOME, IS_PLAYOFF (already present)
  - GAME_IN_SERIES      – 1–7 for playoffs, 0 for regular season

Output: src/NBA/data/player_features.csv

Usage:
    python src/NBA/process_props.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

IN_PATH  = Path("src/NBA/data/player_logs_raw.csv")
OUT_PATH = Path("src/NBA/data/player_features.csv")

STAT_COLS = ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "MIN", "TOV", "FGA", "FTA"]
WINDOWS   = [5, 10, 20]


def rolling_player(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    """Shift-1 rolling mean per player — no look-ahead."""
    return (
        df.groupby("PLAYER_ID")[col]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=max(1, window // 2)).mean())
    )


def build_opp_def_rating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling 10-game pts-allowed average for each team, attached to each game
    as OPP_DEF_RATING_L10.  We compute it on the team rows then merge onto
    player rows via GAME_ID + OPP_TEAM.
    """
    # One row per team-game with pts they allowed (OPP_PTS_ALLOWED already in data)
    team_game = (
        df[["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "OPP_PTS_ALLOWED"]]
        .drop_duplicates(subset=["GAME_ID", "TEAM_ABBREVIATION"])
        .sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
    )
    team_game["OPP_DEF_RATING_L10"] = (
        team_game.groupby("TEAM_ABBREVIATION")["OPP_PTS_ALLOWED"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    )
    # We want the OPPONENT's defensive rating: player on team A faces team B,
    # so we need team B's OPP_PTS_ALLOWED rolling avg.
    # Derive opp team from MATCHUP: if "CLE vs. TOR" opp = TOR, if "CLE @ TOR" opp = TOR
    return team_game[["GAME_ID", "TEAM_ABBREVIATION", "OPP_DEF_RATING_L10"]]


def game_in_series(df: pd.DataFrame) -> pd.Series:
    """
    For playoff rows, count which game in the series this is (1-7).
    Key = (SEASON, IS_PLAYOFF, sorted team pair from MATCHUP).
    For regular season rows returns 0.
    """
    result = pd.Series(0, index=df.index)
    po = df["IS_PLAYOFF"] == 1
    if po.sum() == 0:
        return result

    po_df = df[po].copy()
    # Normalize matchup to a canonical team-pair key
    def matchup_key(row):
        teams = sorted([row["TEAM_ABBREVIATION"],
                        row["MATCHUP"].split(" vs. ")[-1].split(" @ ")[-1].strip()])
        return f"{row['SEASON']}_{teams[0]}_{teams[1]}"

    po_df["SERIES_KEY"] = po_df.apply(matchup_key, axis=1)
    po_df = po_df.sort_values("GAME_DATE")
    po_df["GAME_IN_SERIES"] = po_df.groupby(["PLAYER_ID", "SERIES_KEY"]).cumcount() + 1
    result.loc[po_df.index] = po_df["GAME_IN_SERIES"].values
    return result


def main():
    print(f"Loading {IN_PATH} ...")
    df = pd.read_csv(IN_PATH, parse_dates=["GAME_DATE"])
    print(f"  {len(df):,} rows, {df['PLAYER_ID'].nunique():,} unique players")

    df.sort_values(["PLAYER_ID", "GAME_DATE"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Rolling stat features ─────────────────────────────────────────────────
    print("Building rolling features ...")
    for col in STAT_COLS:
        if col not in df.columns:
            continue
        for w in WINDOWS:
            df[f"{col}_L{w}"] = rolling_player(df, col, w)

    # ── Rest days ─────────────────────────────────────────────────────────────
    print("Computing rest days ...")
    df["REST_DAYS"] = (
        df.groupby("PLAYER_ID")["GAME_DATE"]
        .transform(lambda x: x.diff().dt.days)
        .fillna(3)          # assume 3 days rest for first game of season
        .clip(upper=14)     # cap at 14 (long breaks don't add info)
    )

    # ── Opponent defensive rating ─────────────────────────────────────────────
    print("Building opponent defensive ratings ...")
    opp_def = build_opp_def_rating(df)
    # Derive each player's opp team from MATCHUP
    df["OPP_TEAM"] = df["MATCHUP"].apply(
        lambda m: m.split(" @ ")[-1] if " @ " in m else m.split(" vs. ")[-1]
    ).str.strip()
    df = df.merge(
        opp_def.rename(columns={"TEAM_ABBREVIATION": "OPP_TEAM",
                                 "OPP_DEF_RATING_L10": "OPP_DEF_RATING_L10"}),
        on=["GAME_ID", "OPP_TEAM"],
        how="left",
    )

    # ── Game in series ────────────────────────────────────────────────────────
    print("Tagging game-in-series ...")
    df["GAME_IN_SERIES"] = game_in_series(df)

    # ── Season number (ordinal, for year-over-year drift) ─────────────────────
    season_order = {s: i for i, s in enumerate(sorted(df["SEASON"].unique()))}
    df["SEASON_NUM"] = df["SEASON"].map(season_order)

    # ── Drop rows without enough rolling history (first ~5 games per player) ──
    key_rolling = [f"PTS_L{WINDOWS[0]}", f"REB_L{WINDOWS[0]}", f"AST_L{WINDOWS[0]}"]
    before = len(df)
    df.dropna(subset=key_rolling, inplace=True)
    print(f"Dropped {before - len(df):,} rows lacking rolling history → {len(df):,} remain")

    df.sort_values(["PLAYER_ID", "GAME_DATE"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved → {OUT_PATH}")
    print(df[["PLAYER_NAME", "GAME_DATE", "PTS", "PTS_L10", "REB_L10",
              "AST_L10", "OPP_DEF_RATING_L10", "REST_DAYS"]].tail(5).to_string())


if __name__ == "__main__":
    main()
