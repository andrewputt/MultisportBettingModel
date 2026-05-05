"""
generate_nba_dashboard.py
────────────────────────────────────────────────────────────────────────────
Reads props_edges_today.csv and injects NBA_EDGES into dashboard_template.html,
saving the result as dashboard.html.

Usage:
    python src/NBA/generate_nba_dashboard.py
    python src/NBA/generate_nba_dashboard.py --min-edge 0.10
    python src/NBA/generate_nba_dashboard.py --out dashboard.html
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROPS_TODAY   = Path("src/NBA/data/props_edges_today.csv")
ODDS_CACHE    = Path("src/NBA/data/odds_cache.json")
TEMPLATE      = Path("dashboard_template.html")
DEFAULT_OUT   = Path("dashboard.html")

STAT_TO_MARKET = {
    "PTS":  "Points",
    "REB":  "Rebounds",
    "AST":  "Assists",
    "FG3M": "Threes",
    "STL":  "Steals",
    "BLK":  "Blocks",
    "DD":   "Double-Double",
    "TD":   "Triple-Double",
}

TEAM_FULL = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}


def build_game_times() -> dict[str, str]:
    """Read odds_cache.json to get tip-off times per game (away+home abb → ISO string)."""
    if not ODDS_CACHE.exists():
        return {}
    try:
        events = json.loads(ODDS_CACHE.read_text())
        times = {}
        from predict_props import TEAM_FULL_TO_ABB  # reuse the same map
        for ev in events:
            if not isinstance(ev, dict):
                continue
            home = TEAM_FULL_TO_ABB.get(ev.get("home_team", ""), "")
            away = TEAM_FULL_TO_ABB.get(ev.get("away_team", ""), "")
            tip  = ev.get("commence_time", "")
            if home and away and tip:
                times[f"{away}{home}"] = tip
        return times
    except Exception:
        return {}


def fatigue_obj(rest_days: float) -> dict:
    rest = int(round(rest_days))
    b2b  = rest <= 1
    label = "B2B" if b2b else f"{rest}d rest"
    return {"b2b": b2b, "restDays": rest, "label": label}


def steam_obj() -> dict:
    return {"moved": False, "dir": "UP", "label": "Flat"}


def to_js_obj(d: dict) -> str:
    """Serialize a Python dict to a JavaScript object literal (not JSON — no quotes on keys)."""
    parts = []
    for k, v in d.items():
        if isinstance(v, bool):
            parts.append(f"{k}: {'true' if v else 'false'}")
        elif isinstance(v, str):
            escaped = v.replace("\\", "\\\\").replace('"', '\\"')
            parts.append(f'{k}: "{escaped}"')
        elif isinstance(v, dict):
            parts.append(f"{k}: {to_js_obj(v)}")
        else:
            parts.append(f"{k}: {v}")
    return "{" + ", ".join(parts) + "}"


def build_nba_edges(df: pd.DataFrame, min_edge: float, game_times: dict) -> list[dict]:
    rows = []
    df = df[df["market_type"] == "PROP"].copy()  # props only for now
    df = df[df["edge"] >= min_edge].copy()
    df = df.sort_values("edge", ascending=False).reset_index(drop=True)
    df = df.head(15)  # show top 15 picks only

    for i, r in df.iterrows():
        stat      = r["stat"]
        direction = r["direction"]
        game_id   = r.get("game_id", "")

        # Derive away/home from game_id e.g. 26MAY04PHINYK
        teams_part = re.sub(r"^\d{2}[A-Z]{3}\d{2}", "", game_id)
        away_abb = teams_part[:3] if len(teams_part) == 6 else ""
        home_abb = teams_part[3:] if len(teams_part) == 6 else ""

        opp_abb  = r["opp"] if r.get("opp") else (home_abb if r["team"] == away_abb else away_abb)
        opp_str  = f"vs {opp_abb}" if r.get("is_home") else f"@ {opp_abb}"

        # Game time from cache
        cache_key = f"{away_abb}{home_abb}"
        game_time = game_times.get(cache_key, "")

        # Convert to dashboard units (0–100 %)
        book_pct  = round(float(r["yes_ask"])    * 100, 1)
        model_pct = round(float(r["model_prob"]) * 100, 1)
        edge_pct  = round(float(r["edge"])       * 100, 1)

        # Side: YES→OVER, NO→UNDER
        side = "OVER" if direction == "YES" else "UNDER"

        market = STAT_TO_MARKET.get(stat, stat)

        # Add .5 hook to whole-number lines (books always use .5 hooks)
        line_val = float(r["threshold"])
        if line_val == int(line_val):
            line_val += 0.5

        rank = len(rows)  # 0-indexed position after sorting
        row = {
            "id":        f"nba-{rank+1}",
            "player":    r["player"],
            "team":      r["team"],
            "opp":       opp_str,
            "gameTime":  game_time,
            "market":    market,
            "line":      line_val,
            "side":      side,
            "book":      book_pct,
            "model":     model_pct,
            "edge":      edge_pct,
            "highlight": rank < 5,
            "fatigue":   fatigue_obj(float(r.get("rest_days", 2))),
            "steam":     steam_obj(),
            "restDays":  int(round(float(r.get("rest_days", 2)))),
            "seriesGame": int(r.get("game_in_series", 0)),
        }
        rows.append(row)
    return rows


def rows_to_js(rows: list[dict]) -> str:
    if not rows:
        return "[]"
    parts = [f"    {to_js_obj(r)}" for r in rows]
    return "[\n" + ",\n".join(parts) + "\n  ]"


def inject_into_template(html: str, nba_js: str, today: str) -> str:
    # Replace the NBA_EDGES array
    html = re.sub(
        r"const NBA_EDGES\s*=\s*\[\];",
        f"const NBA_EDGES = {nba_js};",
        html,
    )
    # Update NBA metaLabel date
    html = re.sub(
        r'(NBA:.*?metaLabel:\s*")[^"]*(")',
        rf'\g<1>Player Props · {today}\2',
        html,
        flags=re.DOTALL,
    )
    # Update NBA markets to include Steals and Blocks if present
    html = re.sub(
        r'(NBA:.*?markets:\s*)\[[^\]]*\]',
        r'\1["All Markets", "Points", "Rebounds", "Assists", "Threes", "Steals", "Blocks"]',
        html,
        flags=re.DOTALL,
    )
    return html


def main(min_edge: float = 0.05, out_path: Path = DEFAULT_OUT):
    if not PROPS_TODAY.exists():
        print(f"No props file found at {PROPS_TODAY}.")
        print("Run predict_props.py first, then re-run this script.")
        return

    if not TEMPLATE.exists():
        print(f"Template not found at {TEMPLATE}.")
        return

    df = pd.read_csv(PROPS_TODAY)
    print(f"Loaded {len(df)} prop edges from {PROPS_TODAY}")

    game_times = build_game_times()

    rows = build_nba_edges(df, min_edge, game_times)
    print(f"Built {len(rows)} NBA rows (edge >= {min_edge*100:.0f}%)")

    nba_js = rows_to_js(rows)
    today  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    html   = TEMPLATE.read_text(encoding="utf-8")
    html   = inject_into_template(html, nba_js, today)

    out_path.write_text(html, encoding="utf-8")
    print(f"Dashboard saved → {out_path}")
    print(f"Open in browser: open {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-edge", type=float, default=0.05,
                        help="Minimum edge to include in dashboard (default 5%%)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Output HTML file path (default dashboard.html)")
    args = parser.parse_args()
    main(args.min_edge, args.out)
