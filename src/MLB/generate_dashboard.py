#!/usr/bin/env python3
"""
generate_dashboard.py
─────────────────────────────────────────────────────────────────────────────
Reads golden.csv + summary.json from data/processed/ and writes a
self-contained HTML dashboard with today's real data embedded.

Run after process_model.py:
  python3 src/MLB/generate_dashboard.py

The dashboard is written to:
  dashboard.html   (project root — open directly in any browser)
"""

import csv
import json
import re
import sys
from datetime import date
from pathlib import Path


ROOT          = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
GOLDEN_CSV    = PROCESSED_DIR / "golden.csv"
SUMMARY_JSON  = PROCESSED_DIR / "summary.json"
OUTPUT_HTML   = ROOT / "dashboard.html"

# Must match EV_THRESHOLD in process_model.py (expressed as a percentage, e.g. 10.0 = 10 %)
DASHBOARD_EV_THRESHOLD = 10.0

# Maximum rows injected into MLB_RAW in dashboard.html.
# Keeps the UI fast and the injected block readable.
TOP_N_ROWS = 50

# ── market label map ──────────────────────────────────────────────────────────
MARKET_LABELS = {
    "pitcher_strikeouts":    "Pitcher Ks",
    "batter_hits":           "Hits",
    "batter_total_bases":    "Total Bases",
    "pitcher_outs_recorded": "Pitcher Outs",
    "pitcher_outs":          "Pitcher Outs",   # process_model.py uses this key
}

# ── column name aliases (handle whatever process_model.py uses) ───────────────
def _get(row: dict, *keys, default=""):
    for k in keys:
        if k in row and row[k] not in ("", None):
            return row[k]
    return default

def _pct(val: str) -> float:
    """Parse '58.3%' or '0.583' or '58.3' → float percentage (0–100)."""
    if not val:
        return 0.0
    s = str(val).strip().rstrip("%")
    try:
        f = float(s)
        # if stored as 0–1 decimal, scale to percent
        return round(f * 100, 1) if f <= 1.0 else round(f, 1)
    except ValueError:
        return 0.0


def load_golden(path: Path) -> list[dict]:
    """Load ALL rows from golden.csv — no row cap.
    Rows come in sorted by edge desc (process_model.py's output order).
    We preserve that order and let the dashboard filter/tab on the client.
    Also sniff the first row to detect actual column names so the field
    mapping survives whatever names process_model.py uses.
    """
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols   = set(reader.fieldnames or [])

        # ── debug: print detected column names once ──────────────────
        print(f"  [generate_dashboard] golden.csv columns: {sorted(cols)}")

        for i, row in enumerate(reader):
            raw_market = _get(row, "market", "market_type", "prop_type")
            edge_raw   = _get(row, "edge", "edge_pct", "ev")
            rows.append({
                "rank":      i + 1,
                "player":    _get(row, "player", "player_name", "name"),
                "market":    MARKET_LABELS.get(raw_market, raw_market),
                "marketRaw": raw_market,
                "line":      _get(row, "line", "line_value", "prop_line"),
                "mktProb":   _pct(_get(row, "mkt_prob", "market_prob", "book_prob",
                                        "no_vig_over_prob")),
                "modelProb": _pct(_get(row, "model_prob", "predicted_prob")),
                "edge":      _pct(edge_raw),
                "homeTeam":  _get(row, "home_team", "home"),
                "awayTeam":  _get(row, "away_team", "away"),
                "commence":  _get(row, "commence_time", "game_time", "start_time"),
                "wind":      _get(row, "wind_cat", "wind_direction"),
                "windSpd":   _get(row, "wind_spd", "wind_speed"),
                # iso_adj is a blended team ISO stored in process_model.py
                "isoHome":   float(_get(row, "iso_home", "home_iso", "iso_adj") or 0),
                "isoAway":   float(_get(row, "iso_away", "away_iso", "iso_adj") or 0),
                "bookmaker": _get(row, "bookmaker", "book"),
            })
    return rows


def load_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_meta(rows: list[dict], summary: dict, run_date: str) -> dict:
    """Build the INJECTED_META object consumed by dashboard_template.html.

    Fields must match what the template reads from INJECTED_META:
      date, scanned, plays, avgEdge, signalStrength,
      regime_2025, regime_2026, pivotDate, threshold, ingestion
    """
    edges      = [r["edge"] for r in rows if r.get("edge", 0) > 0]
    plays_n    = sum(1 for r in rows if r.get("edge", 0) >= DASHBOARD_EV_THRESHOLD)
    avg_edge   = round(sum(edges) / len(edges), 1) if edges else 0.0
    scanned    = summary.get("total_props", summary.get("total_rows", len(rows)))

    # Signal strength: scale avg_edge to a rough 0-100 confidence gauge
    # (8% edge ≈ 80, 4% ≈ 60, capped at 95)
    signal     = min(95, max(40, int(avg_edge * 10)))

    return {
        "date":            run_date,
        "scanned":         scanned,
        "plays":           plays_n,
        "avgEdge":         avg_edge,
        "signalStrength":  signal,
        "regime_2025":     75,          # fixed until May 15 2026 pivot
        "regime_2026":     25,
        "pivotDate":       "May 15 2026",
        "threshold":       DASHBOARD_EV_THRESHOLD,
        "ingestion": [
            {"name": "Odds API",           "status": "live",   "latency": 110},
            {"name": "Kalshi Markets",      "status": "live" if summary.get("kalshi_markets", 0) > 0 else "stale", "latency": 64},
            {"name": "pybaseball / B-Ref",  "status": "live",   "latency": 0},
            {"name": "Open-Meteo Wx",       "status": "cached", "latency": 14400},
        ],
    }


PICKS_JSON = PROCESSED_DIR / "picks_history.json"

def save_picks(rows: list[dict], run_date: str) -> None:
    """Append today's top-5 picks (by edge desc) to picks_history.json.

    Schema per entry:
      {
        "date": "2026-04-30",
        "picks": [
          {
            "rank":      1,
            "player":    "Adrian Del Castillo",
            "market":    "Total Bases",
            "marketRaw": "batter_total_bases",
            "line":      "0.5",
            "side":      "OVER",
            "mktProb":   52.0,
            "modelProb": 53.2,
            "edge":      1.2,
            "result":    null   ← fill in manually: "WIN", "LOSS", or "PUSH"
          },
          ...
        ]
      }

    To record results later, run:
      python3 src/MLB/generate_dashboard.py --record 2026-04-30 WIN LOSS WIN WIN LOSS
    """
    # Sort rows by edge desc, take top 5
    top5 = sorted(rows, key=lambda r: r["edge"], reverse=True)[:5]
    picks = [
        {
            "rank":      i + 1,
            "player":    r["player"],
            "market":    r["market"],
            "marketRaw": r["marketRaw"],
            "line":      r["line"],
            "mktProb":   r["mktProb"],
            "modelProb": r["modelProb"],
            "edge":      r["edge"],
            "result":    None,
        }
        for i, r in enumerate(top5)
    ]

    history: dict = {}
    if PICKS_JSON.exists():
        try:
            with open(PICKS_JSON, encoding="utf-8") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            history = {}

    if run_date in history:
        # Preserve any results already recorded; only update pick fields if result is null
        existing = {p["rank"]: p for p in history[run_date].get("picks", [])}
        for p in picks:
            if p["rank"] in existing and existing[p["rank"]].get("result") is not None:
                p["result"] = existing[p["rank"]]["result"]

    history[run_date] = {"picks": picks}

    PICKS_JSON.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [picks_history] Top 5 saved → {PICKS_JSON}")
    print(f"    Record results: python3 src/MLB/generate_dashboard.py --record {run_date} WIN LOSS WIN WIN LOSS")


def record_results(target_date: str, results: list[str]) -> None:
    """Mark outcomes for a past date's top-5 picks.

    Usage:
      python3 src/MLB/generate_dashboard.py --record 2026-04-30 WIN LOSS WIN WIN PUSH
    """
    if not PICKS_JSON.exists():
        print(f"[record_results] No picks_history.json found at {PICKS_JSON}")
        return

    with open(PICKS_JSON, encoding="utf-8") as f:
        history = json.load(f)

    if target_date not in history:
        print(f"[record_results] No picks found for {target_date}")
        print(f"  Available dates: {', '.join(sorted(history.keys()))}")
        return

    picks = history[target_date]["picks"]
    valid = {"WIN", "LOSS", "PUSH", "N/A"}
    for i, pick in enumerate(picks):
        if i < len(results):
            r = results[i].upper()
            if r not in valid:
                print(f"  [warn] #{i+1} result '{r}' not in {valid}, skipping")
                continue
            pick["result"] = r
            print(f"  #{i+1} {pick['player']} {pick['market']} → {r}")

    # Compute summary stats
    recorded = [p["result"] for p in picks if p["result"] in {"WIN", "LOSS", "PUSH"}]
    if recorded:
        wins   = recorded.count("WIN")
        losses = recorded.count("LOSS")
        pushes = recorded.count("PUSH")
        record = f"{wins}-{losses}" + (f"-{pushes}" if pushes else "")
        history[target_date]["record"] = record
        print(f"  Record for {target_date}: {record}")

    PICKS_JSON.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved → {PICKS_JSON}")


def inject(template: str, rows: list[dict], meta: dict) -> str:
    """Replace the /*INJECT_..._START*/.../*INJECT_..._END*/ markers."""
    # One row per line for MLB props — compact but readable, no per-field line breaks.
    # NBA/EPL blocks are outside the injection markers and are never touched.
    rows_json = "[\n" + ",\n".join("  " + json.dumps(r) for r in rows) + "\n]"
    meta_json = json.dumps(meta, indent=2)
    out = re.sub(
        r"/\*INJECT_ROWS_START\*/.*?/\*INJECT_ROWS_END\*/",
        lambda _: f"/*INJECT_ROWS_START*/\n{rows_json}\n/*INJECT_ROWS_END*/",
        template,
        flags=re.DOTALL,
    )
    out = re.sub(
        r"/\*INJECT_META_START\*/.*?/\*INJECT_META_END\*/",
        lambda _: f"/*INJECT_META_START*/\n{meta_json}\n/*INJECT_META_END*/",
        out,
        flags=re.DOTALL,
    )
    return out


def _build_fallback_html(rows: list[dict], meta: dict, run_date: str) -> str:
    """
    Self-contained fallback dashboard written when dashboard_template.html
    is missing.  Shows Top 5 picks + full prop table with no external deps.
    """
    top5   = sorted(rows, key=lambda r: r["edge"], reverse=True)[:5]
    all_r  = sorted(rows, key=lambda r: r["edge"], reverse=True)

    def edge_color(e: float) -> str:
        if e >= 8:  return "#22c55e"
        if e >= 6:  return "#86efac"
        if e >= 4:  return "#fbbf24"
        return "#94a3b8"

    top5_rows = ""
    for i, r in enumerate(top5, 1):
        ec = edge_color(r["edge"])
        top5_rows += (
            f"<tr>"
            f"<td>{i}</td>"
            f"<td><strong>{r['player']}</strong></td>"
            f"<td>{r['market']}</td>"
            f"<td>{r['line']}</td>"
            f"<td>{r['mktProb']:.1f}%</td>"
            f"<td>{r['modelProb']:.1f}%</td>"
            f"<td style='color:{ec};font-weight:700'>+{r['edge']:.1f}%</td>"
            f"<td>{r['homeTeam']}</td>"
            f"</tr>\n"
        )

    all_rows = ""
    for r in all_r:
        ec = edge_color(r["edge"])
        all_rows += (
            f"<tr>"
            f"<td>{r['player']}</td>"
            f"<td>{r['market']}</td>"
            f"<td>{r['line']}</td>"
            f"<td>{r['mktProb']:.1f}%</td>"
            f"<td>{r['modelProb']:.1f}%</td>"
            f"<td style='color:{ec}'>+{r['edge']:.1f}%</td>"
            f"<td>{r['homeTeam']} vs {r['awayTeam']}</td>"
            f"</tr>\n"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EdgeModel — {run_date}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0f172a; color: #e2e8f0; font-family: system-ui, sans-serif;
         font-size: 14px; padding: 24px; }}
  h1   {{ font-size: 22px; font-weight: 700; color: #f8fafc; margin-bottom: 4px; }}
  .sub {{ color: #64748b; margin-bottom: 24px; font-size: 13px; }}
  .card {{ background: #1e293b; border-radius: 12px; padding: 20px;
           margin-bottom: 24px; border: 1px solid #334155; }}
  .card h2 {{ font-size: 15px; font-weight: 600; color: #94a3b8;
              text-transform: uppercase; letter-spacing: .05em; margin-bottom: 16px; }}
  .stats {{ display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 24px; }}
  .stat  {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px;
            padding: 16px 24px; min-width: 120px; }}
  .stat .label {{ font-size: 11px; color: #64748b; text-transform: uppercase;
                  letter-spacing: .08em; margin-bottom: 6px; }}
  .stat .value {{ font-size: 22px; font-weight: 700; color: #f8fafc; }}
  table  {{ width: 100%; border-collapse: collapse; }}
  th     {{ text-align: left; padding: 8px 12px; color: #64748b; font-size: 11px;
            text-transform: uppercase; letter-spacing: .06em;
            border-bottom: 1px solid #334155; }}
  td     {{ padding: 10px 12px; border-bottom: 1px solid #1e293b; }}
  tr:hover td {{ background: #1e293b; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 99px;
            font-size: 11px; font-weight: 600; background: #0f172a; }}
</style>
</head>
<body>
<h1>⚾ EdgeModel MLB Dashboard</h1>
<p class="sub">{run_date} &nbsp;·&nbsp; {meta['scanned']} props scanned
   &nbsp;·&nbsp; {meta['plays']} qualifying play(s) ≥{meta['threshold']}% EV
   &nbsp;·&nbsp; avg edge {meta['avgEdge']}%</p>

<div class="stats">
  <div class="stat"><div class="label">Props Scanned</div>
    <div class="value">{meta['scanned']}</div></div>
  <div class="stat"><div class="label">EV Plays</div>
    <div class="value">{meta['plays']}</div></div>
  <div class="stat"><div class="label">Avg Edge</div>
    <div class="value">{meta['avgEdge']}%</div></div>
  <div class="stat"><div class="label">Signal</div>
    <div class="value">{meta['signalStrength']}</div></div>
</div>

<div class="card">
  <h2>Top 5 Picks</h2>
  <table>
    <thead><tr>
      <th>#</th><th>Player</th><th>Market</th><th>Line</th>
      <th>Mkt%</th><th>Model%</th><th>Edge%</th><th>Home</th>
    </tr></thead>
    <tbody>{top5_rows}</tbody>
  </table>
</div>

<div class="card">
  <h2>All Props ({len(all_r)} rows)</h2>
  <table>
    <thead><tr>
      <th>Player</th><th>Market</th><th>Line</th>
      <th>Mkt%</th><th>Model%</th><th>Edge%</th><th>Matchup</th>
    </tr></thead>
    <tbody>{all_rows}</tbody>
  </table>
</div>
</body>
</html>"""


def main() -> None:
    # ── --record mode: mark outcomes for a past date, then exit ──────────────
    if "--record" in sys.argv:
        idx      = sys.argv.index("--record")
        target   = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        outcomes = sys.argv[idx + 2:idx + 7]
        if not target:
            sys.exit("Usage: generate_dashboard.py --record <YYYY-MM-DD> WIN|LOSS|PUSH ...")
        record_results(target, outcomes)
        return

    if not GOLDEN_CSV.exists():
        sys.exit(
            f"[generate_dashboard] ERROR: {GOLDEN_CSV} not found.\n"
            "Run process_model.py first."
        )

    rows     = load_golden(GOLDEN_CSV)
    summary  = load_summary(SUMMARY_JSON)
    run_date = date.today().isoformat()

    # ── Slice to top N by edge before anything else touches rows ─────────────
    rows = sorted(rows, key=lambda r: r["edge"], reverse=True)[:TOP_N_ROWS]

    # ── Always save picks first — independent of whether template exists ──────
    save_picks(rows, run_date)

    meta = build_meta(rows, summary, run_date)

    # ── Non-destructive write: inject MLB data into existing dashboard ─────────
    # If dashboard.html already exists, read it and only replace the two marker
    # blocks (/*INJECT_ROWS_START*/…/*INJECT_ROWS_END*/ and
    # /*INJECT_META_START*/…/*INJECT_META_END*/).  Everything else — including
    # NBA_EDGES, EPL_EDGES, and any other teammates' additions — is preserved.
    #
    # If the file doesn't exist yet (first run), build it from the fallback
    # template so it's immediately usable.
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_HTML.exists():
        existing = OUTPUT_HTML.read_text(encoding="utf-8")
        if "/*INJECT_ROWS_START*/" in existing and "/*INJECT_META_START*/" in existing:
            html = inject(existing, rows, meta)
            mode = "injected into existing"
        else:
            # File exists but has no markers (old fallback build) — rebuild so
            # markers are present for future non-destructive runs.
            html = _build_fallback_html(rows, meta, run_date)
            mode = "rebuilt (no markers found)"
    else:
        html = _build_fallback_html(rows, meta, run_date)
        mode = "created (first run)"

    OUTPUT_HTML.write_text(html, encoding="utf-8")

    plays = meta["plays"]
    by_market: dict[str, int] = {}
    for r in rows:
        by_market[r["marketRaw"]] = by_market.get(r["marketRaw"], 0) + 1
    market_str = "  ".join(f"{k}={v}" for k, v in sorted(by_market.items()))
    print(
        f"[generate_dashboard] {len(rows)} rows  |  {plays} play(s) ≥{DASHBOARD_EV_THRESHOLD:.0f}% EV"
        f"  |  avg edge {meta['avgEdge']}%  [{mode}]\n"
        f"  Markets: {market_str}\n"
        f"  → {OUTPUT_HTML}"
    )


if __name__ == "__main__":
    main()