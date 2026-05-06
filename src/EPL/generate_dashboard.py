import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT / "data" / "processed" / "epl"

TEMPLATE_PATH = ROOT / "dashboard.html"
OUTPUT_PATH = PROCESSED_DIR / "dashboard.html"
CSV_PATH = PROCESSED_DIR / "golden.csv"


def build_epl_rows(df):
    rows = []

    for i, r in df.iterrows():
        rows.append({
            "id": f"epl-{i+1}",
            "player": str(r["team_a"]),
            "team": str(r["team_a"]),
            "opp": f"vs {r['team_b']}",
            "gameTime": str(r.get("close_time", "")),
            "market": "Match Winner",
            "line": 0,
            "side": "YES",
            "book": float(r["market_prob"]) * 100,
            "model": float(r["model_prob"]) * 100,
            "edge": float(r["edge"]) * 100,
            "xgForm": float(r.get("team_a_ppm_l10", 0)),
            "cleanSheet": None
        })

    return rows


def main():
    if not CSV_PATH.exists():
        print("No golden.csv found")
        return

    if not TEMPLATE_PATH.exists():
        print("No dashboard.html template found in main folder")
        return

    df = pd.read_csv(CSV_PATH)
    epl_rows = build_epl_rows(df)
    epl_json = json.dumps(epl_rows, indent=2)

    html = TEMPLATE_PATH.read_text(encoding="utf-8")

    old = "const EPL_EDGES = [];"
    new = f"const EPL_EDGES = {epl_json};"

    if old not in html:
        print("Could not find: const EPL_EDGES = [];")
        return

    html = html.replace(old, new)

    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"✅ EPL dashboard generated → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
