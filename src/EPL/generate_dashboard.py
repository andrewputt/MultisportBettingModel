#!/usr/bin/env python3
"""
Generate a simple EPL dashboard from data/processed/epl/golden.csv.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT / "data" / "processed" / "epl"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

GOLDEN = PROCESSED_DIR / "golden.csv"
OUT = PROCESSED_DIR / "dashboard.html"


def pct(x):
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return ""


def main():
    if GOLDEN.exists():
        df = pd.read_csv(GOLDEN)
    else:
        df = pd.DataFrame()

    rows = df.to_dict(orient="records") if not df.empty else []
    rows_json = json.dumps(rows)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EPL EdgeModel</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50 text-slate-900">
  <main class="max-w-7xl mx-auto p-6">
    <div class="flex flex-col md:flex-row md:items-end md:justify-between gap-3 mb-6">
      <div>
        <p class="text-sm uppercase tracking-wide text-emerald-700 font-semibold">Premier League • England</p>
        <h1 class="text-4xl font-extrabold">EPL EdgeModel</h1>
        <p class="text-slate-600 mt-2">Kalshi market prices blended with API-Football recent form and head-to-head history.</p>
      </div>
      <div class="text-sm text-slate-500">Generated {generated}</div>
    </div>

    <section class="grid md:grid-cols-4 gap-4 mb-6" id="cards"></section>

    <section class="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
      <div class="p-4 flex items-center justify-between border-b border-slate-200">
        <h2 class="font-bold text-xl">Ranked EPL Markets</h2>
        <input id="search" class="border rounded-xl px-3 py-2 text-sm" placeholder="Search team..." />
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead class="bg-slate-100 text-slate-600">
            <tr>
              <th class="text-left p-3">Rank</th>
              <th class="text-left p-3">Matchup</th>
              <th class="text-left p-3">Prediction</th>
              <th class="text-right p-3">Kalshi Prob</th>
              <th class="text-right p-3">Model Prob</th>
              <th class="text-right p-3">Edge</th>
              <th class="text-right p-3">H2H Games</th>
              <th class="text-left p-3">Signal Notes</th>
            </tr>
          </thead>
          <tbody id="tbody"></tbody>
        </table>
      </div>
    </section>

    <p class="text-xs text-slate-500 mt-4">This is a class-project prediction tool, not financial advice. Kalshi markets are event contracts, not sportsbook odds.</p>
  </main>

<script>
const ROWS = {rows_json};
const tbody = document.getElementById('tbody');
const search = document.getElementById('search');
const cards = document.getElementById('cards');

function fmtPct(x) {{ return ((Number(x) || 0) * 100).toFixed(1) + '%'; }}
function fmtEdge(x) {{ return ((Number(x) || 0) * 100).toFixed(2) + '%'; }}

function renderCards(rows) {{
  const positive = rows.filter(r => Number(r.edge) > 0).length;
  const avgEdge = rows.length ? rows.reduce((a,r)=>a+Number(r.edge || 0),0)/rows.length : 0;
  const maxEdge = rows.length ? Math.max(...rows.map(r=>Number(r.edge || 0))) : 0;
  const cardsData = [
    ['Markets scanned', rows.length],
    ['Positive edges', positive],
    ['Average edge', fmtEdge(avgEdge)],
    ['Best edge', fmtEdge(maxEdge)],
  ];
  cards.innerHTML = cardsData.map(([label, value]) => `
    <div class="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
      <div class="text-slate-500 text-sm">${{label}}</div>
      <div class="text-3xl font-extrabold mt-2">${{value}}</div>
    </div>`).join('');
}}

function render() {{
  const q = search.value.toLowerCase();
  const filtered = ROWS.filter(r => (`${{r.team_a}} ${{r.team_b}} ${{r.market_title}}`).toLowerCase().includes(q));
  tbody.innerHTML = filtered.map(r => `
    <tr class="border-t border-slate-100 hover:bg-slate-50">
      <td class="p-3 font-semibold">#${{r.rank}}</td>
      <td class="p-3 font-bold">${{r.team_a}} vs ${{r.team_b}}</td>
      <td class="p-3">${{r.prediction}}</td>
      <td class="p-3 text-right tabular-nums">${{fmtPct(r.market_prob)}}</td>
      <td class="p-3 text-right tabular-nums font-semibold">${{fmtPct(r.model_prob)}}</td>
      <td class="p-3 text-right tabular-nums font-bold ${{Number(r.edge)>0?'text-emerald-700':'text-rose-700'}}">${{fmtEdge(r.edge)}}</td>
      <td class="p-3 text-right">${{r.h2h_games}}</td>
      <td class="p-3 text-slate-600">PPM: ${{r.team_a_ppm_l10}} vs ${{r.team_b_ppm_l10}} • GD: ${{r.team_a_gd_l10}} vs ${{r.team_b_gd_l10}}</td>
    </tr>`).join('') || `<tr><td colspan="8" class="p-6 text-center text-slate-500">No matched EPL markets yet. Run fetch/process again when Kalshi has EPL markets open.</td></tr>`;
}}

search.addEventListener('input', render);
renderCards(ROWS);
render();
</script>
</body>
</html>"""

    OUT.write_text(html, encoding="utf-8")
    print(f"Dashboard saved → {OUT}")


if __name__ == "__main__":
    main()
