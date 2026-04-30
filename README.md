# Multisport Betting Model (2026 Season)

## Overview

An integrated quantitative pipeline for identifying value edges across **NBA**, **MLB**, and **EPL** markets. The system is designed to fetch live market data, engineer predictive features from historical logs, and execute backtests to surface the **Top 5 most confident plays** across all sports daily.

---

## Project Architecture

The project is split into sport-specific modules under `src/`, sharing a common data structure in `data/`.

### NBA Module (`src/NBA`)

- **Model**: Logistic Regression win-prediction model.
- **Current Accuracy**: **66.06%** (via walk-forward testing).
- **Key Features**: Home/Away win %, Rest days, Rolling L10 offensive/defensive ratings, and Head-to-Head opponent metrics.

### MLB Module (`src/MLB`)

- **Live Market Data**: Triple-stream ingestion pipeline (`fetch_all_data.py`) pulling from three independent sources:
  - **Kalshi**: Player prop markets with implied probability from `yes_ask` fields, filtered to today's games only.
  - **Odds API**: Four prop markets — `pitcher_strikeouts`, `batter_hits`, `batter_total_bases`, `pitcher_outs_recorded` — with credit-efficient date filtering (~10–15 games/day).
  - **pybaseball / Baseball-Reference**: Full 2025–2026 head-to-head game logs for all 30 MLB teams.
- **Cloudflare Bypass**: Global `curl_cffi` session with `impersonate="chrome110"` intercepts all Baseball-Reference requests to bypass bot detection.
- **Feature Engineering** (`process_model.py`): Isolated Power (ISO), park/wind multipliers, Batter vs. Pitcher (BvP) historical edge, no-vig market probability removal, and Kalshi signal blending.
- **Output**: Ranked `golden.csv` (366 daily props) with model probability, market probability, and calibrated edge %.
- **Dashboard** (`dashboard_template.html`): Self-contained React dashboard (EdgeModel) — Top 5 default view, market tabs (Pitcher Ks, Hits, Total Bases, Pitcher Outs), detail panel with signal breakdown, and system health footer. NBA/EPL stubs included for groupmates.
- **Picks Tracker**: `generate_dashboard.py` saves the daily Top 5 to `picks_history.json` on every run. Record outcomes with:
  ```bash
  python3 src/MLB/generate_dashboard.py --record YYYY-MM-DD WIN LOSS WIN WIN PUSH
  ```

**Daily workflow:**

```bash
python3 src/MLB/fetch_all_data.py   # refresh raw data (Odds API, Kalshi, pybaseball)
python3 src/MLB/process_model.py    # run model → generates dashboard + saves picks
open data/processed/dashboard.html  # view in browser
```

### EPL Module (`src/EPL`)

- **Historical Data**: Utilizing multi-year historical H2H (Head-to-Head) match data logs.
- **Status**: In development (Standardizing historical archives for model training).

---

## 📊 Current Progress

The infrastructure is currently active with local `.env` and `.gitignore` systems established to protect secrets and manage large datasets. The MLB data pipeline is **complete and fully operational** — pulling 366 daily player props across four markets, engineering predictive features from 4,900+ historical games, and generating a live EdgeModel dashboard with daily picks tracking. For the NBA module, backtesting is complete, showing a validated model with an average accuracy of 66.06% and a simulated monthly ROI reaching up to 38.24%. EPL integration is currently in progress, focusing on standardizing several years of historical match archives to align with the multisport master backtester.
