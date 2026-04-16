# Multisport Betting Model (2026 Season)

## 🚀 Overview

An integrated quantitative pipeline for identifying value edges across **NBA**, **MLB**, and **EPL** markets. The system is designed to fetch live market data, engineer predictive features from historical logs, and execute backtests to surface the **Top 5 most confident plays** across all sports daily.

---

## 🛠 Project Architecture

The project is split into sport-specific modules under `src/`, sharing a common data structure in `data/`.

### 🏀 NBA Module (`src/NBA`)

- **Model**: Logistic Regression win-prediction model.
- **Current Accuracy**: **66.06%** (via walk-forward testing).
- **Key Features**: Home/Away win %, Rest days, Rolling L10 offensive/defensive ratings, and Head-to-Head opponent metrics.

### ⚾ MLB Module (`src/MLB`)

- **Live Market Data**: Custom `kalshiAPI.py` fetches live Moneyline, Spreads, and Run Totals directly from Kalshi.
- **Data Ingestion**: Triple-stream pipeline (`fetch_all_data.py`) pulls Kalshi candles, Odds API player props, and `pybaseball` logs.
- **Cloudflare Bypass**: Implemented a global `smart_get` interceptor using `curl_cffi` to bypass Baseball-Reference bot detection for historical H2H data.

### ⚽ EPL Module (`src/EPL`)

- **Historical Data**: Utilizing multi-year historical H2H (Head-to-Head) match data logs.
- **Status**: In development (Standardizing historical archives for model training).

---

## 📊 Current Progress

The infrastructure is currently active with local `.env` and `.gitignore` systems established to protect secrets and manage large datasets. The MLB data pipeline is completed and fully operational, successfully pulling over 190 daily markets and all required historical logs for the 2025 and 2026 seasons. For the NBA module, backtesting is complete, showing a validated model with an average accuracy of 66.06% and a simulated monthly ROI reaching up to 38.24%. EPL integration is currently in progress, focusing on standardizing several years of historical match archives to align with the multisport master backtester.
