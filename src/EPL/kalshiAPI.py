"""
Small helper for checking open Kalshi markets that look related to EPL.
Run:
  python3 src/EPL/kalshiAPI.py
"""

from fetch_all_data import fetch_kalshi, RAW_DIR

if __name__ == "__main__":
    markets = fetch_kalshi(RAW_DIR / "kalshi_epl_markets.json")
    for m in markets[:25]:
        print(f"{m.get('ticker')} | {m.get('title')} | YES≈{m.get('yes_prob')}")
